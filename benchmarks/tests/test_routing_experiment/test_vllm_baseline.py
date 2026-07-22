import json
from pathlib import Path
import signal
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from external_baselines.vllm.run_replicated_realtrace import (
    adapter_modules,
    build_server_command,
    ensure_ports_available,
    get_vllm_version,
    load_trace,
    start_servers,
    stop_servers,
    summarize,
    wait_for_process_group_exit,
)


def test_load_trace_requires_total_tokens_within_model_limit(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({
            "req_id": 0,
            "req_time": 0.0,
            "adapter_id": 1,
            "input_len": 1848,
            "output_len": 200,
        }) + "\n"
    )

    with pytest.raises(ValueError, match="total tokens exceed"):
        load_trace(trace, num_adapters=2, max_model_len=2048)


def test_adapter_modules_alternate_rank_paths():
    assert adapter_modules(4, "/rank16", "/rank64") == [
        "adapter-0=/rank16",
        "adapter-1=/rank64",
        "adapter-2=/rank16",
        "adapter-3=/rank64",
    ]


def test_server_command_enables_rank64_multilora():
    args = SimpleNamespace(
        vllm_python="/env/bin/python",
        model="/model",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        max_num_seqs=64,
        max_loras=16,
        max_lora_rank=64,
        num_adapters=100,
        rank16_adapter="/rank16",
        rank64_adapter="/rank64",
        disable_frontend_multiprocessing=True,
    )

    command = build_server_command(args, port=38100)

    assert "--enable-lora" in command
    assert "--disable-log-requests" in command
    assert command[command.index("--max-lora-rank") + 1] == "64"
    assert command[command.index("--max-cpu-loras") + 1] == "100"
    lora_start = command.index("--lora-modules") + 1
    lora_end = command.index("--disable-frontend-multiprocessing")
    assert len(command[lora_start:lora_end]) == 100
    assert command[-1] == "--disable-frontend-multiprocessing"


def test_get_vllm_version_uses_server_python(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(stdout="0.6.3.post1\n")

    monkeypatch.setattr(
        "external_baselines.vllm.run_replicated_realtrace.subprocess.run",
        fake_run,
    )

    assert get_vllm_version("/env/bin/python") == "0.6.3.post1"
    assert observed["command"] == [
        "/env/bin/python",
        "-c",
        "import vllm; print(vllm.__version__)",
    ]
    assert observed["kwargs"]["check"] is True


def test_summarize_uses_successful_requests_only():
    results = [
        {"success": True, "total_latency": 2.0, "ttft": 0.5},
        {"success": True, "total_latency": 4.0, "ttft": 1.5},
        {"success": False, "total_latency": None, "ttft": None},
    ]

    summary = summarize(results, benchmark_time=5.0, req_rate=1.0)

    assert summary["completed_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["throughput"] == pytest.approx(0.4)
    assert summary["avg_latency"] == pytest.approx(3.0)
    assert summary["avg_first_token_latency"] == pytest.approx(1.0)


def test_start_servers_scopes_logs_by_run_label(tmp_path, monkeypatch):
    args = SimpleNamespace(
        output_dir=tmp_path,
        run_label="run2",
        startup_timeout=1.0,
    )
    opened_commands = []

    class FakeProcess:
        pid = 123

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        opened_commands.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(
        "external_baselines.vllm.run_replicated_realtrace.build_server_command",
        lambda args, port: ["server", str(port)],
    )
    monkeypatch.setattr(
        "external_baselines.vllm.run_replicated_realtrace.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "external_baselines.vllm.run_replicated_realtrace.wait_for_server",
        lambda process, port, timeout: None,
    )
    monkeypatch.setattr(
        "external_baselines.vllm.run_replicated_realtrace.ensure_ports_available",
        lambda ports: None,
    )

    processes = start_servers(args, [1], [38100])
    try:
        assert processes[0][2].name == "server_run2_replica0_gpu1.log"
        assert opened_commands[0][0] == ["server", "38100"]
    finally:
        for _, log_handle, _ in processes:
            log_handle.close()


def test_stop_servers_force_kills_residual_process_group(tmp_path, monkeypatch):
    log_handle = (tmp_path / "server.log").open("w")

    class FakeProcess:
        pid = 123

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    signals = []

    def fake_killpg(pid, sig):
        signals.append((pid, sig))

    monkeypatch.setattr(
        "external_baselines.vllm.run_replicated_realtrace.os.killpg",
        fake_killpg,
    )
    monkeypatch.setattr(
        "external_baselines.vllm.run_replicated_realtrace.wait_for_process_group_exit",
        lambda process_group_id, timeout: False,
    )

    stop_servers([(FakeProcess(), log_handle, tmp_path / "server.log")])

    assert signals == [(123, signal.SIGKILL)]
    assert log_handle.closed


def test_wait_for_process_group_exit_detects_missing_group(monkeypatch):
    def missing_group(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(
        "external_baselines.vllm.run_replicated_realtrace.os.killpg",
        missing_group,
    )

    assert wait_for_process_group_exit(123, timeout=1.0)


def test_ensure_ports_available_rejects_listener():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        with pytest.raises(RuntimeError, match=f"Port {port} is already in use"):
            ensure_ports_available([port])
