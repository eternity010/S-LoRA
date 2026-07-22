# vLLM Replicated Multi-LoRA Baseline

This baseline runs three independent vLLM 0.4.0 Multi-LoRA servers and routes
requests with client-side round robin. Each invocation is one fresh-server run.

## Validated Environment

- Conda environment: `/home/hzheng/.conda/envs/vllm-cu118`
- vLLM: `0.4.0+cu118`
- PyTorch CUDA runtime: `12.1`
- Base model: `/home/hzheng/models/llama-7b`
- LoRA ranks: 16 and 64

The smoke test successfully loaded both adapters and completed requests through
the OpenAI completions API.

## Common Trace

Cross-system experiments use:

```text
real_workload/outputs/azure_llm_http_top100_6rps_180s_modelctx2048_v1.jsonl
```

The trace enforces `input_len + output_len <= 2046`. The existing dummy prompt
adds two tokenizer tokens, so the actual prompt plus generated tokens fits the
Llama-7B `max_model_len=2048` limit.

## Run 6 RPS

Run 1:

```bash
cd /home/hzheng/S-LoRA/benchmarks
conda activate slora

python external_baselines/vllm/run_replicated_realtrace.py \
  --trace-file real_workload/outputs/azure_llm_http_top100_6rps_180s_modelctx2048_v1.jsonl \
  --output-dir routing_comparison_results/post_state_fix_v1/08_external_baselines/vllm_6rps \
  --run-label run1
```

Run 2 repeats the command with `--run-label run2`. The runner starts fresh
servers and appends the second record to the same `results.jsonl`.

## Matching RankFlow Runs

Run each command twice with separate output directories (`run1` and `run2`):

```bash
python run_routing_comparison.py \
  --suite dp-realtrace-roundrobin-6rps-modelctx2048 \
  --output-dir routing_comparison_results/post_state_fix_v1/08_external_baselines/rlora_rr_6rps_run1

python run_routing_comparison.py \
  --suite dp-realtrace-rwpt-active-6rps-modelctx2048 \
  --output-dir routing_comparison_results/post_state_fix_v1/08_external_baselines/rlora_rwpt_active_6rps_run1
```

The second repetitions use the same commands with output directories ending in
`run2`. This keeps every cross-system repetition on a fresh server.
