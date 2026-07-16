import json

import pytest

from slora.server.api_server import update_routing_config


class StubRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


@pytest.mark.asyncio
async def test_update_routing_config_accepts_rwpt_active(tmp_path, monkeypatch):
    config_file = tmp_path / "slora_routing_config_update.json"
    real_open = open

    def redirect_config_file(path, *args, **kwargs):
        if path == "/tmp/slora_routing_config_update.json":
            path = config_file
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", redirect_config_file)

    response = await update_routing_config(StubRequest({
        "load_metric": "rwpt_active",
        "reset_stats": True,
        "update_id": "test-update",
    }))

    assert response.status_code == 200
    assert json.loads(response.body)["config"]["load_metric"] == "rwpt_active"
    assert json.loads(config_file.read_text())["load_metric"] == "rwpt_active"


@pytest.mark.asyncio
async def test_update_routing_config_rejects_unknown_metric():
    response = await update_routing_config(StubRequest({
        "load_metric": "unknown",
    }))

    assert response.status_code == 400
    assert "rwpt_active" in json.loads(response.body)["message"]
