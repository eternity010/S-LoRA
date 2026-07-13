from unittest.mock import Mock

import pytest

from slora.server.router.model_infer.model_rpc import ModelRpcClient


def _make_client(use_rpc, offload_adapters):
    client = ModelRpcClient.__new__(ModelRpcClient)
    client.use_rpc = use_rpc
    client._offload_adapters = offload_adapters
    return client


@pytest.mark.asyncio
async def test_clear_all_adapters_waits_for_rpc_completion():
    result = Mock()
    result.value = "cleared"
    offload_adapters = Mock(return_value=result)
    client = _make_client(True, offload_adapters)

    assert await client.clear_all_adapters() == "cleared"

    offload_adapters.assert_called_once_with([])
    result.wait.assert_called_once_with()


@pytest.mark.asyncio
async def test_clear_all_adapters_calls_local_model_synchronously():
    offload_adapters = Mock(return_value=None)
    client = _make_client(False, offload_adapters)

    assert await client.clear_all_adapters() is None
    offload_adapters.assert_called_once_with([])


@pytest.mark.asyncio
async def test_clear_all_adapters_propagates_rpc_failure():
    result = Mock()
    result.wait.side_effect = RuntimeError("remote reset failed")
    client = _make_client(True, Mock(return_value=result))

    with pytest.raises(RuntimeError, match="remote reset failed"):
        await client.clear_all_adapters()
