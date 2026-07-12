import json

from slora.server.router.response_merger import ResponseMerger


def test_record_reset_cache_response_writes_acknowledgement():
    response = {
        "type": "reset_cache_response",
        "request_id": "pytest_reset_ack",
        "worker_id": 2,
        "success": True,
    }
    ack_file = "/tmp/slora_reset_cache_pytest_reset_ack.jsonl"

    try:
        ResponseMerger._record_reset_cache_response(response)

        with open(ack_file) as f:
            saved = json.loads(f.read())
        assert saved == response
    finally:
        try:
            import os
            os.remove(ack_file)
        except FileNotFoundError:
            pass
