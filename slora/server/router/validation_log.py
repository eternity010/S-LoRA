"""Opt-in, per-worker evidence for GPU correctness checks."""

import json
import time
from pathlib import Path


class ValidationLog:
    def __init__(self, directory, worker_id):
        self.path = Path(directory) / f'worker_{worker_id}.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.worker_id = worker_id
        self.sequence = 0

    def __call__(self, event, **fields):
        self.sequence += 1
        record = dict(fields, event=event, worker_id=self.worker_id,
                      sequence=self.sequence, timestamp=time.time())
        # Each worker owns a different file; closing flushes phase-end evidence.
        with self.path.open('a') as handle:
            handle.write(json.dumps(record) + '\n')
