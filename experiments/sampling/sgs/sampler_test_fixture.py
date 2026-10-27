"""Small synthetic training history; no robot results or checkpoints required."""
import json
from pathlib import Path

POOL_HASH = 'synthetic-test-pool'
CHECKPOINT_STEPS = 32


def write_history(folder):
    path = Path(folder) / 'episodes.jsonl'
    episodes = [dict(pool_sha256=POOL_HASH, training_transitions=i + 1,
                     configuration_id=i, success=bool(i % 2)) for i in range(32)]
    # This episode occurs after the checkpoint and must be excluded.
    episodes.append(dict(pool_sha256=POOL_HASH, training_transitions=33,
                         configuration_id=0, success=True))
    path.write_text(''.join(json.dumps(e) + '\n' for e in episodes))
    return path
