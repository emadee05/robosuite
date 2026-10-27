"""Validate SGS mathematics and warmup history without loading PPO or MuJoCo."""
import json
import math
from tempfile import TemporaryDirectory
from sgs.sampler_test_fixture import write_history, POOL_HASH, CHECKPOINT_STEPS
from sgs.sgs_sampler import SGSSampler


def main():
    sampler = SGSSampler()
    assert all(p == 1/32 for p in sampler.probabilities())
    print('PASS: no history gives uniform sampling')
    sampler = SGSSampler(configurations=3)
    for _ in range(50):
        sampler.update(0, False)
        sampler.update(1, False)
        sampler.update(1, True)
        sampler.update(2, True)
    p = sampler.probabilities()
    assert math.isclose(p[0], p[2]) and p[1] > p[0] > 0
    assert math.isclose(sum(p), 1)
    print('PASS: intermediate success favored; zero and perfect success retain probability')
    for _ in range(100):
        sampler.update(1, True)
    assert len(sampler.history[1]) == 100 and all(sampler.history[1])
    print('PASS: only the latest 100 outcomes enter each success estimate')
    with TemporaryDirectory() as folder:
        history = write_history(folder)
        sampler = SGSSampler()
        sampler.load_training_history(history, pool_sha256=POOL_HASH,
                                      through_transition=CHECKPOINT_STEPS)
        assert sampler.total_visits == [1] * 32
        assert sampler.total_successes == [i % 2 for i in range(32)]
        assert len(sampler.history[0]) == 1  # Post-checkpoint episode excluded.
        try:
            SGSSampler().load_training_history(history, pool_sha256='wrong-pool',
                                               through_transition=CHECKPOINT_STEPS)
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched pool must be rejected')
    print('PASS: training history, checkpoint cutoff, and pool mismatch validation')
    print('No simulation or training performed.')


if __name__ == '__main__':
    main()
