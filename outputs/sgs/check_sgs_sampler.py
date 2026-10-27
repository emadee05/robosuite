"""Validate SGS mathematics and warmup history without loading PPO or MuJoCo."""
import hashlib
import json
import math
from pathlib import Path
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
    root = Path(__file__).parent.parent / 'lift_uniform_baseline' / 'uniform_run_20261026_205152'
    settings = json.loads((root/'settings.json').read_text())
    assert hashlib.sha256((root/'train_pool.json').read_bytes()).hexdigest() == settings['pool_sha256']
    sampler = SGSSampler()
    sampler.load_training_history(root/'training_episodes.jsonl',
        pool_sha256=settings['pool_sha256'], through_transition=1001472)
    results = json.loads((root/'results.json').read_text())
    assert sampler.total_visits == results['visits_by_configuration']
    assert sampler.total_successes == results['training_successes_by_configuration']
    assert sum(sampler.total_visits) == 1000 and sum(sampler.total_successes) == 218
    output = root/'sgs_initial_sampler.json'
    output.write_text(json.dumps(sampler.snapshot(), indent=2)+'\n')
    p = sampler.probabilities()
    print('PASS: initialized from 1,000 training episodes, including 218 successes')
    print(f'Initial SGS probabilities: {min(p):.2%}–{max(p):.2%}; uniform: {1/32:.2%}')
    print('Saved:', output.resolve())
    print('No simulation or training performed. Next: shared sampler and matched continuation.')


if __name__ == '__main__':
    main()
