"""Four-process coordination check; no robot simulation or policy training."""
import json
import multiprocessing as mp
from pathlib import Path
from sgs.shared_sampler import SamplerManager


def report(shared, worker):
    for _ in range(25):
        shared.record(worker, bool(worker % 2))
    p, version = shared.distribution()
    assert len(p) == 32 and abs(sum(p)-1) < 1e-12 and version >= 25


def main():
    root = Path(__file__).parent.parent/'lift_uniform_baseline'/'uniform_run_20261026_205152'
    settings = json.loads((root/'settings.json').read_text())
    ctx = mp.get_context('spawn')
    with SamplerManager(ctx=ctx) as manager:
        for mode, kappa in (('uniform', 1), ('sgs', 1), ('sgs', 5)):
            shared = manager.SharedSampler(mode, str(root/'training_episodes.jsonl'),
                settings['pool_sha256'], settings['actual_target_steps'], kappa, 2)
            before = shared.snapshot()
            workers = [ctx.Process(target=report, args=(shared,i)) for i in range(4)]
            for worker in workers: worker.start()
            for worker in workers:
                worker.join(timeout=30)
                if worker.is_alive():
                    worker.terminate(); worker.join()
                    raise RuntimeError('Worker timed out')
                assert worker.exitcode == 0
            after = shared.snapshot()
            assert after['continuation_episodes'] == 100
            assert sum(after['total_visits']) == 1100
            for i in range(32):
                assert after['total_visits'][i]-before['total_visits'][i] == (25 if i < 4 else 0)
            if mode == 'uniform':
                assert after['sampling_probabilities'] == [1/32]*32
            else:
                assert after['sampling_probabilities'] != before['sampling_probabilities']
            assert after['kappa'] == kappa
            print(f'PASS {mode} kappa={kappa}: all 100 reports from four processes retained; probabilities correct')
    print('No simulation or PPO training performed.')


if __name__ == '__main__':
    main()
