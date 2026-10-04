"""Execute E6's independent seed chains in dependency order.

The plan is reviewable before running; no checkpoint is silently reused.
Use --dry-run to print all jobs without launching training.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.udr.prepare_e6_multiseed import digest, merge_seed


def schedule(plan):
    ordered = []
    for run in plan['runs']:
        for job in run['jobs']:
            if job['name'] == 'e4_a':
                ordered.append(dict(seed=run['seed'], action='merge',
                                    output=run['checkpoints']['e4_initial']))
            ordered.append(dict(seed=run['seed'], action='train',
                                config=job['config'], output=job['output_checkpoint'],
                                name=job['name']))
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(ROOT/'experiments/E6_plan/E6_plan.json'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-completed', action='store_true',
                        help='Resume after interruption by reusing existing scheduled outputs.')
    args = parser.parse_args()
    path = Path(args.plan).resolve()
    if not path.is_file():
        parser.error('Generate E6_plan.json with prepare_e6_multiseed.py first.')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if digest(plan['e4_selection_file']) != plan['e4_selection_sha256']:
        raise ValueError('E4 selection changed after E6 plan generation.')
    for step in schedule(plan):
        destination = Path(step['output'])
        if args.dry_run:
            print(f"seed {step['seed']} {step['action']} {step.get('name', '')} "
                  f"{step.get('config', '')} -> {destination}")
            continue
        if destination.exists():
            if not args.skip_completed:
                raise FileExistsError(f'E6 refuses to reuse an existing seed checkpoint: {destination}')
            if step['action'] == 'merge':
                record = destination.with_suffix('.json')
                if not record.is_file() or json.loads(record.read_text(encoding='utf-8'))['e4_initial_sha256'] != digest(destination):
                    raise ValueError(f'E6 existing merged checkpoint lacks valid lineage: {destination}')
            print('E6 skipped completed scheduled output:', destination, flush=True)
            continue
        if step['action'] == 'merge':
            merge_seed(SimpleNamespace(output=str(path.parent), seed=step['seed']))
        else:
            subprocess.run([sys.executable, str(ROOT/'basicsr/train.py'),
                            '-opt', step['config']], cwd=ROOT, check=True)
        if not destination.is_file():
            raise FileNotFoundError(f'E6 job did not produce its scheduled checkpoint: {destination}')
    print('Completed every independent E6 seed chain.')


if __name__ == '__main__':
    main()
