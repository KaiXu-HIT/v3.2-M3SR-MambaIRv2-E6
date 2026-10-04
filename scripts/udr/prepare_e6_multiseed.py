"""Generate independent E6 training chains and merge each seed's E4 start.

One selected E2 uncertainty *method* and E3 alpha setting are held fixed, but
RGB, UDR-v1, E2, E3 and E4 weights are retrained separately for every seed.
This addresses initialization/training fluctuation; a single E4 checkpoint
evaluated with three Gumbel seeds cannot do that.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.udr.prepare_e4_integration import (MODES, TAGS, merge_parameters,
                                                parameters)


def read_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'E6 source configuration missing: {path}')
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def read_selection(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('E6 needs measured E4 selection: ' + str(path))
    selection = json.loads(path.read_text(encoding='utf-8'))
    u = selection['selected_uncertainty']['variant']
    a = selection['selected_alpha']['tag']
    if u not in MODES or a not in TAGS:
        raise ValueError('Invalid selected E2/E3 variant in E4 manifest.')
    return selection, u, a


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def checkpoint(name, step):
    return str((ROOT/'experiments'/name/'models'/f'net_g_{step}.pth').resolve())


def configure(source, name, seed, pretrain=None, kind=None, teacher=None):
    config = deepcopy(source)
    config['name'] = name
    config['manual_seed'] = seed
    if pretrain is not None:
        config['path']['pretrain_network_g'] = pretrain
    if kind is not None:
        config['path']['pretrain_kind'] = kind
    config['path']['resume_state'] = None
    if teacher is not None:
        config['train']['rgb_teacher_checkpoint'] = teacher
    return config


def make_seed_configs(seed, selected_u, selected_a, sources, folder):
    """Keep each template's architecture, data, optimizer and iteration budget."""
    label = f'E6_s{seed}'
    names = dict(rgb=f'{label}_RGB_x4', v1_a=f'{label}_UDRv1_phaseA_x4',
                 v1_b=f'{label}_UDRv1_phaseB_x4',
                 e2_a=f'{label}_E2_{selected_u}_phaseA_x4',
                 e2_b=f'{label}_E2_{selected_u}_phaseB_x4',
                 e3_a=f'{label}_E3_{selected_a}_phaseA_x4',
                 e3_b=f'{label}_E3_{selected_a}_phaseB_x4',
                 e4_a=f'{label}_UDRv2_phaseA_x4',
                 e4_b=f'{label}_UDRv2_phaseB_x4')
    paths = dict(rgb=checkpoint(names['rgb'], 490000),
                 v1_a=checkpoint(names['v1_a'], 100000),
                 v1_b=checkpoint(names['v1_b'], 100000),
                 e2_a=checkpoint(names['e2_a'], 30000) if selected_u == 'U3' else None,
                 e2_b=checkpoint(names['e2_b'], 100000),
                 e3_a=checkpoint(names['e3_a'], 30000),
                 e3_b=checkpoint(names['e3_b'], 100000),
                 e4_initial=str((folder.parent/f'E4_initial_s{seed}.pth').resolve()),
                 e4_a=checkpoint(names['e4_a'], 30000),
                 e4_b=checkpoint(names['e4_b'], 100000))
    jobs = []

    def add(key, config):
        path = folder/f's{seed}_{key}.yml'
        path.write_text('# E6 independent seed chain; original dataset paths are preserved.\n'
                        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                        encoding='utf-8')
        jobs.append(dict(name=key, config=str(path.resolve()),
                         output_checkpoint=paths[key]))

    add('rgb', configure(sources['rgb'], names['rgb'], seed))
    add('v1_a', configure(sources['v1_a'], names['v1_a'], seed,
                          paths['rgb'], 'rgb'))
    add('v1_b', configure(sources['v1_b'], names['v1_b'], seed,
                          paths['v1_a'], 'udr'))
    if selected_u == 'U3':
        add('e2_a', configure(sources['e2_a'], names['e2_a'], seed,
                              paths['v1_b'], 'e0', paths['rgb']))
        e2_start, e2_kind = paths['e2_a'], 'e2'
    else:
        e2_start, e2_kind = paths['v1_b'], 'e0'
    add('e2_b', configure(sources['e2_b'], names['e2_b'], seed,
                          e2_start, e2_kind, paths['rgb']))
    add('e3_a', configure(sources['e3_a'], names['e3_a'], seed,
                          paths['v1_b'], 'e0'))
    add('e3_b', configure(sources['e3_b'], names['e3_b'], seed,
                          paths['e3_a'], 'e3'))
    add('e4_a', configure(sources['e4_a'], names['e4_a'], seed,
                          paths['e4_initial'], 'e4',
                          paths['rgb'] if selected_u == 'U3' else None))
    add('e4_b', configure(sources['e4_b'], names['e4_b'], seed,
                          paths['e4_a'], 'e4',
                          paths['rgb'] if selected_u == 'U3' else None))
    return dict(seed=seed, checkpoints=paths, jobs=jobs)


def load_sources(u, a):
    train = ROOT/'options/train/mambairv2'
    return dict(
        rgb=read_yaml(train/'train_S0_RGB_MambaIRv2_x4.yml'),
        v1_a=read_yaml(train/'train_UDR_MambaSR_x4_phaseA.yml'),
        v1_b=read_yaml(train/'train_UDR_MambaSR_x4_phaseB.yml'),
        e2_a=(read_yaml(train/'train_E2_U3_phaseA_x4.yml') if u == 'U3' else None),
        e2_b=read_yaml(train/f'train_E2_{u}_phaseB_x4.yml'),
        e3_a=read_yaml(train/f'train_E3_{a}_phaseA_x4.yml'),
        e3_b=read_yaml(train/f'train_E3_{a}_phaseB_x4.yml'),
        e4_a=read_yaml(train/'train_E4_phaseA_x4.yml'),
        e4_b=read_yaml(train/'train_E4_phaseB_x4.yml'))


def prepare(args):
    selection, u, a = read_selection(args.selection)
    sources = load_sources(u, a)
    e4_test = read_yaml(args.e4_test_config)
    if (sources['e4_a']['network_g']['uncertainty_mode'] != MODES[u] or
            e4_test['network_g']['uncertainty_mode'] != MODES[u] or
            sources['e4_b']['network_g']['uncertainty_mode'] != MODES[u] or
            sources['e4_a']['train']['lambda_a'] != TAGS[a] or
            sources['e4_b']['train']['lambda_a'] != TAGS[a]):
        raise ValueError('E6 templates do not match measured E4 winners.')
    budgets = dict(rgb=500000, v1_a=100000, v1_b=100000,
                   e2_b=100000, e3_a=30000, e3_b=100000,
                   e4_a=30000, e4_b=100000)
    if u == 'U3':
        budgets['e2_a'] = 30000
    for stage, steps in budgets.items():
        if sources[stage]['train']['total_iter'] != steps:
            raise ValueError(f'E6 {stage} training budget changed from its source experiment.')
    seeds = args.seeds
    if len(seeds) not in (3, 5) or len(set(seeds)) != len(seeds):
        raise ValueError('E6 needs 3 distinct seeds, or optional 5 distinct seeds.')
    output = Path(args.output).resolve()
    folder = output/'configs'
    folder.mkdir(parents=True, exist_ok=True)
    runs = [make_seed_configs(seed, u, a, sources, folder) for seed in seeds]
    # The same architecture/alpha choice is tested across independent RGB
    # initialization, E0/E2/E3 training, then E4 adaptation for each seed.
    plan = dict(schema=1, seeds=seeds,
                selected_uncertainty=u, selected_alpha=a,
                uncertainty_mode=MODES[u], lambda_a=TAGS[a],
                e4_selection_file=str(Path(args.selection).resolve()),
                e4_selection_sha256=digest(args.selection),
                e4_test_config=str(Path(args.e4_test_config).resolve()),
                runs=runs,
                protocol='independent RGB/v1/E2/E3/E4 training per seed; matched same-seed RGB/v1/v2 inference')
    (output/'E6_plan.json').write_text(json.dumps(plan, indent=2,
                                             ensure_ascii=False), encoding='utf-8')
    lines = ['# E6 independent training commands', '',
             'Run from this repository root after E4 winner selection.',
             'Each seed uses its own RGB initialization and its own full ancestor chain.', '',
             '```bash']
    for run in runs:
        lines.append(f'# Seed {run["seed"]}')
        for job in run['jobs']:
            if job['name'] == 'e4_a':
                lines.append(f'python scripts/udr/prepare_e6_multiseed.py --stage merge --seed {run["seed"]}')
            lines.append(f'CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt "{job["config"]}"')
            lines.append(f'test -f "{job["output_checkpoint"]}"')
    lines += ['```', '', 'Do not use cross-phase --auto_resume or substitute checkpoints from another seed.', '']
    (output/'E6_commands.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved E6 independent training plan and commands to', output)


def merge_seed(args):
    plan_path = Path(args.output).resolve()/'E6_plan.json'
    if not plan_path.is_file():
        raise FileNotFoundError('Run --stage prepare after E4 selection first.')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if args.seed not in plan['seeds']:
        raise ValueError('The merge seed is absent from the E6 plan.')
    run = next(item for item in plan['runs'] if item['seed'] == args.seed)
    paths = run['checkpoints']
    e2, e3 = parameters(paths['e2_b']), parameters(paths['e3_b'])
    merged = merge_parameters(e2, e3, plan['selected_uncertainty'])
    from basicsr.archs.e4_udrv2_arch import E4UDRMambaIRv2
    config = read_yaml(plan['e4_test_config'])
    network = deepcopy(config['network_g'])
    network.pop('type')
    network['uncertainty_mode'] = plan['uncertainty_mode']
    model = E4UDRMambaIRv2(**network)
    model.load_state_dict(merged, strict=True)
    del model
    output = Path(paths['e4_initial'])
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = dict(seed=args.seed, selected_uncertainty=plan['selected_uncertainty'],
                      selected_alpha=plan['selected_alpha'],
                      e2_checkpoint=paths['e2_b'], e3_checkpoint=paths['e3_b'],
                      e2_sha256=digest(paths['e2_b']), e3_sha256=digest(paths['e3_b']))
    torch.save(dict(params=merged, e6_provenance=provenance), output)
    provenance['e4_initial_sha256'] = digest(output)
    output.with_suffix('.json').write_text(json.dumps(provenance, indent=2),
                                           encoding='utf-8')
    print('Strictly merged seed-specific E4 initial checkpoint:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'merge'), default='prepare')
    parser.add_argument('--selection', default=str(ROOT/'experiments/E4_selection/E4_selection.json'))
    parser.add_argument('--e4-test-config', default=str(ROOT/'options/test/mambairv2/test_E4_x4.yml'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[10, 11, 12])
    parser.add_argument('--seed', type=int, help='One planned seed for the merge stage.')
    parser.add_argument('--output', default=str(ROOT/'experiments/E6_plan'))
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args)
    else:
        if args.seed is None:
            parser.error('--stage merge requires --seed.')
        merge_seed(args)


if __name__ == '__main__':
    main()
