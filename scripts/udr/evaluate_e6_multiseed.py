"""E6: matched inference for three independently trained RGB/v1/v2 chains.

Each seed uses its own checkpoints and the same Gumbel seed across models.
The original sorted five-set test worker and uint8 Y-channel metrics are reused.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.udr.prepare_e6_multiseed import digest

DATASETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
PRIMARY = ('rgb', 'udrv1', 'udrv2')


def evidence(deltas):
    positives = sum(value > 0 for value in deltas)
    mean = statistics.mean(deltas)
    if positives == len(deltas):
        level = 'strong'
    elif mean > 0 and positives >= math.ceil(len(deltas) * 2 / 3):
        level = 'moderate'
    elif mean > 0 and positives > 0:
        level = 'weak'
    else:
        level = 'no_positive_evidence'
    return dict(level=level, positive_seeds=positives,
                total_seeds=len(deltas), mean_delta_psnr=mean,
                per_seed_delta_psnr=deltas)


def summarize(runs, seeds, labels):
    for label in labels:
        if set(runs[label]) != set(seeds):
            raise ValueError(f'E6 missing checkpoint/inference seed for {label}.')
    report = dict(seeds=seeds, std_ddof=1, models=labels,
                  per_dataset={}, five_set={})
    for dataset in DATASETS:
        report['per_dataset'][dataset] = {}
        baseline = [runs['rgb'][seed][dataset] for seed in seeds]
        for label in labels:
            values = [runs[label][seed][dataset] for seed in seeds]
            metrics = {field: dict(mean=statistics.mean(v[field] for v in values),
                                   std=statistics.stdev(v[field] for v in values),
                                   by_seed=[v[field] for v in values])
                       for field in ('psnr', 'ssim')}
            paired = [v['psnr'] - b['psnr'] for v, b in zip(values, baseline)]
            verdict = (evidence(paired) if label != 'rgb' else
                       dict(level='reference', positive_seeds=0,
                            total_seeds=len(seeds), mean_delta_psnr=0.,
                            per_seed_delta_psnr=paired))
            entry = dict(metrics=metrics,
                         delta_psnr_vs_rgb=dict(mean=statistics.mean(paired),
                                                std=statistics.stdev(paired),
                                                by_seed=paired),
                         evidence_vs_rgb=verdict)
            if label == 'udrv2':
                v1 = [runs['udrv1'][seed][dataset] for seed in seeds]
                gain = [v['psnr'] - old['psnr'] for v, old in zip(values, v1)]
                entry['delta_psnr_vs_udrv1'] = dict(mean=statistics.mean(gain),
                                                    std=statistics.stdev(gain),
                                                    by_seed=gain)
                entry['evidence_vs_udrv1'] = evidence(gain)
            report['per_dataset'][dataset][label] = entry
    for label in labels:
        avg_psnr = [statistics.mean(runs[label][seed][dataset]['psnr']
                                    for dataset in DATASETS) for seed in seeds]
        gain = [statistics.mean(runs[label][seed][dataset]['psnr'] -
                                runs['rgb'][seed][dataset]['psnr']
                                for dataset in DATASETS) for seed in seeds]
        report['five_set'][label] = dict(
            psnr_mean=statistics.mean(avg_psnr),
            psnr_std=statistics.stdev(avg_psnr),
            delta_psnr_vs_rgb=dict(mean=statistics.mean(gain),
                                   std=statistics.stdev(gain),
                                   by_seed=gain),
            evidence_vs_rgb=(evidence(gain) if label != 'rgb' else
                             dict(level='reference', positive_seeds=0,
                                  total_seeds=len(seeds), mean_delta_psnr=0.,
                                  per_seed_delta_psnr=gain)))
    report['note'] = ('Seed-to-seed std includes independent initialization/training '
                      'and matched Gumbel inference variation; do not reinterpret '
                      'these three values as pure inference-only repeats.')
    return report


def validate_configs(configs):
    rgb = configs['rgb']
    if rgb['network_g']['type'] != 'MambaIRv2' or configs['udrv1']['network_g']['type'] != 'UDRMambaIRv2' or configs['udrv2']['network_g']['type'] != 'E4UDRMambaIRv2':
        raise ValueError('E6 requires RGB, UDR-v1 and final E4 UDR-v2 only.')
    if rgb['scale'] != 4 or tuple(rgb['datasets'][key]['name']
                                   for key in sorted(rgb['datasets'])) != DATASETS:
        raise ValueError('E6 must use the original five x4 test sets.')
    for label, config in configs.items():
        if config['scale'] != 4 or config['val']['metrics'] != rgb['val']['metrics']:
            raise ValueError(f'E6 {label} metric/scale mismatch.')
        if set(config['datasets']) != set(rgb['datasets']):
            raise ValueError(f'E6 {label} dataset count mismatch.')
        for key in rgb['datasets']:
            for field in ('name', 'dataroot_gt', 'dataroot_lq', 'filename_tmpl'):
                if config['datasets'][key][field] != rgb['datasets'][key][field]:
                    raise ValueError(f'E6 {label} differs at {key}/{field}.')


def read_plan(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('E6 plan missing; run prepare_e6_multiseed.py.')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if len(plan['seeds']) not in (3, 5) or len(set(plan['seeds'])) != len(plan['seeds']):
        raise ValueError('E6 plan must contain 3 or optional 5 distinct seeds.')
    if digest(plan['e4_selection_file']) != plan['e4_selection_sha256']:
        raise ValueError('E4 winner selection changed after E6 planning.')
    return plan


def checkpoint_provenance(plan, labels):
    lookup = dict(rgb='rgb', udrv1='v1_b', udrv2='e4_b',
                  e2_best='e2_b', e3_best='e3_b')
    provenance = {}
    for label in labels:
        hashes = {}
        for run in plan['runs']:
            seed = run['seed']
            path = Path(run['checkpoints'][lookup[label]])
            if not path.is_file():
                raise FileNotFoundError(f'E6 {label} seed {seed} checkpoint missing: {path}')
            hashes[seed] = dict(path=str(path.resolve()), sha256=digest(path))
        if len({item['sha256'] for item in hashes.values()}) != len(hashes):
            raise ValueError(f'E6 {label} has duplicate checkpoint content across seeds.')
        provenance[label] = hashes
    # Audit every ancestor, not just the final three models: a missing E2/E3
    # or seed-specific merged start invalidates the initialization claim.
    lineage = {}
    for run in plan['runs']:
        seed = run['seed']
        for key in ('rgb', 'v1_a', 'v1_b', 'e2_b', 'e3_a', 'e3_b',
                    'e4_initial', 'e4_a', 'e4_b'):
            if not Path(run['checkpoints'][key]).is_file():
                raise FileNotFoundError(f'E6 seed {seed} ancestor missing: {key}')
        if plan['selected_uncertainty'] == 'U3' and not Path(run['checkpoints']['e2_a']).is_file():
            raise FileNotFoundError(f'E6 seed {seed} U3 Phase A checkpoint missing.')
        record_path = Path(run['checkpoints']['e4_initial']).with_suffix('.json')
        if not record_path.is_file():
            raise FileNotFoundError(f'E6 seed {seed} strict-merge lineage missing: {record_path}')
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if (record['seed'] != seed or
                record['selected_uncertainty'] != plan['selected_uncertainty'] or
                record['selected_alpha'] != plan['selected_alpha'] or
                record['e2_checkpoint'] != run['checkpoints']['e2_b'] or
                record['e3_checkpoint'] != run['checkpoints']['e3_b'] or
                record['e2_sha256'] != digest(record['e2_checkpoint']) or
                record['e3_sha256'] != digest(record['e3_checkpoint']) or
                record['e4_initial_sha256'] != digest(run['checkpoints']['e4_initial'])):
            raise ValueError(f'E6 seed {seed} merged E4 lineage does not match its source checkpoints.')
        lineage[seed] = record
    for field in ('e2_sha256', 'e3_sha256'):
        if len({record[field] for record in lineage.values()}) != len(lineage):
            raise ValueError(f'E6 {field} repeats across independent training seeds.')
    provenance['strict_merge_lineage'] = lineage
    return provenance


def run(args):
    plan = read_plan(args.plan)
    labels = list(PRIMARY)
    if args.include_e2_e3:
        labels += ['e2_best', 'e3_best']
    cfgpaths = dict(rgb=args.rgb_config, udrv1=args.udrv1_config,
                    udrv2=plan['e4_test_config'],
                    e2_best=str(ROOT/f"options/test/mambairv2/test_E2_{plan['selected_uncertainty']}_x4.yml"),
                    e3_best=str(ROOT/f"options/test/mambairv2/test_E3_{plan['selected_alpha']}_x4.yml"))
    configs = {label: yaml.safe_load(Path(cfgpaths[label]).read_text(encoding='utf-8'))
               for label in labels}
    validate_configs(configs)
    if configs['udrv2']['network_g']['uncertainty_mode'] != plan['uncertainty_mode']:
        raise ValueError('E6 test E4 mode disagrees with fixed winner selection.')
    provenance = checkpoint_provenance(plan, labels)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = {label: {} for label in labels}
    worker = ROOT/'scripts/grs/evaluate_repeated.py'
    lookup = {run['seed']: run for run in plan['runs']}
    for seed in plan['seeds']:
        for label in labels:
            config = deepcopy(configs[label])
            config['name'] = f'E6_{label}_s{seed}_paired_test'
            config['manual_seed'] = seed
            config['path']['pretrain_network_g'] = provenance[label][seed]['path']
            config['path']['strict_load_g'] = True
            cfg_path = output/f'{label}_s{seed}.yml'
            result_path = output/f'{label}_s{seed}.json'
            cfg_path.write_text(yaml.safe_dump(config, sort_keys=False,
                                               allow_unicode=True), encoding='utf-8')
            # Separate workers exactly reuse the existing sorted-image test
            # pipeline and reset the same seed before every dataset/model.
            subprocess.run([sys.executable, str(worker), '--worker',
                            str(cfg_path), str(result_path)], cwd=ROOT, check=True)
            measured = json.loads(result_path.read_text(encoding='utf-8'))
            if set(measured) != set(DATASETS):
                raise ValueError(f'E6 worker returned incomplete datasets: {label} seed {seed}')
            results[label][seed] = measured
    report = summarize(results, plan['seeds'], labels)
    report.update(selected_uncertainty=plan['selected_uncertainty'],
                  selected_alpha=plan['selected_alpha'],
                  checkpoints=provenance,
                  protocol=plan['protocol'])
    (output/'E6_multiseed.json').write_text(json.dumps(report, ensure_ascii=False,
                                                    indent=2, allow_nan=False),
                                           encoding='utf-8')
    lines = ['# E6 matched independent-seed confirmation', '',
             f"Training/inference seeds: {plan['seeds']}. Three independently trained checkpoint chains.",
             'PSNR/SSIM mean±sample std across seeds; ΔPSNR is paired to the same-seed RGB model.', '',
             '| Dataset | Model | PSNR mean±std | SSIM mean±std | ΔPSNR vs RGB mean±std | Evidence |',
             '|---|---|---:|---:|---:|---|']
    for dataset in DATASETS:
        for label in labels:
            item = report['per_dataset'][dataset][label]
            p, s, d = (item['metrics']['psnr'], item['metrics']['ssim'],
                       item['delta_psnr_vs_rgb'])
            lines.append(f"| {dataset} | {label} | {p['mean']:.4f}±{p['std']:.4f} | "
                         f"{s['mean']:.4f}±{s['std']:.4f} | "
                         f"{d['mean']:+.4f}±{d['std']:.4f} | "
                         f"{item['evidence_vs_rgb']['level']} "
                         f"({item['evidence_vs_rgb']['positive_seeds']}/{len(plan['seeds'])}) |")
    lines += ['', 'Strong: all seeds positive. Moderate: positive mean and at least '
              'ceil(2n/3) positive seeds (2/3 or 4/5). Weak: positive mean with fewer '
              'positive seeds. No positive evidence otherwise.',
              'See JSON for all per-seed values, five-set averages and UDR-v2 versus UDR-v1.', '']
    (output/'E6_multiseed.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved E6 matched independent-seed report to', output)


def self_test():
    seeds = [10, 11, 12]
    labels = list(PRIMARY)
    runs = {label: {} for label in labels}
    for seed in seeds:
        for label, offset in (('rgb', 0), ('udrv1', .02), ('udrv2', .05)):
            runs[label][seed] = {dataset: dict(psnr=30 + seed / 100 + offset,
                                              ssim=.9 + offset / 10)
                                 for dataset in DATASETS}
    report = summarize(runs, seeds, labels)
    assert report['per_dataset']['Set5']['udrv2']['evidence_vs_rgb']['level'] == 'strong'
    assert abs(report['five_set']['udrv2']['delta_psnr_vs_rgb']['mean'] - .05) < 1e-10
    runs['udrv2'][12]['Set5']['psnr'] = runs['rgb'][12]['Set5']['psnr'] - .01
    report = summarize(runs, seeds, labels)
    assert report['per_dataset']['Set5']['udrv2']['evidence_vs_rgb']['level'] == 'moderate'
    print('PASS: independent-seed paired means/std and 3/3 versus 2/3 evidence labels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(ROOT/'experiments/E6_plan/E6_plan.json'))
    parser.add_argument('--rgb-config', default=str(ROOT/'options/test/mambairv2/test_UDR_RGB_reference_x4.yml'))
    parser.add_argument('--udrv1-config', default=str(ROOT/'options/test/mambairv2/test_UDR_MambaSR_x4.yml'))
    parser.add_argument('--include-e2-e3', action='store_true')
    parser.add_argument('--output', default=str(ROOT/'results/E6_multiseed'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run(args)


if __name__ == '__main__':
    main()
