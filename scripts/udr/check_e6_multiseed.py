"""Offline E6 contracts: full seed-chain YAML and paired report semantics."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.udr import prepare_e6_multiseed as planner
from scripts.udr.evaluate_e6_multiseed import (checkpoint_provenance,
                                               self_test as report_self_test)
from scripts.udr.run_e6_training import schedule


def main():
    original_root = planner.ROOT
    u, a = 'U3', 'a1e4'
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = original_root/'options/train/mambairv2'
            target_dir = root/'options/train/mambairv2'
            target_dir.mkdir(parents=True)
            for name in ('train_S0_RGB_MambaIRv2_x4.yml',
                         'train_UDR_MambaSR_x4_phaseA.yml',
                         'train_UDR_MambaSR_x4_phaseB.yml',
                         'train_E2_U3_phaseA_x4.yml',
                         'train_E2_U3_phaseB_x4.yml',
                         'train_E2_U1_phaseB_x4.yml',
                         'train_E3_a1e4_phaseA_x4.yml',
                         'train_E3_a1e4_phaseB_x4.yml'):
                shutil.copy2(source_dir/name, target_dir/name)
            for phase in ('A', 'B'):
                config = yaml.safe_load((source_dir/f'train_E3_a1e4_phase{phase}_x4.yml').read_text(encoding='utf-8'))
                config['model_type'] = 'E4UDRMambaIRv2Model'
                config['network_g']['type'] = 'E4UDRMambaIRv2'
                config['network_g']['uncertainty_mode'] = 'learned_error'
                config['train']['lambda_u'] = .01
                (target_dir/f'train_E4_phase{phase}_x4.yml').write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            test_dir = root/'options/test/mambairv2'
            test_dir.mkdir(parents=True)
            test = yaml.safe_load((original_root/'options/test/mambairv2/test_E3_a1e4_x4.yml').read_text(encoding='utf-8'))
            test['network_g']['type'] = 'E4UDRMambaIRv2'
            test['network_g']['uncertainty_mode'] = 'learned_error'
            test_path = test_dir/'test_E4_x4.yml'
            test_path.write_text(yaml.safe_dump(test), encoding='utf-8')
            selection = root/'selection.json'
            selection.write_text(json.dumps(dict(selected_uncertainty=dict(variant=u),
                                                 selected_alpha=dict(tag=a))), encoding='utf-8')
            planner.ROOT = root
            args = type('Args', (), dict(selection=str(selection),
                                         e4_test_config=str(test_path),
                                         seeds=[10, 11, 12],
                                         output=str(root/'experiments/E6_plan')))()
            planner.prepare(args)
            plan = json.loads((root/'experiments/E6_plan/E6_plan.json').read_text(encoding='utf-8'))
            assert plan['selected_uncertainty'] == u and plan['selected_alpha'] == a
            assert len(plan['runs']) == 3
            ordered = schedule(plan)
            assert [step['action'] for step in ordered].count('merge') == 3
            for seed in plan['seeds']:
                steps = [step for step in ordered if step['seed'] == seed]
                assert [step.get('name', step['action']) for step in steps][-3:] == [
                    'merge', 'e4_a', 'e4_b']
            for run in plan['runs']:
                seed, jobs, paths = run['seed'], run['jobs'], run['checkpoints']
                assert len(jobs) == 9  # RGB, v1 A/B, U3 A/B, E3 A/B, E4 A/B
                config = {job['name']: yaml.safe_load(Path(job['config']).read_text(encoding='utf-8'))
                          for job in jobs}
                assert all(cfg['manual_seed'] == seed for cfg in config.values())
                assert config['rgb']['path']['pretrain_network_g'] is None
                assert config['v1_a']['path']['pretrain_network_g'] == paths['rgb']
                assert config['e2_a']['path']['pretrain_network_g'] == paths['v1_b']
                assert config['e2_b']['path']['pretrain_network_g'] == paths['e2_a']
                assert config['e3_a']['path']['pretrain_network_g'] == paths['v1_b']
                assert config['e4_a']['path']['pretrain_network_g'] == paths['e4_initial']
                assert config['e4_b']['path']['pretrain_network_g'] == paths['e4_a']
                assert config['e4_a']['train']['rgb_teacher_checkpoint'] == paths['rgb']
                assert config['e4_a']['datasets'] == config['e3_a']['datasets']
                assert config['e4_a']['train']['lambda_a'] == 1e-4
                assert config['e4_a']['train']['lambda_u'] == .01
            commands = (root/'experiments/E6_plan/E6_commands.md').read_text(encoding='utf-8')
            assert commands.count('--stage merge') == 3
            # Lightweight fake checkpoint bytes exercise the full lineage
            # verifier without requiring any research data or CUDA weights.
            for run in plan['runs']:
                paths = run['checkpoints']
                for key, value in paths.items():
                    if value is None:
                        continue
                    target = Path(value)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(f'{run["seed"]}:{key}'.encode('ascii'))
                record = dict(seed=run['seed'], selected_uncertainty=u,
                              selected_alpha=a, e2_checkpoint=paths['e2_b'],
                              e3_checkpoint=paths['e3_b'],
                              e2_sha256=planner.digest(paths['e2_b']),
                              e3_sha256=planner.digest(paths['e3_b']),
                              e4_initial_sha256=planner.digest(paths['e4_initial']))
                Path(paths['e4_initial']).with_suffix('.json').write_text(
                    json.dumps(record), encoding='utf-8')
            provenance = checkpoint_provenance(plan, ['rgb', 'udrv1', 'udrv2'])
            assert len(provenance['strict_merge_lineage']) == 3
            first, second = plan['runs'][:2]
            second_rgb = Path(second['checkpoints']['rgb'])
            original = second_rgb.read_bytes()
            second_rgb.write_bytes(Path(first['checkpoints']['rgb']).read_bytes())
            try:
                checkpoint_provenance(plan, ['rgb', 'udrv1', 'udrv2'])
            except ValueError:
                pass
            else:
                raise AssertionError('Copied RGB checkpoint passed independent-seed check.')
            second_rgb.write_bytes(original)
            # U1/U2 are parameter-free: E2 has no head-only Phase A and its
            # joint phase starts directly from this seed's UDR-v1 checkpoint.
            for phase in ('A', 'B'):
                path = target_dir/f'train_E4_phase{phase}_x4.yml'
                cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
                cfg['network_g']['uncertainty_mode'] = 'route_concentration'
                cfg['train']['lambda_u'] = 0.
                path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
            test['network_g']['uncertainty_mode'] = 'route_concentration'
            test_path.write_text(yaml.safe_dump(test), encoding='utf-8')
            selection.write_text(json.dumps(dict(selected_uncertainty=dict(variant='U1'),
                                                 selected_alpha=dict(tag=a))), encoding='utf-8')
            args.output = str(root/'experiments/E6_plan_U1')
            planner.prepare(args)
            u1_plan = json.loads((Path(args.output)/'E6_plan.json').read_text(encoding='utf-8'))
            for run in u1_plan['runs']:
                jobs = {job['name']: job for job in run['jobs']}
                assert len(jobs) == 8 and 'e2_a' not in jobs
                e2 = yaml.safe_load(Path(jobs['e2_b']['config']).read_text(encoding='utf-8'))
                assert e2['path']['pretrain_network_g'] == run['checkpoints']['v1_b']
                assert e2['path']['pretrain_kind'] == 'e0'
    finally:
        planner.ROOT = original_root
    report_self_test()
    print('PASS: three independent RGB→v1→E2/E3→E4 chains preserve dataset/loss settings')


if __name__ == '__main__':
    main()
