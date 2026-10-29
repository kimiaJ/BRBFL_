"""Run prepared recovery pilots sequentially, stopping on errors or incomplete receipts."""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from brbfl.experiments.config import load_experiment_config
from brbfl.experiments.prepare_recovery_followup import ROOT, configurations

FILES = ('validation.json', 'manifest.json', 'global_metrics.csv', 'messages.csv', 'system_resources.csv')


def validate_output(folder: Path, config: dict, config_path: Path) -> None:
    """Reject truncated runs before proceeding to another expensive training job."""
    if any(not (folder / name).is_file() for name in FILES):
        raise RuntimeError(f'missing output files: {folder}')
    data = json.loads((folder / 'validation.json').read_text(encoding='utf-8'))
    if (data.get('configured_rounds') != config['rounds'] or len(data.get('rounds', ())) != config['rounds']
            or data.get('final_model_consensus') is not True):
        raise RuntimeError(f'incomplete rounds or final consensus: {folder}')
    typed = load_experiment_config(config_path)
    expected = {key: getattr(typed, key) for key in (
        'nodes', 'rounds', 'epochs', 'seed', 'protocol', 'framework', 'aggregator', 'batch_size', 'eligible_trainers'
    )}
    expected['topology'] = typed.topology.value
    expected.update({key: asdict(getattr(typed, key)) for key in ('model_detector', 'ca', 'trust', 'dataset', 'attack', 'validation')})
    expected['selection'] = asdict(typed.participant_selection)
    expected = json.loads(json.dumps(expected))
    if data.get('configuration') != expected:
        raise RuntimeError(f'output configuration differs from protocol: {folder}')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('git_commit') != data.get('provenance', {}).get('producing_commit'):
        raise RuntimeError(f'inconsistent producing commit: {folder}')
    verified = data.get('ledger', {}).get('round_verification', {})
    if set(verified) != {str(i) for i in range(config['rounds'])} or any(value is not True for value in verified.values()):
        raise RuntimeError(f'unverified ledger rounds: {folder}')


def main() -> None:
    """Run one relapse pilot or the remaining eleven-config matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-only', action='store_true', help='Run only the new seed-666 relapse pilot.')
    parser.add_argument('--skip-completed', action='store_true', help='Validate and skip existing complete outputs.')
    args = parser.parse_args()
    rows = configurations()
    if args.pilot_only:
        rows = {name: row for name, row in rows.items() if name == 'relapse_enabled_666.yaml'}
    pending = []
    for name, config in rows.items():
        path = ROOT / 'configs/ablation/ca-recovery-followup' / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if json.loads(json.dumps(config)) != yaml.safe_load(path.read_text(encoding='utf-8')):
            raise RuntimeError(f'prepared configuration differs from protocol: {path}')
        folder = ROOT / config['output_dir']
        if folder.exists() and any(folder.iterdir()):
            if not args.skip_completed:
                raise FileExistsError(f'output already exists: {folder}; use --skip-completed only for validated receipts')
            validate_output(folder, config, path)
            print(f'Skip completed: {name}', flush=True)
        else:
            pending.append((path, config))
    for index, (path, config) in enumerate(pending, 1):
        print(f'[{index}/{len(pending)}] {path.name}', flush=True)
        subprocess.run([sys.executable, '-m', 'p2pfl.examples.mnist.mnist', '--config', str(path)], cwd=ROOT, check=True)
        validate_output(ROOT / config['output_dir'], config, path)
    print('Completed. Archive results/ca-recovery-followup/', flush=True)


if __name__ == '__main__':
    main()
