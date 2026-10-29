"""Prepare the eleven remaining recovery pilots without changing defense rules."""

from copy import deepcopy
from pathlib import Path

import yaml

from brbfl.experiments.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ('relapse_enabled', 'transient_disabled', 'transient_enabled', 'persistent_enabled', 'clean_enabled')


def configurations() -> dict[str, dict]:
    """Reuse the recorded seed-666 protocol; add a fixed relapse schedule."""
    source = ROOT / 'configs/ablation/ca-recovery'
    result = {}
    for seed in (666, 667, 668):
        for scenario in SCENARIOS:
            if seed == 666 and scenario != 'relapse_enabled':
                continue
            template = 'transient_enabled' if scenario == 'relapse_enabled' else scenario
            row = deepcopy(yaml.safe_load((source / f'{template}_666.yaml').read_text()))
            row['seed'] = seed
            row['output_dir'] = f'results/ca-recovery-followup/{seed}/{scenario}'
            if scenario == 'relapse_enabled':
                row['attack']['parameters']['active_rounds'] = [0, 1, 9, 10, 11]
            result[f'{scenario}_{seed}.yaml'] = row
    return result


def prepare(destination: Path) -> tuple[Path, ...]:
    """Refuse to overwrite any config from a previous preparation."""
    rows = configurations()
    if any((destination / name).exists() for name in rows):
        raise FileExistsError('followup configs already exist')
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, row in rows.items():
        path = destination / name
        with path.open('x', encoding='utf-8') as stream:
            yaml.safe_dump(row, stream, sort_keys=False)
        load_experiment_config(path)
        paths.append(path)
    return tuple(paths)


if __name__ == '__main__':
    for path in prepare(ROOT / 'configs/ablation/ca-recovery-followup'):
        print(path.relative_to(ROOT))
