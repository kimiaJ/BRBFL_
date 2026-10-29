"""Control the followup matrix and reject incomplete experiment receipts."""

import json
from copy import deepcopy

import pytest
import yaml

from brbfl.experiments.config import load_experiment_config
from brbfl.experiments.prepare_recovery_followup import ROOT, configurations, prepare
from brbfl.experiments.run_recovery_followup import FILES, validate_output


def test_followup_has_eleven_new_runs_and_preserves_the_recorded_protocol(tmp_path):
    """Only seed, relapse schedule and output location extend the existing pilot."""
    paths = prepare(tmp_path / 'matrix')
    assert len(paths) == 11
    configs = configurations()
    assert len({row['output_dir'] for row in configs.values()}) == 11
    assert [name for name in configs if name.endswith('_666.yaml')] == ['relapse_enabled_666.yaml']
    for path in paths:
        typed = load_experiment_config(path)
        row = yaml.safe_load(path.read_text())
        scenario, seed = path.stem.rsplit('_', 1)
        template = 'transient_enabled' if scenario == 'relapse_enabled' else scenario
        original = yaml.safe_load((ROOT / f'configs/ablation/ca-recovery/{template}_666.yaml').read_text())
        assert row['seed'] == int(seed)
        assert row['rounds'] == 12
        assert typed.ca.policy.promotion_min_trust == .7
        assert typed.ca.policy.recovery_min_trust == .6
        if scenario == 'relapse_enabled':
            assert row['attack']['parameters']['active_rounds'] == [0, 1, 9, 10, 11]
            row['attack']['parameters']['active_rounds'] = [0, 1]
        row['seed'] = 666
        row['output_dir'] = original['output_dir']
        assert row == original
    before = {p.name: p.read_bytes() for p in paths}
    with pytest.raises(FileExistsError):
        prepare(tmp_path / 'matrix')
    assert before == {p.name: p.read_bytes() for p in paths}


def test_output_verifier_rejects_truncated_run(tmp_path):
    """A process success exit alone cannot advance the batch."""
    config = configurations()['relapse_enabled_666.yaml']
    for name in FILES:
        (tmp_path / name).write_text('{}')
    data = {'configured_rounds': 12, 'rounds': [{}] * 11, 'final_model_consensus': True}
    (tmp_path / 'validation.json').write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='incomplete'):
        validate_output(tmp_path, config, ROOT / 'configs/ablation/ca-recovery-followup/relapse_enabled_666.yaml')


def test_transient_pair_remains_controlled():
    """Recovery is the only defense-policy difference within each paired seed."""
    configs = configurations()
    for seed in (667, 668):
        pair = [deepcopy(configs[f'transient_{mode}_{seed}.yaml']) for mode in ('disabled', 'enabled')]
        for row in pair:
            row.pop('output_dir')
            row['ca'].pop('recovery_probes_enabled')
            row['ca']['policy'].pop('recovery_enabled')
        assert pair[0] == pair[1]


@pytest.mark.parametrize('failure', ['process', 'receipt'])
def test_batch_stops_before_launching_next_job_after_failure(tmp_path, monkeypatch, failure):
    """Never spend another training run after a failed or incomplete experiment."""
    import subprocess
    import sys

    from brbfl.experiments import run_recovery_followup as runner

    rows = dict(list(configurations().items())[:2])
    folder = tmp_path / 'configs/ablation/ca-recovery-followup'
    folder.mkdir(parents=True)
    for name, config in rows.items():
        (folder / name).write_text(yaml.safe_dump(config))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'configurations', lambda: rows)
    monkeypatch.setattr(sys, 'argv', ['runner'])
    launched = []

    def launch(command, **kwargs):
        launched.append(command)
        if failure == 'process':
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, 'run', launch)
    expected = subprocess.CalledProcessError if failure == 'process' else RuntimeError
    with pytest.raises(expected):
        runner.main()
    assert len(launched) == 1
