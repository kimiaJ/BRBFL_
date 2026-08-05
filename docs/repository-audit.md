# Repository Audit

## Scope

This audit inspected the repository without deleting, moving, or restructuring code. It focuses on the upstream base, the MNIST experiment path, thesis-specific attack code, duplicate application paths, imports, global arguments, hard-coded parameters, nondeterminism, and existing checks.

## Upstream base

See `docs/upstream-version.md`. Local metadata identifies the project as P2PFL `0.4.4`; no exact upstream commit can be verified from this clone because there are no remotes or tags, and an attempted upstream tag query failed with a `CONNECT tunnel failed, response 403` network/proxy error.

## Thesis-specific code map

Primary thesis-specific files are under `p2pfl/examples/mnist`:

- `p2pfl/examples/mnist/mnist.py`: custom MNIST experiment CLI, attack selection, adversary registration, plotting, CSV export, and custom convergence wait loop.
- `p2pfl/examples/mnist/attacks/base.py`: base attack lifecycle hooks and default pass-through poison/manipulation methods.
- `p2pfl/examples/mnist/attacks/registry.py`: process-local global registry keyed by node address.
- `p2pfl/examples/mnist/attacks/poisoned_model.py`: `LightningModel` subclass that looks up attacks by node address and applies `manipulate_update()` in `get_parameters()`.
- `p2pfl/examples/mnist/attacks/label_flipping.py`: data poisoning through private `P2PFLDataset` internals.
- `p2pfl/examples/mnist/attacks/sign_flipping.py`: model update scaling with a negative factor.
- `p2pfl/examples/mnist/attacks/scale.py`: state or delta scaling attack.
- `p2pfl/examples/mnist/attacks/backdoor.py`: batch poisoning helper with trigger insertion and target labels.
- `p2pfl/examples/mnist/attacks/model_replacement.py`: model replacement / scaled delta attack; currently references an undefined `adversary_indices` name.
- `p2pfl/examples/mnist/attacks/sybil_backdoor.py`: backdoor subclass with sybil-count logging.
- `p2pfl/examples/mnist/attacks/colluding_backdoor.py`: backdoor subclass that scales updates.
- `p2pfl/examples/mnist/attacks/free_rider.py`: zero/random/scale free-rider update generation.
- `p2pfl/examples/mnist/attacks/delay_drop.py`: probabilistic delay/drop replacement using `random` and NumPy noise.
- `p2pfl/examples/mnist/attacks/model_wrapper.py`: older wrapper path not used by current `mnist.py`.
- `p2pfl/examples/mnist/analyze_results.ipynb`, `grid_search.ipynb`, and `attack results/`: analysis artifacts and thesis outputs.
- Top-level images `performance over time.png` and `individual node performance.png`: likely generated thesis figures.

Git history also identifies thesis-specific commits:

```text
0497364 final version
35270a1 final correct model replacement
9a06873 model_replacement attack module
48d8e68 add scale attack module
ad8a103 test analyze_results
cb3abe5 base set up of p2pfl and labelflip and sign flip attacks
```

## Duplicate attack application paths

Several paths can apply or attempt to apply attacks:

1. `mnist.py` directly poisons partitions for label flipping.
2. `mnist.py` contains direct monkey-patching code for `sign_flipping` and `scale` via `original_model.get_parameters`, but the patched `original_model` is not passed to `Node`; `PoisonedLightningModel(original_model.model, ...)` is used instead. The `scale` path defines `poisoned_get_parameters()` but does not assign it back to `original_model.get_parameters`.
3. `PoisonedLightningModel.get_parameters()` applies registered attacks at send/read time.
4. `AttackableLightningModel` is an older wrapper that also overrides `get_parameters()`, but it is currently commented out / unused in `mnist.py`.
5. `BackdoorAttack.poison_batch()` only matters if the training model calls it. Current `mnist.py` registers backdoor objects, but the visible wrapper only manipulates `get_parameters()`; verify whether the PyTorch model calls the registry or attack object during training before relying on batch poisoning.

Risk: model poisoning can be silently absent, duplicated, or applied at the wrong lifecycle point depending on which wrapper/patch path is active.

## Inconsistent imports

- `mnist.py` mixes package imports (`p2pfl.examples.mnist.attacks...`) and local imports (`from attacks...`). Running from the repository root can fail for local `attacks` imports because `attacks` is not a top-level package.
- Several attack files use relative imports (`from .base import BaseAttack`), while others use local absolute imports (`from attacks.backdoor import BackdoorAttack`).
- `ScaleAttack` and `RandomIIDPartitionStrategy` are imported twice in `mnist.py`.
- `torch`, `sys`, and some typing imports appear unused in the current script or attack modules.

Risk: behavior depends on current working directory and `sys.path`, making script execution and module execution inconsistent.

## Global args usage

`mnist()` accepts parameters but still reads or mutates global `args` for:

- topology switching (`args.topology`), despite having a `topology` parameter;
- adversary parsing (`args.adversaries`);
- attack selection (`args.attack`);
- attack-specific parameters (`args.flip_map`, `args.scale_factor`, `args.scale_on`);
- error text (`args.framework`).

Risk: direct imports/tests cannot call `mnist()` safely, and CLI parsing is coupled to experiment execution.

## Hard-coded parameters

Notable hard-coded values:

- Default attack is `colluding_backdoor` and default adversaries are `0,1,2,3,4`.
- `Settings.gossip.TTL = 1000` is forced in `mnist()`.
- Reduced dataset partitions use `n * 50` partitions rather than a named sample/reduction parameter.
- `nodes[0].set_start_learning(..., trainset_size=5)` ignores dataset size and CLI/YAML settings.
- Convergence loop waits up to `300` seconds and requires `n - 1` neighbors.
- `wait_to_finish(nodes, timeout=60 * 60)` is fixed at one hour.
- Attack parameters such as model replacement scaling, trigger sizes, target classes, poison rates, delay/drop rates, sybil count, and colluding scale factor are embedded in code.
- `ModelReplacementAttack` hard-codes `total_nodes = 10` and references undefined `adversary_indices`.
- Plot and CSV paths under `results/` are hard-coded in metric plotting even when `output_dir` is provided.

## Nondeterministic behavior

- `torch.randperm()` in backdoor poisoning is not explicitly seeded locally.
- `random.random()` in delay/drop is not seeded by `Settings.general.SEED` unless some other setup seeds Python's `random`.
- `np.random.randn()` in delay/drop and free-rider random mode is not explicitly seeded locally.
- Network startup/convergence timing and asynchronous node behavior can vary run to run.
- Plot output can vary if metric availability differs, and `plt.show()` can behave differently across environments.

## Existing checks

### Succeeded

```sh
.venv/bin/python -m compileall p2pfl test
```

Result: succeeded. Python compilation completed for `p2pfl` and `test`.

### Failed

```sh
.venv/bin/python -m pytest
```

Result: failed during collection with 10 import errors. The root cause is missing optional dependency `opendp`, raised from `p2pfl/learning/compression/dp_strategy.py` as `ImportError: Please install with \`pip install p2pfl[dp]\``.

```sh
git ls-remote --tags https://github.com/p2pfl/p2pfl.git 'refs/tags/*'
```

Result: failed due network/proxy restriction: `CONNECT tunnel failed, response 403`.

## Missing dependencies

- `opendp` is missing for the current test suite because importing compression modules imports DP compression eagerly.
- The MNIST PyTorch experiment also requires the torch optional dependency set (`torch`, `torchvision`, `torchmetrics`, `lightning`). These appear present enough for compilation, but a full smoke run was not attempted in this audit because the goal was inspection plus existing tests/compilation.
- Profiling requires `yappi` if `--profiling` is used; it is not listed in `pyproject.toml` dependencies.

## Risky refactoring areas

1. Normalize attack imports only after choosing one supported invocation style (`python file.py` vs `python -m package.module`).
2. Consolidate attack application into one path, preferably `PoisonedLightningModel` plus explicit data-poisoning hooks, and remove stale monkey-patching only after tests cover it.
3. Remove global `args` reads from `mnist()` by passing a typed config or explicit parameters.
4. Make attack parameters CLI-configurable before changing defaults.
5. Fix clean-run flags (`--show_metrics`, `--save_csv`) before using the script for regression checks.
6. Isolate plotting/result writing so clean experiments do not require all attack metrics.
7. Seed Python `random`, NumPy, and Torch consistently at experiment start.
8. Fix `ModelReplacementAttack` undefined `adversary_indices` and hard-coded `total_nodes` before relying on it.
9. Avoid direct mutation of `P2PFLDataset._data` unless wrapped in a clearly documented compatibility layer.

## Recommended order of changes

1. Add a minimal smoke-test command and fix CLI booleans/defaults so `--attack none` with two memory nodes can run cleanly.
2. Standardize imports in MNIST attack modules and make the entry point robust from the repository root.
3. Replace global `args` usage in `mnist()` with explicit parameters/config.
4. Choose a single model-update attack path and remove or quarantine obsolete wrappers/monkey patches in a later cleanup.
5. Parameterize hard-coded attack and experiment settings.
6. Add deterministic seeding for Python `random`, NumPy, Torch, and data poisoning selection.
7. Add targeted unit tests for each attack's data/update transformation.
8. Add a small integration smoke test for clean MNIST memory mode.
9. After behavior is covered, remove duplicate/stale files or move artifacts in a separate structural cleanup PR.
