## Quick context for AI coding agents

This repo implements P2PFL — a peer-to-peer federated learning framework. Focus on these core concepts first so your edits are correct and helpful:

- Entrypoints & CLI
  - Primary CLI: `p2pfl.management.cli:app` (exposed as the `p2pfl` script via `pyproject.toml` [project.scripts]). See `p2pfl/management/cli.py` for commands: `run`, `run-variations`, `login`.
  - Module entry: `python -m p2pfl` runs `p2pfl.__main__` which calls the CLI app.

- Big-picture architecture (modules & responsibilities)
  - p2pfl/node.py — Node lifecycle, networking hooks, learning orchestration (uses CommunicationProtocol, Learner, Aggregator, and Workflows).
  - p2pfl/communication/ — communication layer. Default protocol: gRPC implementation in `communication/protocols/protobuff/grpc` and protocol entrypoints under `communication/protocols`.
  - p2pfl/learning/ — learners, datasets and aggregators. Learners are created via `LearnerFactory` and wrapped via `try_init_learner_with_ray` for optional Ray support.
  - p2pfl/stages/ — stage-based workflow engine. `LearningWorkflow` (in `stages/workflows.py`) drives per-round state transitions.
  - p2pfl/management/logger.py — node registration and runtime telemetry calls (important when modifying start/stop flows).

- Typical runtime/data flow
  1. CLI `p2pfl run config.yaml` → `management.launch_from_yaml.run_from_yaml` creates nodes/experiment config.
  2. `Node.start()` starts the communication protocol (gRPC server by default) and registers the node with the logger.
  3. Commands arrive through `communication` and are mapped to Command classes (see `p2pfl/communication/commands/*`). Node registers these commands in `Node.__init__`.
  4. Learning uses `Learner` implementations (framework-specific) + `Aggregator` classes to exchange weights (commands in `communication/commands/weights`).

- Project conventions & important config
  - Type checking: mypy is strict (see `pyproject.toml`), many modules use Python 3.10+ typing features. Keep signatures typed.
  - Linting/format: ruff settings in `pyproject.toml` (line-length 140). Respect existing formatting.
  - Tests: pytest is configured with `-v --cov=p2pfl` (see `[tool.pytest]`); tests live under `test/`.
  - Extras: Optional framework extras (torch/tensorflow/flax) are installed via extra dependencies (see `pyproject.toml` optional-dependencies).

- Project-specific patterns to follow
  - Command pattern for network messages: implement a Command class and add it to the protocol via `add_command()` — see how `Node.__init__` registers `StartLearningCommand`, `InitModelCommand`, etc.
  - Workflow/stage pattern: stages are small classes created by `StageFactory` and executed in order by `LearningWorkflow`. Prefer adding stages via `stages/` following existing pattern.
  - Learner factory: to add new framework support, add a learner to `learning/frameworks/learner_factory.py` so `LearnerFactory.create_learner()` can construct it.
  - Communication protocol pluggability: default is gRPC, but `Node` accepts a protocol instance — add or swap protocols by implementing `CommunicationProtocol` and placing it under `communication/protocols`.

- Dev workflows & commands (concrete)
  - Install for development (recommended): use `uv` as described in the README: `uv sync --all-extras` (installs dev+extras). On Windows prefer activating the created venv before using Ray workers.
  - Run CLI locally: `python -m p2pfl run path/to/config.yaml` or use installed entry `p2pfl run <example-name>` (examples are YAML under `p2pfl/examples/*/*.yaml`).
  - Run tests: use pytest (project config already sets `-v --cov=p2pfl`): `pytest` (or `uv run pytest` if using uv).
  - Lint & typing: `ruff .` and `mypy p2pfl` per `pyproject.toml` rules. Fixes can be applied incrementally; respect `tool.ruff.fixable` list.

- Integration & external dependencies to be careful with
  - gRPC/protobuf: proto-generated files live under `communication/protocols/protobuff/proto`. Avoid editing generated code manually.
  - Ray: optional; learners may be initialized to use Ray — `try_init_learner_with_ray` is used. Be careful when changing process/serialization of learners.
  - Framework extras: heavy libraries (torch, tensorflow) are optional and gated by extras; CI and local dev may not have them installed.

## When you change code, check these files/areas
- `p2pfl/node.py` — lifecycle and command wiring
- `p2pfl/communication/commands/` — to add/modify network messages
- `p2pfl/learning/frameworks/` & `learning/aggregators/` — to add algorithms or frameworks
- `p2pfl/management/cli.py` + `p2pfl/management/launch_from_yaml.py` — to add CLI flags or new example wiring
- `pyproject.toml` — dependency and tooling rules (mypy/ruff, script entry)

If anything in this summary is unclear or you want me to expand a section (examples, tests, or a sample change to add a command/learner/protocol), tell me which area and I will iterate the file with concrete edits and tests.
