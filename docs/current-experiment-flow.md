# Current MNIST Experiment Flow

## Entry points

The current custom MNIST entry point is:

```sh
python p2pfl/examples/mnist/mnist.py
```

The module contains the custom CLI options for attacks, adversaries, topology, CSV output, and plotting. The MNIST README still contains typos (`minst.py` / `minst.yaml`) and should not be treated as authoritative until corrected.

A YAML entry point also exists:

```sh
python -m p2pfl run p2pfl/examples/mnist/mnist.yaml
```

However, the YAML path uses the generic launcher and currently does not include the thesis attack configuration. For Byzantine MNIST attack work, `mnist.py` is the active entry point.

## Smallest clean experiment

Use no attack, the memory transport, two nodes, one round, one epoch, reduced data, and metric display disabled by overriding the default attack and default plotting behavior:

```sh
.venv/bin/python p2pfl/examples/mnist/mnist.py \
  --nodes 2 \
  --rounds 1 \
  --epochs 1 \
  --protocol memory \
  --framework pytorch \
  --attack none \
  --adversaries "" \
  --reduced_dataset \
  --show_metrics False \
  --save_csv False
```

Important caveat: `--show_metrics` and `--save_csv` are currently declared with `action="store_true"` and `default=True`, so passing `False` as shown above is not accepted by argparse. Until that is fixed, the least disruptive practical clean run is:

```sh
.venv/bin/python p2pfl/examples/mnist/mnist.py \
  --nodes 2 \
  --rounds 1 \
  --epochs 1 \
  --protocol memory \
  --framework pytorch \
  --attack none \
  --adversaries "" \
  --reduced_dataset \
  --output_dir results/audit-clean-smoke
```

This may still create plots and CSV output because the defaults are enabled.

## Flow

1. `__parse_args()` defines experiment settings and attack parameters.
2. `__main__` parses arguments, calls `set_standalone_settings()`, optionally configures web logging/profiling, stores `args.seed` in `Settings.general.SEED`, and calls `mnist(...)`.
3. `mnist(...)` sets `Settings.gossip.TTL = 1000`, mutates `args.topology` for large rings, loads the selected model builder, downloads/loads `p2pfl/MNIST`, sets batch size, and generates IID partitions.
4. Node addresses are generated from the selected protocol.
5. For adversary node indices, an attack object is constructed. Some attacks poison data immediately; others are registered for later update manipulation.
6. Every model is wrapped in `PoisonedLightningModel(original_model.model, node_addr=address)`.
7. Nodes are created, started, optionally registered in the global attack registry, and connected through `TopologyFactory`.
8. A custom polling loop waits up to five minutes for all nodes to have `n - 1` neighbors.
9. Learning starts from `nodes[0].set_start_learning(rounds=r, epochs=e, trainset_size=5)`.
10. The script waits up to one hour, plots metrics, writes result CSVs if enabled, and stops nodes in `finally`.

## Current limitations for clean runs

- The default attack is `colluding_backdoor`, not `none`.
- Default adversaries are `0,1,2,3,4`, which exceeds the node count for small experiments and can make every node malicious in small runs.
- `--show_metrics` and `--save_csv` cannot be disabled from the CLI because they are `store_true` flags with `default=True`.
- `mnist()` reads the global `args` object instead of only using its function parameters, so direct programmatic calls to `mnist()` are unsafe unless global `args` exists.
- The script assumes metrics such as `backdoor_asr` exist during plotting, which may fail in clean runs if that metric was not logged.
