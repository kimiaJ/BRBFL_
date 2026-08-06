#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Example of a P2PFL MNIST experiment, using a MLP model and a MnistFederatedDM."""

# source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
# snakeviz _MainThread-0.pstat
# gprof2dot -f pstats Gossiper-10.pstat | dot -Tpng -o output.png && open output.png

import argparse
import os
import re
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from brbfl.attacks import clear_attacks, create_attack, prepare_dataset, register_attack
from brbfl.attacks.poisoned_model import PoisonedLightningModel
from brbfl.experiments.attack_evidence import audit_partition, labels, parameter_hash, write_evidence
from brbfl.experiments.config import AttackConfig, DatasetConfig, ExperimentConfig, load_experiment_config
from brbfl.experiments.config import TopologyType as ConfigTopologyType
from brbfl.experiments.datasets import partition_dataset
from brbfl.experiments.manifest import write_manifest
from brbfl.experiments.reproducibility import seed_everything
from brbfl.experiments.sign_flipping_evidence import AuditedModelUpdateAttack
from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import set_standalone_settings, wait_to_finish


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL MNIST experiment using the Web Logger.")
    parser.add_argument("--nodes", type=int, help="The number of nodes.", default=10)
    parser.add_argument("--rounds", type=int, help="The number of rounds.", default=15)
    parser.add_argument("--epochs", type=int, help="The number of epochs.", default=1)
    parser.add_argument("--show_metrics", action="store_true", help="Show metrics.", default=True)
    parser.add_argument("--measure_time", action="store_true", help="Measure time.", default=False)
    parser.add_argument("--token", type=str, help="The API token for the Web Logger.", default="")
    parser.add_argument("--protocol", type=str, help="The protocol to use.", default="grpc", choices=["grpc", "unix", "memory"])
    parser.add_argument("--framework", type=str, help="The framework to use.", default="pytorch", choices=["pytorch", "tensorflow", "flax"])
    parser.add_argument("--aggregator", type=str, help="The aggregator to use.", default="fedavg", choices=["fedavg", "scaffold"])
    parser.add_argument("--profiling", action="store_true", help="Enable profiling.", default=False)
    parser.add_argument("--reduced_dataset", action="store_true", help="Use a reduced dataset just for testing.", default=False)
    parser.add_argument("--use_scaffold", action="store_true", help="Use the Scaffold aggregator.", default=False)
    parser.add_argument("--seed", type=int, help="The seed to use.", default=666)
    parser.add_argument("--batch_size", type=int, help="The batch size for training.", default=128)
    parser.add_argument(
        "--attack",
        type=str,
        choices=[
            "none",
            "label_flipping",
            "sign_flipping",
            "scale",
            "backdoor",
            "model_replacement",
            "sybil_backdoor",
            "free_rider",
            "delay_drop",
            "colluding_backdoor",
        ],
        default="colluding_backdoor",
    )
    parser.add_argument("--adversaries", type=str, default="0,1,2,3,4", help="Comma-separated node indices to be adversaries")
    parser.add_argument("--flip_pairs", type=str, default="0-1,2-3,4-5,6-7,8-9", help="Label pairs to flip (e.g., 0-1)")
    parser.add_argument("--scale_factor", type=float, default=3.0, help="Boost factor for scale attack")
    parser.add_argument(
        "--scale_on",
        type=str,
        choices=["delta", "state"],
        default="delta",
        help="Scale the delta or the whole state",
    )
    parser.add_argument("--save_csv", action="store_true", help="Save results to CSV files.", default=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save CSV results.", default="results/mnist")
    parser.add_argument("--config", type=str, help="Path to a YAML experiment configuration.", default=None)
    parser.add_argument(
        "--topology",
        type=str,
        choices=[t.value for t in TopologyType],
        default="full",
        help="The network topology (star, full, line, ring).",
    )
    args = parser.parse_args()
    # parse topology to TopologyType enum
    args.topology = TopologyType(args.topology)
    if args.flip_pairs:
        pairs = [tuple(map(int, p.split("-"))) for p in args.flip_pairs.split(",")]
        flip_map = {}
        for a, b in pairs:
            flip_map[a] = b
            flip_map[b] = a
        args.flip_map = flip_map
    else:
        args.flip_map = {}
    # parse topology to TopologyType enum
    args.topology = TopologyType(args.topology)

    return args


def save_experiment_results(output_dir: Path, start_time: float | None = None) -> None:
    """
    Save experiment results to CSV files.

    Args:
        output_dir: Directory to save results
        start_time: Start time of the experiment for execution time calculation

    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save message logs
    all_msgs = logger.get_messages(direction="all")
    if all_msgs:
        try:
            pandas_msgs = pd.DataFrame(all_msgs)
            msg_csv_path = output_dir / "messages.csv"
            pandas_msgs.to_csv(msg_csv_path, index=False)
            print(f"Saved messages log to: {msg_csv_path}")
        except Exception as e:
            print(f"Error saving messages log: {e}")

    # Save global metrics
    global_metrics_data = logger.get_global_logs()
    if global_metrics_data:
        flattened_global_metrics = []
        try:
            for exp, nodes in global_metrics_data.items():
                for node, metrics in nodes.items():
                    for metric_name, values in metrics.items():
                        for round_num, value in values:
                            flattened_global_metrics.append(
                                {"experiment": exp, "node": node, "metric": metric_name, "round": round_num, "value": value}
                            )

            if flattened_global_metrics:
                pandas_global_metrics = pd.DataFrame(flattened_global_metrics)
                global_metrics_csv_path = output_dir / "global_metrics.csv"
                pandas_global_metrics.to_csv(global_metrics_csv_path, index=False)
                print(f"Saved global metrics log to: {global_metrics_csv_path}")
        except Exception as e:
            print(f"Error saving global metrics: {e}")

    # Save system metrics
    system_metrics_data = logger.get_system_metrics()
    if system_metrics_data:
        flattened_system_metrics = []
        try:
            for timestamp, sys_metrics in system_metrics_data.items():
                for sys_metric_name, sys_value in sys_metrics.items():
                    flattened_system_metrics.append(
                        {"timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"), "metric_name": sys_metric_name, "metric_value": sys_value}
                    )

            if flattened_system_metrics:
                pandas_system_metrics = pd.DataFrame(flattened_system_metrics)
                system_metrics_csv_path = output_dir / "system_resources.csv"
                pandas_system_metrics.to_csv(system_metrics_csv_path, index=False)
                print(f"Saved system resource metrics log to: {system_metrics_csv_path}")
        except Exception as e:
            print(f"Error saving system resource metrics: {e}")

    # Save execution time if start_time is provided
    if start_time is not None:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time:.4f} seconds")

        time_csv_path = output_dir / "execution_time.csv"
        try:
            time_df = pd.DataFrame({"Execution Time (s)": [f"{execution_time:.4f}"]})
            time_df.to_csv(time_csv_path, index=False)
            print(f"Saved execution time to: {time_csv_path}")
        except Exception as e:
            print(f"Error saving execution time: {e}")


def _config_from_legacy_args(
    n: int,
    r: int,
    e: int,
    show_metrics: bool,
    measure_time: bool,
    protocol: str,
    framework: str,
    aggregator: str,
    reduced_dataset: bool,
    topology: TopologyType | ConfigTopologyType,
    batch_size: int,
    save_csv: bool,
    output_dir: str,
) -> ExperimentConfig:
    attack_name = globals().get("args", argparse.Namespace(attack="none")).attack
    adversaries_raw = globals().get("args", argparse.Namespace(adversaries="")).adversaries
    adversaries = tuple(int(x) for x in adversaries_raw.split(",") if x) if adversaries_raw else ()
    attack_params = {
        "flip_map": getattr(globals().get("args", argparse.Namespace()), "flip_map", {}),
        "scale_factor": getattr(globals().get("args", argparse.Namespace()), "scale_factor", 3.0),
        "scale_on": getattr(globals().get("args", argparse.Namespace()), "scale_on", "delta"),
    }
    return ExperimentConfig(
        nodes=n,
        rounds=r,
        epochs=e,
        seed=Settings.general.SEED,
        protocol=protocol,
        framework=framework,
        aggregator=aggregator,
        topology=ConfigTopologyType(topology.value),
        batch_size=batch_size,
        show_metrics=show_metrics,
        measure_time=measure_time,
        save_csv=save_csv,
        output_dir=output_dir,
        dataset=DatasetConfig(reduced=reduced_dataset),
        attack=AttackConfig(name=attack_name, adversaries=adversaries, parameters=attack_params),
    )


def mnist(
    n: int | None = None,
    r: int | None = None,
    e: int | None = None,
    show_metrics: bool = True,
    measure_time: bool = False,
    protocol: str = "grpc",
    framework: str = "pytorch",
    aggregator: str = "fedavg",
    reduced_dataset: bool = False,
    topology: TopologyType | ConfigTopologyType = TopologyType.FULL,
    batch_size: int = 128,
    save_csv: bool = False,
    output_dir: str = "results/mnist",
    config: ExperimentConfig | None = None,
) -> None:
    """Run the P2PFL MNIST experiment from an explicit experiment configuration."""
    config = config or _config_from_legacy_args(
        n or 10,
        r or 15,
        e or 1,
        show_metrics,
        measure_time,
        protocol,
        framework,
        aggregator,
        reduced_dataset,
        topology,
        batch_size,
        save_csv,
        output_dir,
    )
    seed_everything(config.seed)
    start_dt = datetime.now(timezone.utc)
    start_time = time.time() if config.measure_time else None
    write_manifest(config.output_dir, config, start_dt)

    Settings.gossip.TTL = 1000
    n = config.nodes
    r = config.rounds
    e = config.epochs
    topology = config.topology

    if topology.value == "ring" and n > 20:
        print("Large ring detected — switching to full topology for speed")
        topology = ConfigTopologyType.FULL

    if n > Settings.gossip.TTL:
        raise ValueError(
            "For in-line topology TTL must be greater than the number of nodes.Otherwise, some messages will not be delivered."
        )

    if config.framework == "tensorflow":
        from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn  # type: ignore

        model_fn = model_build_fn  # type: ignore
    elif config.framework == "pytorch":
        from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn  # type: ignore

        model_fn = model_build_fn  # type: ignore
    else:
        raise ValueError(f"Framework {config.framework} not added on this example.")

    data = P2PFLDataset.from_huggingface(config.dataset.name)
    source_labels_before = labels(data)
    partitions = partition_dataset(data, config)
    original_labels = [labels(partition) for partition in partitions[:n]]

    nodes = []
    adversary_indices = list(config.attack.adversaries)
    attack_name = config.attack.name
    attack_params = config.attack.parameters
    clear_attacks()
    partition_audits = []
    model_update_audits = {}
    for i in range(n):
        if config.protocol == "memory":
            address = f"node-{i}"
        elif config.protocol == "unix":
            address = f"unix:///tmp/p2pfl-{i}.sock"
        else:
            address = "127.0.0.1"
        attack_obj = None
        original_model = model_fn()
        if i in adversary_indices and attack_name != "none":
            attack_obj = create_attack(attack_name, attack_params)
            partitions[i] = prepare_dataset(partitions[i], attack_obj)
            if attack_name == "sign_flipping":
                attack_obj = AuditedModelUpdateAttack(attack_obj, [name for name, _ in original_model.model.named_parameters()])
                model_update_audits[f"node-{i}"] = attack_obj

        partition_audits.append(
            audit_partition(
                i,
                original_labels[i],
                labels(partitions[i]),
                attack_params.get("flip_map", {}),
                attack_obj is not None and attack_name == "label_flipping",
            )
        )

        model_to_use = PoisonedLightningModel(original_model.model, node_addr=address)
        node = Node(
            model_to_use,
            partitions[i],
            protocol=MemoryCommunicationProtocol() if config.protocol == "memory" else GrpcCommunicationProtocol(),
            addr=address,
            aggregator=Scaffold() if config.aggregator == "scaffold" else None,
        )
        node.start()
        if attack_obj:
            register_attack(address, attack_obj)
            attack_obj.on_attach(node)
        logger.info(node.addr, f"node: {i}")
        node_attack = attack_name if i in adversary_indices else "N/A"
        logger.info(node.addr, f"Node {i} | Adversary: {i in adversary_indices} | Attack: {node_attack}")
        nodes.append(node)

    try:
        adjacency_matrix = TopologyFactory.generate_matrix(topology.value, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)
        print(f"Waiting for {n} nodes to connect (this may take 1-2 minutes)...")
        max_time = 300
        convergence_start = time.time()
        while time.time() - convergence_start < max_time:
            time.sleep(5)
            connected = sum(1 for node in nodes if len(node.get_neighbors()) >= n - 1)
            print(f"   {connected}/{n} nodes fully connected...")
            if connected == n:
                print(f"CONVERGENCE ACHIEVED in {int(time.time() - convergence_start)} seconds!")
                break
        else:
            print("Partial convergence — continuing anyway (safe for training)")

        print("Starting federated learning...")
        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        nodes[0].set_start_learning(rounds=r, epochs=e, trainset_size=5)
        wait_to_finish(nodes, timeout=60 * 60)

        global_logs = logger.get_global_logs()
        metric_rows = []
        for experiment_nodes in global_logs.values():
            for node_id, node_metrics in experiment_nodes.items():
                for metric, values in node_metrics.items():
                    metric_rows.extend(
                        {"node_id": node_id, "metric": metric, "round": round_number, "value": value} for round_number, value in values
                    )
        participating = [f"node-{i}" for i in range(n)]
        if attack_name == "sign_flipping":
            for audit in model_update_audits.values():
                audit.validate_eligible_updates()
        lifecycle_stage = (
            "update_transmission_after_local_training_before_aggregation"
            if attack_name == "sign_flipping"
            else "dataset_preparation_after_partitioning_before_node_creation"
        )
        write_evidence(
            config.output_dir,
            {
                "lifecycle_stage": lifecycle_stage,
                "malicious_node_ids": [f"node-{i}" for i in adversary_indices],
                "source_target_labels": {str(key): value for key, value in attack_params.get("flip_map", {}).items()},
                "original_dataset_unchanged": labels(data) == source_labels_before,
                "partitions": partition_audits,
                "configured_rounds": r,
                "model_update_event_trace": {node_id: audit.event_trace for node_id, audit in model_update_audits.items()},
                "model_update_counters": {
                    node_id: {
                        "eligible_logical_updates": len(audit.eligible_update_ids()),
                        "logical_attack_applications": len(audit.events),
                        "attack_hook_invocations": len(audit.hook_invocations),
                        "network_transmissions": len(audit.transmissions),
                        "aggregation_observations": sum(
                            event["event_type"] == "update_received_for_aggregation" for event in audit.event_trace
                        ),
                    }
                    for node_id, audit in model_update_audits.items()
                },
                "rounds": [
                    {
                        "round": round_number,
                        "participating_node_ids": [
                            node_id
                            for node_id in participating
                            if node_id not in model_update_audits or model_update_audits[node_id].participated_in_round(round_number)
                        ],
                        "malicious_participant_ids": [
                            node_id for node_id, audit in model_update_audits.items() if audit.participated_in_round(round_number)
                        ],
                        "attack_application_counts": {
                            f"node-{i}": (
                                sum(event["round_id"] == str(round_number) for event in model_update_audits[f"node-{i}"].events)
                                if f"node-{i}" in model_update_audits
                                else 0
                            )
                            for i in range(n)
                        }
                        if attack_name == "sign_flipping"
                        else {row["node_id"]: row["attack_application_count"] for row in partition_audits},
                        "model_update_transformations": {
                            node_id: audit.evidence_for_round(round_number)
                            for node_id, audit in model_update_audits.items()
                            if str(round_number) in audit.eligible_round_ids()
                        },
                        "model_update_transmissions": {
                            node_id: [event for event in audit.transmissions if event["round_id"] == str(round_number)]
                            for node_id, audit in model_update_audits.items()
                        },
                        "per_node_metrics": [row for row in metric_rows if row["round"] == round_number],
                    }
                    for round_number in range(r)
                ],
                "final_model_sha256": parameter_hash(nodes[0].get_model().get_parameters()),
            },
        )

        if config.show_metrics:
            global_logs = logger.get_global_logs()
            rows = []
            if global_logs != {}:
                logs_g = list(global_logs.items())[0][1]
                for node_name, node_metrics in logs_g.items():
                    safe_node_name = re.sub(r"[^a-zA-Z0-9_-]", "_", node_name)
                    for metric, values in node_metrics.items():
                        x, y = zip(*values, strict=False)
                        for round_num, val in values:
                            rows.append({"node": node_name, "metric": metric, "round": round_num, "value": val})
                        plt.plot(x, y, label=metric)
                        plt.scatter(x[-1], y[-1], color="red")
                        plt.title(f"{node_name} - {metric}")
                        plt.xlabel("Epoch")
                        plt.ylabel(metric)
                        plt.legend()
                        plt.savefig(f"results/{safe_node_name}_{metric}.png", dpi=300, bbox_inches="tight")
                        plt.show()
                for metric in ["test_loss", "test_metric", "backdoor_asr"]:
                    if all(metric in node_metrics for node_metrics in logs_g.values()):
                        plt.figure()
                        for node_name, node_metrics in logs_g.items():
                            x, y = zip(*node_metrics[metric], strict=False)
                            plt.plot(x, y, label=node_name)
                        plt.title(f"{metric} comparison between nodes")
                        plt.xlabel("Epoch")
                        plt.ylabel(metric)
                        plt.legend()
                        plt.savefig(f"results/{safe_node_name}_{metric}_3.png", dpi=300, bbox_inches="tight")
                        plt.show()
            if rows:
                pd.DataFrame(rows).to_csv("results/metrics_all_nodes.csv", index=False)

    finally:
        for node in nodes:
            node.stop()
        if config.measure_time and start_time is not None:
            print("--- %s seconds ---" % (time.time() - start_time))
        if config.save_csv:
            save_experiment_results(Path(config.output_dir), start_time)


if __name__ == "__main__":
    # Parse args
    args = __parse_args()

    set_standalone_settings()

    if args.profiling:
        import os  # noqa: I001
        import yappi  # type: ignore

        # Start profiler
        yappi.start()

    # Set logger
    if args.token != "":
        logger.connect(p2pfl_web_url="http://localhost:3000/api/v1", p2pfl_web_key=args.token)

    # Launch experiment
    try:
        config = (
            load_experiment_config(args.config)
            if args.config
            else _config_from_legacy_args(
                args.nodes,
                args.rounds,
                args.epochs,
                args.show_metrics,
                args.measure_time,
                args.protocol,
                args.framework,
                args.aggregator,
                args.reduced_dataset,
                args.topology,
                args.batch_size,
                args.save_csv,
                args.output_dir,
            )
        )
        if not args.config:
            config = replace(config, seed=args.seed)
        mnist(config=config)
    finally:
        if args.profiling:
            print("Im in true condition")
            # Stop profiler
            yappi.stop()
            # Save stats
            profile_dir = os.path.join("profile", "mnist", str(uuid.uuid4()))
            os.makedirs(profile_dir, exist_ok=True)
            for thread in yappi.get_thread_stats():
                yappi.get_func_stats(ctx_id=thread.id).save(f"{profile_dir}/{thread.name}-{thread.id}.pstat", type="pstat")
