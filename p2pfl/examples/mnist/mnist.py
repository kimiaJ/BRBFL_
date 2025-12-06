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

import re
import sys
import os

import argparse
import time
from typing import Optional
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from attacks.label_flipping import LabelFlippingAttack
from attacks.sign_flipping import SignFlippingAttack
from attacks.scale import ScaleAttack
from attacks.base import BaseAttack
from attacks.registry import register_attack, clear_attacks
from attacks.poisoned_model import PoisonedLightningModel
from attacks.scale import ScaleAttack
from attacks.backdoor import BackdoorAttack

from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol

from p2pfl.examples.mnist.attacks.model_replacement import ModelReplacementAttack
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy, RandomIIDPartitionStrategy
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import set_standalone_settings, wait_convergence, wait_to_finish



def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL MNIST experiment using the Web Logger.")
    parser.add_argument("--nodes", type=int, help="The number of nodes.", default=10)
    parser.add_argument("--rounds", type=int, help="The number of rounds.", default=5)
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
    parser.add_argument("--attack", type=str, choices=["none", "label_flipping", "sign_flipping" , "scale", "backdoor", "model_replacement"], default="model_replacement")
    parser.add_argument("--adversaries", type=str, default="0", help="Comma-separated node indices to be adversaries")
    parser.add_argument("--flip_pairs", type=str, default="0-1,2-3,4-5,6-7,8-9", help="Label pairs to flip (e.g., 0-1)")
    parser.add_argument( "--scale_factor", type=float, default=3.0, help="Boost factor for scale attack")
    parser.add_argument("--scale_on",type=str,choices=["delta", "state"],default="delta",help="Scale the delta or the whole state",)
    parser.add_argument("--save_csv", action="store_true", help="Save results to CSV files.", default=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save CSV results.", default="results/mnist")
    parser.add_argument(
        "--topology",
        type=str,
        choices=[t.value for t in TopologyType],
        default="ring",
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

def mnist(
    n: int,
    r: int,
    e: int,
    show_metrics: bool = True,
    measure_time: bool = False,
    protocol: str = "grpc",
    framework: str = "pytorch",
    aggregator: str = "fedavg",
    reduced_dataset: bool = False,
    topology: TopologyType = TopologyType.RING,
    batch_size: int = 128,  
     save_csv: bool = False,
    output_dir: str = "results/mnist",
) -> None:
    """
    P2PFL MNIST experiment.

    Args:
        n: The number of nodes.
        r: The number of rounds.
        e: The number of epochs.
        show_metrics: Show metrics.
        measure_time: Measure time.
        protocol: The protocol to use.
        framework: The framework to use.
        aggregator: The aggregator to use.
        reduced_dataset: Use a reduced dataset just for testing.
        topology: The network topology (star, full, line, ring).
        batch_size: The batch size for training.
        save_csv: Save results to CSV files.
        output_dir: Directory to save CSV results.

    """
    if measure_time:
        start_time = time.time()

    # Check settings
    if n > Settings.gossip.TTL:
        raise ValueError(
            "For in-line topology TTL must be greater than the number of nodes.Otherwise, some messages will not be delivered."
        )

    # Imports
    if framework == "tensorflow":
        from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn  # type: ignore
  
        model_fn = model_build_fn  # type: ignore
    elif framework == "pytorch":
        from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn  # type: ignore
        # from attacks.model_wrapper import AttackableLightningModel
        # model_fn = AttackableLightningModel(model_build_fn,attack=LabelFlippingAttack)
        model_fn = model_build_fn  # type: ignore
    else:
        raise ValueError(f"Framework {args.framework} not added on this example.")

    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    data.set_batch_size(batch_size)
    partitions = data.generate_partitions(
        n * 50 if reduced_dataset else n,
        # DirichletPartitionStrategy,
        # params={"alpha": 0.5},
        RandomIIDPartitionStrategy,  # type: ignore
    )
    

    # Node Creation
    nodes = []
    adversary_indices = [int(x) for x in args.adversaries.split(",")] if args.adversaries else []
    clear_attacks()
    for i in range(n):
        address = f"node-{i}" if protocol == "memory" else f"unix:///tmp/p2pfl-{i}.sock" if protocol == "unix" else "127.0.0.1"
        # Build attack object
        attack_obj: Optional[BaseAttack] = None
        original_model = model_fn()
        if i in adversary_indices and args.attack != "none":
            if args.attack == "label_flipping":
                attack_obj = LabelFlippingAttack(flip_map=args.flip_map)
             # Apply data poisoning
                partitions[i] = attack_obj.poison_data(partitions[i])
            elif args.attack == "sign_flipping":
                attack_obj = SignFlippingAttack(scale=-3.0)
                original_get_params = original_model.get_parameters
                # print(f"the original params are : {original_model.get_parameters()}" ) #just for checking if the signs are actually flipped
                def poisoned_get_parameters(attack_obj = attack_obj):
                    params = original_get_params()  # List[np.ndarray]
                    return attack_obj.manipulate_update(params)

                original_model.get_parameters = poisoned_get_parameters
                print(f"[Node {i}] Patched get_parameters for sign flipping")
                # print(f"the poisoned params are : {original_model.get_parameters()}" ) #just for checking if the signs are actually flipped
            elif args.attack == "scale":
                attack_obj = ScaleAttack(factor=args.scale_factor, apply_on=args.scale_on)

                original_get_params = original_model.get_parameters

                def poisoned_get_parameters(atk=attack_obj):
                    params = original_get_params()
                    return atk.manipulate_update(params)

               
                print(f"[Node {i}] ScaleAttack activated ×{args.scale_factor} on {args.scale_on}")
            elif args.attack == "backdoor":
                attack_obj = BackdoorAttack(
                    trigger_size=4,
                    target_class=2,
                    poison_rate=0.3
                )
            elif args.attack == "model_replacement":
                attack_obj = ModelReplacementAttack(
                    scaling_factor=50.0,    # 1000–10000 = instant takeover
                    trigger_size=16,
                    target_class=2,
                    poison_rate=1
                )
                print(f"[Node {i}] MODEL REPLACEMENT ATTACK ACTIVATED (scaling={5000})")
       # Build model
       
        # if attack_obj:
        #     model_to_use = PoisonedLightningModel(original_model, attack=attack_obj)
        # else:
        #     model_to_use = PoisonedLightningModel(original_model, attack=None)
        model_to_use = PoisonedLightningModel(original_model.model, node_addr=address)
        node = Node(
            model_to_use,
            partitions[i],
            protocol=MemoryCommunicationProtocol() if protocol == "memory" else GrpcCommunicationProtocol(),
            addr=address,
            aggregator=Scaffold() if aggregator == "scaffold" else None,
        )
        node.start()
        if attack_obj:
            register_attack(address, attack_obj)
            attack_obj.on_attach(node)
        logger.info(node.addr, f"node: {i}")
        logger.info(node.addr,f"Node {i} | Adversary: {i in adversary_indices} | Attack: {args.attack if i in adversary_indices else 'N/A'}")
        nodes.append(node)

    try:
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)

        wait_convergence(nodes, n - 1, only_direct=False, wait=60)  # type: ignore

        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        # Start Learning
        nodes[0].set_start_learning(rounds=r, epochs=e, trainset_size=5) #at first assuming all nodes participating in the learning

        # Wait and check
        wait_to_finish(nodes, timeout=60 * 60)  # 1 hour
        # Local Logs
        if show_metrics:
            local_logs = logger.get_local_logs()
            # if local_logs != {}:
            #     logs_l = list(local_logs.items())[0][1]
            #     #  Plot experiment metrics
            #     for round_num, round_metrics in logs_l.items():
            #         for node_name, node_metrics in round_metrics.items():
            #             for metric, values in node_metrics.items():
            #                 x, y = zip(*values, strict=False)
            #                 plt.plot(x, y, label=metric)
            #                 # Add a red point to the last data point
            #                 plt.scatter(x[-1], y[-1], color="red")
            #                 plt.title(f"Round {round_num} - {node_name}")
            #                 plt.xlabel("Epoch")
            #                 plt.ylabel(metric)
            #                 plt.legend()
            #                 plt.show()

            # Global Logs
            global_logs = logger.get_global_logs()
            all_metrics = {}
            rows = []
            if global_logs != {}:
                logs_g = list(global_logs.items())[0][1]  # Accessing the nested dictionary directly
                # Plot experiment metrics
                for node_name, node_metrics in logs_g.items():
                    safe_node_name = re.sub(r'[^a-zA-Z0-9_-]', '_', node_name)
                    for metric, values in node_metrics.items():
                        x, y = zip(*values, strict=False)
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].append(y)
                        for round, val in values:
                            rows.append({"node": node_name, "metric": metric, "round": round, "value": val})
                        plt.plot(x, y, label=metric)
                        # Add a red point to the last data point
                        plt.scatter(x[-1], y[-1], color="red")
                        plt.title(f"{node_name} - {metric}")
                        plt.xlabel("Epoch")
                        plt.ylabel(metric)
                        plt.legend()
                        plt.savefig(f"results/{safe_node_name}_{metric}.png", dpi=300,bbox_inches="tight")
                        plt.show()
                        
                for metric in ["test_loss", "test_metric"]:
                    plt.figure()
                    for node_name, node_metrics in logs_g.items():
                        x, y = zip(*node_metrics[metric], strict=False)
                        plt.plot(x, y, label=node_name)
                    plt.title(f"{metric} comparison between nodes")
                    plt.xlabel("Epoch")
                    plt.ylabel(metric)
                    plt.legend()
                    plt.savefig(f"results/{safe_node_name}_{metric}_3.png", dpi=300,bbox_inches="tight")
                    plt.show()    
            df = pd.DataFrame(rows)
            print(rows[0])
            df.to_csv("results/metrics_all_nodes.csv", index=False)

    except Exception as e:
        raise e
    finally:
        # Stop Nodes
        for node in nodes:
            node.stop()

        if measure_time:
            print("--- %s seconds ---" % (time.time() - start_time))

        # Save CSV results if requested
        if save_csv:
            output_path = Path(output_dir)
            save_experiment_results(output_path, start_time if measure_time else None)



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

    # Seed
    if args.seed is not None:
        Settings.general.SEED = args.seed

 

    # Launch experiment
    try:
        mnist(
            args.nodes,
            args.rounds,
            args.epochs,
            show_metrics=args.show_metrics,
            measure_time=args.measure_time,
            protocol=args.protocol,
            framework=args.framework,
            aggregator=args.aggregator,
            reduced_dataset=args.reduced_dataset,
            topology=args.topology,
            batch_size=args.batch_size,
            save_csv=args.save_csv,
            output_dir=args.output_dir,
        )
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
