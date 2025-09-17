import time
from typing import List, Dict, Any, Tuple
import statistics

import sys

from experiments.robustness.config import ExperimentConfig, CircuitBatch
from utils.circuit_utils import (
    compute_jaccard_index_edges,
    compute_jaccard_index_nodes,
    compute_logit_diff_relative_change,
)


class MetricComputer:
    """
    Computes various metrics on circuits.

    This class provides methods for computing logit differences, Jaccard indices,
    and other metrics on circuit batches.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the metrics computer.

        Args:
            config: Experiment configuration containing device and other settings
        """
        self.config = config

    def compute_logit_differences(
        self, circuit_batch: CircuitBatch
    ) -> Dict[str, float]:
        """
        Compute logit differences for each circuit in the batch.

        Args:
            circuit_batch: CircuitBatch containing circuits and experiment setup

        Returns:
            Dictionary mapping run_id to logit difference value
        """
        if self.config.verbose:
            print("Computing logit differences...")

        # Ensure corrupted cache is set up
        circuit_batch.experiment.setup_corrupted_cache()

        # Define metric function
        def get_logit_diff_metric(data: Any) -> Dict[str, float]:
            return {
                f"test_{name}": fn(data).item()
                for name, fn in circuit_batch.things.test_metrics.items()
            }

        # Compute for each circuit
        logit_diffs = {}
        for run_id, circuit in circuit_batch.circuits.items():
            if self.config.verbose:
                print(f"  Computing logit diff for {run_id}...")

            metrics = circuit_batch.experiment.call_metric_with_corr(
                circuit, get_logit_diff_metric, circuit_batch.things.test_data
            )
            logit_diffs[run_id] = metrics["test_logit_diff"]

        if self.config.verbose:
            print(f"Computed logit differences for {len(logit_diffs)} circuits")

        return logit_diffs

    def compute_pairwise_jaccard_indices(
        self, circuit_batch: CircuitBatch
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute pairwise Jaccard indices between all circuits.

        Args:
            circuit_batch: CircuitBatch containing circuits

        Returns:
            Nested dictionary: {run_id1: {run_id2: {"edges": jaccard_edges, "nodes": jaccard_nodes}}}
        """
        if self.config.verbose:
            print("Computing pairwise Jaccard indices...")

        run_ids = list(circuit_batch.circuits.keys())
        pairwise_results = {}

        for i, run_id1 in enumerate(run_ids):
            pairwise_results[run_id1] = {}
            circuit1 = circuit_batch.circuits[run_id1]

            for j, run_id2 in enumerate(run_ids):
                if i <= j:  # Only compute upper triangle + diagonal
                    circuit2 = circuit_batch.circuits[run_id2]

                    # Compute Jaccard indices
                    jaccard_edges = compute_jaccard_index_edges(circuit1, circuit2)
                    jaccard_nodes = compute_jaccard_index_nodes(circuit1, circuit2)

                    pairwise_results[run_id1][run_id2] = {
                        "edges": jaccard_edges,
                        "nodes": jaccard_nodes,
                    }

                    # Fill symmetric entry
                    if i != j:
                        if run_id2 not in pairwise_results:
                            pairwise_results[run_id2] = {}
                        pairwise_results[run_id2][run_id1] = {
                            "edges": jaccard_edges,
                            "nodes": jaccard_nodes,
                        }

        if self.config.verbose:
            print(f"Computed pairwise Jaccard indices for {len(run_ids)} circuits")

        return pairwise_results

    def compute_baseline_jaccard_indices(
        self, baseline_run_id: str, circuit_batch: CircuitBatch
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute Jaccard indices between baseline circuit and all other circuits.

        Args:
            baseline_run_id: Run ID of the baseline circuit
            circuit_batch: CircuitBatch containing circuits

        Returns:
            Dictionary mapping run_id to {"edges": jaccard_edges, "nodes": jaccard_nodes}
        """
        if self.config.verbose:
            print(f"Computing baseline Jaccard indices with {baseline_run_id}...")

        if baseline_run_id not in circuit_batch.circuits:
            raise ValueError(
                f"Baseline run ID {baseline_run_id} not found in circuit batch"
            )

        baseline_circuit = circuit_batch.circuits[baseline_run_id]
        baseline_results = {}

        for run_id, circuit in circuit_batch.circuits.items():
            if run_id != baseline_run_id:
                # Compute Jaccard indices
                jaccard_edges = compute_jaccard_index_edges(baseline_circuit, circuit)
                jaccard_nodes = compute_jaccard_index_nodes(baseline_circuit, circuit)

                baseline_results[run_id] = {
                    "edges": jaccard_edges,
                    "nodes": jaccard_nodes,
                }

        if self.config.verbose:
            print(
                f"Computed baseline Jaccard indices for {len(baseline_results)} circuits"
            )

        return baseline_results

    def compute_baseline_logit_diff_relative_change(
        self, baseline_run_id: str, circuit_batch: CircuitBatch
    ) -> Dict[str, float]:
        """
        Compute relative change in logit difference between baseline and other circuits.

        Args:
            baseline_run_id: Run ID of the baseline circuit
            circuit_batch: CircuitBatch containing circuits

        Returns:
            Dictionary mapping run_id to relative change value
        """
        if self.config.verbose:
            print(
                f"Computing baseline logit diff relative changes with {baseline_run_id}..."
            )

        if baseline_run_id not in circuit_batch.circuits:
            raise ValueError(
                f"Baseline run ID {baseline_run_id} not found in circuit batch"
            )

        baseline_circuit = circuit_batch.circuits[baseline_run_id]
        relative_changes = {}

        for run_id, circuit in circuit_batch.circuits.items():
            if run_id != baseline_run_id:
                # Compute relative change
                relative_change = compute_logit_diff_relative_change(
                    baseline_circuit, circuit
                )
                relative_changes[run_id] = relative_change

        if self.config.verbose:
            print(f"Computed relative changes for {len(relative_changes)} circuits")

        return relative_changes

    def compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Compute basic statistics for a list of values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary containing mean, std, min, max, count
        """
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def compute_jaccard_statistics(
        self, jaccard_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for Jaccard index results.

        Args:
            jaccard_results: Dictionary of Jaccard results with "edges" and "nodes" keys

        Returns:
            Dictionary containing statistics for edges and nodes separately
        """
        if not jaccard_results:
            return {
                "edges": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0},
                "nodes": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0},
            }

        # Extract edges and nodes values separately
        edges_values = [result["edges"] for result in jaccard_results.values()]
        nodes_values = [result["nodes"] for result in jaccard_results.values()]

        return {
            "edges": self.compute_statistics(edges_values),
            "nodes": self.compute_statistics(nodes_values),
        }

    def compute_pairwise_jaccard_statistics(
        self, pairwise_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for pairwise Jaccard results.

        Args:
            pairwise_results: Nested dictionary of pairwise Jaccard results

        Returns:
            Dictionary containing statistics for edges and nodes across all pairs
        """
        if not pairwise_results:
            return {
                "edges": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0},
                "nodes": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0},
            }

        # Collect all edges and nodes values (only upper triangle to avoid duplicates)
        edges_values = []
        nodes_values = []

        run_ids = list(pairwise_results.keys())
        for i, run_id1 in enumerate(run_ids):
            for j, run_id2 in enumerate(run_ids):
                if i < j:  # Only upper triangle (i < j)
                    jaccard_data = pairwise_results[run_id1][run_id2]
                    edges_values.append(jaccard_data["edges"])
                    nodes_values.append(jaccard_data["nodes"])

        return {
            "edges": self.compute_statistics(edges_values),
            "nodes": self.compute_statistics(nodes_values),
        }

    def get_metric_summary(
        self, metric_name: str, values: List[float]
    ) -> Dict[str, Any]:
        """
        Get a comprehensive summary for a metric.

        Args:
            metric_name: Name of the metric
            values: List of metric values

        Returns:
            Dictionary containing metric summary with statistics
        """
        stats = self.compute_statistics(values)

        return {
            "metric_name": metric_name,
            "statistics": stats,
            "values": values,
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def compute_all_metrics(
        self, circuit_batch: CircuitBatch, baseline_run_id: str = None
    ) -> Dict[str, Any]:
        """
        Compute all available metrics for a circuit batch.

        Args:
            circuit_batch: CircuitBatch containing circuits
            baseline_run_id: Optional baseline run ID for baseline comparisons

        Returns:
            Dictionary containing all computed metrics and statistics
        """
        if self.config.verbose:
            print("Computing all metrics...")

        results = {}

        # Compute logit differences
        logit_diffs = self.compute_logit_differences(circuit_batch)
        results["logit_differences"] = {
            "individual": logit_diffs,
            "summary": self.compute_statistics(list(logit_diffs.values())),
        }

        # Compute pairwise Jaccard indices
        pairwise_jaccard = self.compute_pairwise_jaccard_indices(circuit_batch)
        results["pairwise_jaccard"] = {
            "individual": pairwise_jaccard,
            "summary": self.compute_pairwise_jaccard_statistics(pairwise_jaccard),
        }

        # Compute baseline metrics if baseline provided
        if baseline_run_id:
            baseline_jaccard = self.compute_baseline_jaccard_indices(
                baseline_run_id, circuit_batch
            )
            results["baseline_jaccard"] = {
                "baseline_run_id": baseline_run_id,
                "individual": baseline_jaccard,
                "summary": self.compute_jaccard_statistics(baseline_jaccard),
            }

            baseline_logit_diff = self.compute_baseline_logit_diff_relative_change(
                baseline_run_id, circuit_batch
            )
            results["baseline_logit_diff"] = {
                "baseline_run_id": baseline_run_id,
                "individual": baseline_logit_diff,
                "summary": self.compute_statistics(list(baseline_logit_diff.values())),
            }

        # Add metadata
        results["metadata"] = {
            "num_circuits": circuit_batch.num_circuits,
            "run_ids": circuit_batch.run_ids,
            "baseline_run_id": baseline_run_id,
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if self.config.verbose:
            print(f"Computed all metrics")

        return results
