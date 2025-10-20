import time
from typing import List, Dict, Any, Tuple
import statistics

import sys

from experiments.robustness.config import ExperimentConfig, CircuitBatch
from utils.circuit_utils import (
    compute_jaccard_index_edges,
    compute_jaccard_index_nodes,
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

    def _get_all_test_metrics(self, circuit_batch: CircuitBatch, data: Any) -> Dict[str, float]:
        """
        Compute all test metrics for given data.
        
        Args:
            circuit_batch: CircuitBatch containing experiment setup
            data: Input data to compute metrics on
            
        Returns:
            Dictionary of computed metrics
        """
        return {
            f"test_{name}": fn(data).item() for name, fn in circuit_batch.things.test_metrics.items()
        }

    def _compute_logit_diff_for_circuit(
        self, circuit: Any, circuit_batch: CircuitBatch, run_id: str = None
    ) -> float:
        """
        Compute logit difference for a single circuit.
        
        Args:
            circuit: Circuit correspondence object
            circuit_batch: CircuitBatch containing experiment setup
            run_id: Optional run ID for logging
            
        Returns:
            Logit difference value
        """
        if self.config.verbose and run_id:
            print(f"  Computing logit diff for {run_id}...")

        metrics = circuit_batch.experiment.call_metric_with_corr(
            circuit, lambda data: self._get_all_test_metrics(circuit_batch, data), circuit_batch.things.test_data
        )
        return metrics["test_logit_diff"]

    def _filter_circuit_batch(self, circuit_batch: CircuitBatch, run_ids: List[str]) -> CircuitBatch:
        """
        Filter a circuit batch to only include circuits for the specified run_ids.
        
        Args:
            circuit_batch: Full circuit batch to filter
            run_ids: List of run IDs to keep in the filtered batch
            
        Returns:
            Filtered CircuitBatch containing only the specified circuits
        """
        # Check that all requested run_ids exist in the batch
        missing_ids = [run_id for run_id in run_ids if run_id not in circuit_batch.circuits]
        if missing_ids:
            raise ValueError(f"Run IDs not found in circuit batch: {missing_ids}")
        
        # Filter circuits and metadata
        filtered_circuits = {run_id: circuit_batch.circuits[run_id] for run_id in run_ids}
        filtered_metadata = {run_id: circuit_batch.run_metadata[run_id] for run_id in run_ids}
        
        # Create new filtered batch with same experiment setup
        filtered_batch = CircuitBatch(
            circuits=filtered_circuits,
            experiment=circuit_batch.experiment,
            things=circuit_batch.things,
            run_metadata=filtered_metadata,
            config=circuit_batch.config,
            timestamp=circuit_batch.timestamp,
        )
        
        if self.config.verbose:
            print(f"Filtered circuit batch: {len(circuit_batch.circuits)} -> {len(filtered_circuits)} circuits")
        
        return filtered_batch

    def compute_logit_differences(
        self, circuit_batch: CircuitBatch, run_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute logit differences for each circuit in the batch.

        Args:
            circuit_batch: CircuitBatch containing circuits and experiment setup
            run_ids: List of run IDs to compute logit differences for

        Returns:
            Dictionary mapping run_id to logit difference value
        """
        if self.config.verbose:
            print("Computing logit differences...")

        # Filter circuit batch to only include requested run_ids
        filtered_circuit_batch = self._filter_circuit_batch(circuit_batch, run_ids)

        # Ensure corrupted cache is set up
        filtered_circuit_batch.experiment.setup_corrupted_cache()

        # Compute for each circuit
        logit_diffs = {}
        for run_id in filtered_circuit_batch.run_ids:
            circuit = filtered_circuit_batch.get_circuit(run_id)
            logit_diffs[run_id] = self._compute_logit_diff_for_circuit(
                circuit, filtered_circuit_batch, run_id
            )

        if self.config.verbose:
            print(f"Computed logit differences for {len(logit_diffs)} circuits")

        return logit_diffs

    def compute_pairwise_jaccard_indices(
        self, circuit_batch: CircuitBatch, run_ids: List[str]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute pairwise Jaccard indices between all circuits.

        Args:
            circuit_batch: CircuitBatch containing circuits
            run_ids: List of run IDs to compute pairwise Jaccard indices for

        Returns:
            Nested dictionary: {run_id1: {run_id2: {"edges": jaccard_edges, "nodes": jaccard_nodes}}}
        """
        if self.config.verbose:
            print("Computing pairwise Jaccard indices...")

        # Filter circuit batch to only include requested run_ids
        filtered_circuit_batch = self._filter_circuit_batch(circuit_batch, run_ids)

        pairwise_results = {}

        for i, run_id1 in enumerate(run_ids):
            pairwise_results[run_id1] = {}
            circuit1 = filtered_circuit_batch.get_circuit(run_id1)

            for j, run_id2 in enumerate(run_ids):
                if i <= j:  # Only compute upper triangle + diagonal
                    circuit2 = filtered_circuit_batch.get_circuit(run_id2)

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
        self, baseline_run_id: str, circuit_batch: CircuitBatch, run_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute Jaccard indices between baseline circuit and all other circuits.

        Args:
            baseline_run_id: Run ID of the baseline circuit
            circuit_batch: CircuitBatch containing circuits
            run_ids: List of run IDs to compute Jaccard indices for (excluding baseline)

        Returns:
            Dictionary mapping run_id to {"edges": jaccard_edges, "nodes": jaccard_nodes}
        """
        if self.config.verbose:
            print(f"Computing baseline Jaccard indices with {baseline_run_id}...")

        # Include baseline in the filtering
        all_run_ids = [baseline_run_id] + run_ids
        filtered_circuit_batch = self._filter_circuit_batch(circuit_batch, all_run_ids)

        if baseline_run_id not in filtered_circuit_batch.run_ids:
            raise ValueError(
                f"Baseline run ID {baseline_run_id} not found in circuit batch"
            )

        baseline_circuit = filtered_circuit_batch.get_circuit(baseline_run_id)
        baseline_results = {}

        for run_id in run_ids:
            circuit = filtered_circuit_batch.get_circuit(run_id)
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

    def compute_multiple_baseline_jaccard_indices(
        self, baseline_run_ids: List[str], circuit_batch: CircuitBatch, run_ids: List[str]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute Jaccard indices between multiple baseline circuits and all other circuits.

        Args:
            baseline_run_ids: List of baseline run IDs
            circuit_batch: CircuitBatch containing circuits
            run_ids: List of run IDs to compute Jaccard indices for (excluding baselines)

        Returns:
            Dictionary mapping baseline_run_id to {run_id: {"edges": jaccard_edges, "nodes": jaccard_nodes}}
        """
        if self.config.verbose:
            print(f"Computing multiple baseline Jaccard indices with {len(baseline_run_ids)} baselines...")

        # Include all baselines in the filtering
        all_run_ids = baseline_run_ids + run_ids
        filtered_circuit_batch = self._filter_circuit_batch(circuit_batch, all_run_ids)

        # Validate all baseline IDs exist
        missing_baselines = [bid for bid in baseline_run_ids if bid not in filtered_circuit_batch.run_ids]
        if missing_baselines:
            raise ValueError(f"Baseline run IDs not found in circuit batch: {missing_baselines}")

        all_results = {}
        
        for baseline_run_id in baseline_run_ids:
            baseline_results = self.compute_baseline_jaccard_indices(baseline_run_id, filtered_circuit_batch, run_ids)
            all_results[baseline_run_id] = baseline_results

        if self.config.verbose:
            print(f"Computed multiple baseline Jaccard indices for {len(baseline_run_ids)} baselines")

        return all_results



    def compute_multiple_baseline_jaccard_summaries(
        self, multiple_baseline_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """
        Compute summaries for multiple baseline Jaccard results.

        Args:
            multiple_baseline_results: Results from compute_multiple_baseline_jaccard_indices

        Returns:
            Dictionary containing per-baseline summaries and overall summary
        """
        summaries = {}
        
        # Compute per-baseline summaries
        for baseline_id, baseline_results in multiple_baseline_results.items():
            summaries[baseline_id] = self.compute_jaccard_statistics(baseline_results)
        
        # Compute overall summary across all baselines and circuits
        all_edges_values = []
        all_nodes_values = []
        
        for baseline_results in multiple_baseline_results.values():
            for circuit_result in baseline_results.values():
                all_edges_values.append(circuit_result["edges"])
                all_nodes_values.append(circuit_result["nodes"])
        
        overall_summary = {
            "edges": self.compute_statistics(all_edges_values),
            "nodes": self.compute_statistics(all_nodes_values)
        }
        
        summaries["overall"] = overall_summary
        
        return summaries


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
        
