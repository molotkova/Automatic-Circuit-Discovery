import time
from typing import List, Dict, Any, Union, Optional

import sys

from experiments.robustness.config import ExperimentConfig, ExperimentResult, CircuitBatch
from experiments.robustness.core import CircuitLoader, MetricComputer


class JaccardCrossSimilarityAnalysis:
    """
    Compare baseline circuit similarity with other circuits.

    This experiment takes a single baseline run ID and a list of other run IDs,
    computes Jaccard indices for nodes and edges between the baseline and each
    other circuit, and reports indices for each pair along with average and
    standard deviation.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the jaccard cross similarity analysis experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.loader = CircuitLoader(config)
        self.metrics = MetricComputer(config)

    def run(
        self, 
        baseline_run_ids: Union[List[str], Dict[str, List[str]]], 
        run_ids: Optional[List[str]] = None, 
        circuit_batch: Optional[CircuitBatch] = None
    ) -> ExperimentResult:
        """
        Run the jaccard cross similarity analysis experiment.

        Args:
            baseline_run_ids: Either:
                - List of baseline run IDs (original format)
                - Dict mapping baseline_id -> [circuit_ids] (new format)
            run_ids: List of other W&B run IDs to compare with baseline(s) (only used if baseline_run_ids is List[str])
            circuit_batch: Optional pre-loaded circuit batch for caching

        Returns:
            ExperimentResult containing baseline Jaccard indices and statistics
        """
        if isinstance(baseline_run_ids, dict):
            # New dict format: {baseline_id: [circuit_ids]}
            return self._run_dict_format(baseline_run_ids, circuit_batch)
        else:
            # Original format: baseline_run_ids + run_ids
            if run_ids is None:
                raise ValueError("run_ids must be provided when baseline_run_ids is a list")
            return self._run_list_format(baseline_run_ids, run_ids, circuit_batch)

    def _run_list_format(
        self, 
        baseline_run_ids: List[str], 
        run_ids: List[str], 
        circuit_batch: Optional[CircuitBatch]
    ) -> ExperimentResult:
        """Handle original list format: baseline_run_ids + run_ids."""
        if self.config.verbose:
            if len(baseline_run_ids) == 1:
                print(f"Running jaccard cross similarity analysis with baseline {baseline_run_ids[0]}...")
            else:
                print(f"Running jaccard cross similarity analysis with {len(baseline_run_ids)} baselines...")

        # Include all baselines in the run_ids for result metadata
        all_run_ids = baseline_run_ids + run_ids
        
        # Load circuits in batch if not provided
        if circuit_batch is None:
            circuit_batch = self.loader.load_circuits_batch(all_run_ids)

        if len(baseline_run_ids) == 1:
            # Single baseline case - use original method
            baseline_results = self.metrics.compute_baseline_jaccard_indices(
                baseline_run_ids[0], circuit_batch, run_ids
            )
            summary = self.metrics.compute_jaccard_statistics(baseline_results)
            
            result = ExperimentResult(
                experiment_type="jaccard_cross_similarity",
                run_ids=all_run_ids,
                results={
                    "baseline_run_id": baseline_run_ids[0],
                    "baseline_jaccard_indices": baseline_results,
                    "summary": summary,
                },
                metadata={
                    "baseline_run_id": baseline_run_ids[0],
                    "total_circuits": len(run_ids),
                    "total_comparisons": len(baseline_results),
                },
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                config=self.config.to_dict(),
            )
        else:
            # Multiple baselines case
            multiple_baseline_results = self.metrics.compute_multiple_baseline_jaccard_indices(
                baseline_run_ids, circuit_batch, run_ids
            )
            summaries = self.metrics.compute_multiple_baseline_jaccard_summaries(multiple_baseline_results)
            
            result = ExperimentResult(
                experiment_type="jaccard_cross_similarity",
                run_ids=all_run_ids,
                results={
                    "baseline_run_ids": baseline_run_ids,
                    "multiple_baseline_jaccard_indices": multiple_baseline_results,
                    "summaries": summaries,
                },
                metadata={
                    "baseline_run_ids": baseline_run_ids,
                    "num_baselines": len(baseline_run_ids),
                    "total_circuits": len(run_ids),
                    "total_comparisons": sum(len(results) for results in multiple_baseline_results.values()),
                },
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                config=self.config.to_dict(),
            )

        if self.config.verbose:
            if len(baseline_run_ids) == 1:
                print(f"Jaccard cross similarity analysis complete:")
                print(f"  Baseline: {baseline_run_ids[0]}")
                print(f"  Comparisons: {len(baseline_results)}")
                print(
                    f"  Edges - Mean: {summary['edges']['mean']:.4f}, Std: {summary['edges']['std']:.4f}"
                )
                print(
                    f"  Nodes - Mean: {summary['nodes']['mean']:.4f}, Std: {summary['nodes']['std']:.4f}"
                )
            else:
                print(f"Jaccard cross similarity analysis complete:")
                print(f"  Baselines: {baseline_run_ids}")
                print(f"  Total comparisons: {result.metadata['total_comparisons']}")
                overall_summary = summaries["overall"]
                print(
                    f"  Overall Edges - Mean: {overall_summary['edges']['mean']:.4f}, Std: {overall_summary['edges']['std']:.4f}"
                )
                print(
                    f"  Overall Nodes - Mean: {overall_summary['nodes']['mean']:.4f}, Std: {overall_summary['nodes']['std']:.4f}"
                )

        return result

    def _run_dict_format(
        self, 
        baseline_to_circuits: Dict[str, List[str]], 
        circuit_batch: Optional[CircuitBatch]
    ) -> ExperimentResult:
        """Handle dict format: {baseline_id: [circuit_ids]}."""
        if self.config.verbose:
            print(f"Running jaccard cross similarity analysis with {len(baseline_to_circuits)} baseline(s)...")

        # Collect all unique run IDs
        all_run_ids = []
        for baseline_id, circuit_ids in baseline_to_circuits.items():
            all_run_ids.extend([baseline_id] + circuit_ids)
        all_run_ids = list(set(all_run_ids))  # Remove duplicates
        
        # Load circuits in batch if not provided
        if circuit_batch is None:
            circuit_batch = self.loader.load_circuits_batch(all_run_ids)

        # Process each baseline separately
        multiple_baseline_results = {}
        for baseline_id, circuit_ids in baseline_to_circuits.items():
            baseline_results = self.metrics.compute_baseline_jaccard_indices(
                baseline_id, circuit_batch, circuit_ids
            )
            multiple_baseline_results[baseline_id] = baseline_results

        # Compute summaries
        summaries = self.metrics.compute_multiple_baseline_jaccard_summaries(multiple_baseline_results)
        
        result = ExperimentResult(
            experiment_type="jaccard_cross_similarity",
            run_ids=all_run_ids,
            results={
                "baseline_to_circuits": baseline_to_circuits,
                "multiple_baseline_jaccard_indices": multiple_baseline_results,
                "summaries": summaries,
            },
            metadata={
                "baseline_to_circuits": baseline_to_circuits,
                "num_baselines": len(baseline_to_circuits),
                "total_circuits": sum(len(circuit_ids) for circuit_ids in baseline_to_circuits.values()),
                "total_comparisons": sum(len(results) for results in multiple_baseline_results.values()),
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config.to_dict(),
        )

        if self.config.verbose:
            print(f"Jaccard cross similarity analysis complete:")
            print(f"  Baselines: {list(baseline_to_circuits.keys())}")
            print(f"  Total comparisons: {result.metadata['total_comparisons']}")
            overall_summary = summaries["overall"]
            print(
                f"  Overall Edges - Mean: {overall_summary['edges']['mean']:.4f}, Std: {overall_summary['edges']['std']:.4f}"
            )
            print(
                f"  Overall Nodes - Mean: {overall_summary['nodes']['mean']:.4f}, Std: {overall_summary['nodes']['std']:.4f}"
            )

        return result
