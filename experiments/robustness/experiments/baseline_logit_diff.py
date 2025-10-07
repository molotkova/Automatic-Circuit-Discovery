"""
Baseline Logit Difference Robustness Analysis experiment.

This module implements Experiment 4: Analyze baseline circuit robustness via relative logit difference changes.
"""

import time
from typing import List, Dict, Any

import sys


from experiments.robustness.config import ExperimentConfig, ExperimentResult, CircuitBatch
from experiments.robustness.core import CircuitLoader, MetricComputer


class BaselineLogitDiffAnalysis:
    """
    Analyze baseline circuit robustness via relative logit difference changes.

    This experiment takes a single baseline run ID and a list of other run IDs,
    computes the logit difference averaged over circuits from the list and the
    relative change of the baseline logit difference compared to the averaged
    logit difference.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the baseline logit difference analysis experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.loader = CircuitLoader(config)
        self.metrics = MetricComputer(config)

    def run(self, baseline_run_ids: List[str], run_ids: List[str], circuit_batch: CircuitBatch = None) -> ExperimentResult:
        """
        Run experiment for the logit difference relative change.

        Args:
            baseline_run_ids: List of baseline run IDs (can be single element or multiple elements)
            run_ids: List of other W&B run IDs to compare with baseline(s)
            circuit_batch: Optional pre-loaded circuit batch for caching

        Returns:
            ExperimentResult containing relative changes and robustness analysis
        """
        if self.config.verbose:
            if len(baseline_run_ids) == 1:
                print(f"Running baseline logit diff robustness analysis with baseline {baseline_run_ids[0]}...")
            else:
                print(f"Running baseline logit diff robustness analysis with {len(baseline_run_ids)} baselines...")

        # Include all baselines in the run_ids for result metadata
        all_run_ids = baseline_run_ids + run_ids
        
        # Load circuits in batch if not provided
        if circuit_batch is None:
            circuit_batch = self.loader.load_circuits_batch(all_run_ids)

        # Compute logit differences for all circuits once
        all_logit_diffs = self.metrics.compute_logit_differences(circuit_batch)

        if len(baseline_run_ids) == 1:
            # Single baseline case - use original method
            relative_changes = self.metrics.compute_baseline_logit_diff_relative_change(
                baseline_run_ids[0], circuit_batch
            )
            values = list(relative_changes.values())
            summary = self.metrics.compute_statistics(values)

            # Extract baseline and other circuit logit differences
            baseline_logit_diff = all_logit_diffs[baseline_run_ids[0]]
            other_logit_diffs = [all_logit_diffs[run_id] for run_id in run_ids]
            avg_logit_diff = (
                sum(other_logit_diffs) / len(other_logit_diffs)
                if other_logit_diffs
                else 0.0
            )

            result = ExperimentResult(
                experiment_type="baseline_logit_diff_robustness",
                run_ids=all_run_ids,
                results={
                    "baseline_run_id": baseline_run_ids[0],
                    "baseline_logit_diff": baseline_logit_diff,
                    "average_logit_diff": avg_logit_diff,
                    "relative_changes": relative_changes,
                    "summary": summary,
                },
                metadata={
                    "baseline_run_id": baseline_run_ids[0],
                    "baseline_logit_diff": baseline_logit_diff,
                    "average_logit_diff": avg_logit_diff,
                    "total_circuits": len(run_ids),
                    "total_comparisons": len(relative_changes),
                },
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                config=self.config.to_dict(),
            )
        else:
            # Multiple baselines case
            multiple_baseline_results = self.metrics.compute_multiple_baseline_logit_diff_relative_change(
                baseline_run_ids, circuit_batch
            )
            summaries = self.metrics.compute_multiple_baseline_logit_diff_summaries(multiple_baseline_results)

            # Extract baseline and other circuit logit differences
            baseline_logit_diffs = {bid: all_logit_diffs[bid] for bid in baseline_run_ids}
            other_logit_diffs = [all_logit_diffs[run_id] for run_id in run_ids]
            avg_logit_diff = (
                sum(other_logit_diffs) / len(other_logit_diffs)
                if other_logit_diffs
                else 0.0
            )

            result = ExperimentResult(
                experiment_type="baseline_logit_diff_robustness",
                run_ids=all_run_ids,
                results={
                    "baseline_run_ids": baseline_run_ids,
                    "baseline_logit_diffs": baseline_logit_diffs,
                    "average_logit_diff": avg_logit_diff,
                    "multiple_baseline_relative_changes": multiple_baseline_results,
                    "summaries": summaries,
                },
                metadata={
                    "baseline_run_ids": baseline_run_ids,
                    "num_baselines": len(baseline_run_ids),
                    "baseline_logit_diffs": baseline_logit_diffs,
                    "average_logit_diff": avg_logit_diff,
                    "total_circuits": len(run_ids),
                    "total_comparisons": sum(len(results) for results in multiple_baseline_results.values()),
                },
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                config=self.config.to_dict(),
            )

        if self.config.verbose:
            if len(baseline_run_ids) == 1:
                print(f"Baseline logit diff robustness analysis complete:")
                print(f"  Baseline: {baseline_run_ids[0]}")
                print(f"  Baseline logit diff: {baseline_logit_diff:.4f}")
                print(f"  Average logit diff: {avg_logit_diff:.4f}")
                print(
                    f"  Relative changes - Mean: {summary['mean']:.4f}, Std: {summary['std']:.4f}"
                )
            else:
                print(f"Baseline logit diff robustness analysis complete:")
                print(f"  Baselines: {baseline_run_ids}")
                print(f"  Average logit diff: {avg_logit_diff:.4f}")
                print(f"  Total comparisons: {result.metadata['total_comparisons']}")
                overall_summary = summaries["overall"]
                print(
                    f"  Overall Relative changes - Mean: {overall_summary['mean']:.4f}, Std: {overall_summary['std']:.4f}"
                )

        return result
