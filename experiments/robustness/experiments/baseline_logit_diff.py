"""
Baseline Logit Difference Robustness Analysis experiment.

This module implements Experiment 4: Analyze baseline circuit robustness via relative logit difference changes.
"""

import time
from typing import List, Dict, Any

import sys
sys.path.append('.')

from experiments.robustness.config import ExperimentConfig, ExperimentResult
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
    
    def run(self, baseline_run_id: str, run_ids: List[str]) -> ExperimentResult:
        """
        Run the baseline logit difference robustness analysis experiment.
        
        Args:
            baseline_run_id: Run ID of the baseline circuit
            run_ids: List of other W&B run IDs to compare with baseline
            
        Returns:
            ExperimentResult containing relative changes and robustness analysis
        """
        if self.config.verbose:
            print(f"Running baseline logit diff robustness analysis with baseline {baseline_run_id}...")
        
        # Include baseline in the run_ids for loading
        all_run_ids = [baseline_run_id] + run_ids
        circuit_batch = self.loader.load_circuits_batch(all_run_ids)
        
        # Compute baseline logit difference relative changes
        relative_changes = self.metrics.compute_baseline_logit_diff_relative_change(baseline_run_id, circuit_batch)
        
        # Compute statistics for relative changes
        values = list(relative_changes.values())
        summary = self.metrics.compute_statistics(values)
        
        # Compute logit differences for all circuits once
        all_logit_diffs = self.metrics.compute_logit_differences(circuit_batch)
        
        # Extract baseline and other circuit logit differences
        baseline_logit_diff = all_logit_diffs[baseline_run_id]
        other_logit_diffs = [all_logit_diffs[run_id] for run_id in run_ids]
        avg_logit_diff = sum(other_logit_diffs) / len(other_logit_diffs) if other_logit_diffs else 0.0
        
        # Create experiment result
        result = ExperimentResult(
            experiment_type="baseline_logit_diff_robustness",
            run_ids=all_run_ids,
            results={
                "baseline_run_id": baseline_run_id,
                "baseline_logit_diff": baseline_logit_diff,
                "average_logit_diff": avg_logit_diff,
                "relative_changes": relative_changes,
                "summary": summary
            },
            metadata={
                "baseline_run_id": baseline_run_id,
                "baseline_logit_diff": baseline_logit_diff,
                "average_logit_diff": avg_logit_diff,
                "total_circuits": len(run_ids),
                "total_comparisons": len(relative_changes)
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config.to_dict()
        )
        
        if self.config.verbose:
            print(f"Baseline logit diff robustness analysis complete:")
            print(f"  Baseline: {baseline_run_id}")
            print(f"  Baseline logit diff: {baseline_logit_diff:.4f}")
            print(f"  Average logit diff: {avg_logit_diff:.4f}")
            print(f"  Relative changes - Mean: {summary['mean']:.4f}, Std: {summary['std']:.4f}")
        
        return result
