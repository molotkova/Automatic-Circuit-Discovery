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
    Experiment 4: Analyze baseline circuit robustness via relative logit difference changes.
    
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
        
        # Compute baseline logit difference
        baseline_logit_diff = self.metrics.compute_logit_differences(circuit_batch)[baseline_run_id]
        
        # Compute average logit difference for other circuits
        other_logit_diffs = [self.metrics.compute_logit_differences(circuit_batch)[run_id] for run_id in run_ids]
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
    
    def get_robustness_ranking(self, result: ExperimentResult) -> List[Dict[str, Any]]:
        """
        Get circuits ranked by robustness (relative change).
        
        Args:
            result: ExperimentResult from baseline logit diff analysis
            
        Returns:
            List of dictionaries containing circuit information and robustness scores
        """
        relative_changes = result.results["relative_changes"]
        circuits = []
        
        # Collect all circuits with their robustness scores
        for run_id, relative_change in relative_changes.items():
            circuits.append({
                "run_id": run_id,
                "relative_change": relative_change,
                "robustness_score": 1.0 - abs(relative_change)  # Higher is more robust
            })
        
        # Sort by robustness score (descending)
        circuits.sort(key=lambda x: x["robustness_score"], reverse=True)
        
        return circuits
    
    def get_most_robust_circuits(self, result: ExperimentResult, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most robust circuits (closest to baseline performance).
        
        Args:
            result: ExperimentResult from baseline logit diff analysis
            top_k: Number of most robust circuits to return
            
        Returns:
            List of dictionaries containing circuit information and robustness scores
        """
        ranking = self.get_robustness_ranking(result)
        return ranking[:top_k]
    
    def get_least_robust_circuits(self, result: ExperimentResult, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the least robust circuits (furthest from baseline performance).
        
        Args:
            result: ExperimentResult from baseline logit diff analysis
            top_k: Number of least robust circuits to return
            
        Returns:
            List of dictionaries containing circuit information and robustness scores
        """
        ranking = self.get_robustness_ranking(result)
        return ranking[-top_k:]
    
    def get_robustness_categories(self, result: ExperimentResult, thresholds: Dict[str, float] = None) -> Dict[str, List[str]]:
        """
        Categorize circuits based on robustness thresholds.
        
        Args:
            result: ExperimentResult from baseline logit diff analysis
            thresholds: Dictionary with 'high', 'medium', 'low' thresholds
            
        Returns:
            Dictionary containing categorized circuit lists
        """
        if thresholds is None:
            thresholds = {
                "high": 0.1,    # Very robust (small relative change)
                "medium": 0.3,  # Moderately robust
                "low": 0.5      # Less robust (large relative change)
            }
        
        relative_changes = result.results["relative_changes"]
        
        high_robust = []
        medium_robust = []
        low_robust = []
        
        for run_id, relative_change in relative_changes.items():
            abs_change = abs(relative_change)
            if abs_change <= thresholds["high"]:
                high_robust.append(run_id)
            elif abs_change <= thresholds["medium"]:
                medium_robust.append(run_id)
            else:
                low_robust.append(run_id)
        
        return {
            "high_robust": high_robust,
            "medium_robust": medium_robust,
            "low_robust": low_robust,
            "thresholds": thresholds
        }
    
    def get_robustness_summary(self, result: ExperimentResult) -> Dict[str, Any]:
        """
        Get comprehensive robustness summary.
        
        Args:
            result: ExperimentResult from baseline logit diff analysis
            
        Returns:
            Dictionary containing robustness summary statistics
        """
        relative_changes = result.results["relative_changes"]
        values = list(relative_changes.values())
        
        # Compute additional statistics
        positive_changes = [v for v in values if v > 0]
        negative_changes = [v for v in values if v < 0]
        
        return {
            "total_circuits": len(values),
            "baseline_logit_diff": result.results["baseline_logit_diff"],
            "average_logit_diff": result.results["average_logit_diff"],
            "relative_changes_stats": result.results["summary"],
            "positive_changes": len(positive_changes),
            "negative_changes": len(negative_changes),
            "improvement_rate": len(positive_changes) / len(values) if values else 0.0,
            "degradation_rate": len(negative_changes) / len(values) if values else 0.0
        }
