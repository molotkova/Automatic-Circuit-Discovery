import time
from typing import List, Dict, Any

import sys

from experiments.robustness.config import ExperimentConfig, ExperimentResult
from experiments.robustness.core import CircuitLoader, MetricComputer


class BaselineJaccardAnalysis:
    """
    Compare baseline circuit similarity with other circuits.
    
    This experiment takes a single baseline run ID and a list of other run IDs,
    computes Jaccard indices for nodes and edges between the baseline and each
    other circuit, and reports indices for each pair along with average and
    standard deviation.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the baseline Jaccard analysis experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.loader = CircuitLoader(config)
        self.metrics = MetricComputer(config)
    
    def run(self, baseline_run_id: str, run_ids: List[str]) -> ExperimentResult:
        """
        Run the baseline Jaccard similarity analysis experiment.
        
        Args:
            baseline_run_id: Run ID of the baseline circuit
            run_ids: List of other W&B run IDs to compare with baseline
            
        Returns:
            ExperimentResult containing baseline Jaccard indices and statistics
        """
        if self.config.verbose:
            print(f"Running baseline Jaccard analysis with baseline {baseline_run_id}...")
        
        # Include baseline in the run_ids for loading
        all_run_ids = [baseline_run_id] + run_ids
        circuit_batch = self.loader.load_circuits_batch(all_run_ids)
        
        # Compute baseline Jaccard indices
        baseline_results = self.metrics.compute_baseline_jaccard_indices(baseline_run_id, circuit_batch)
        
        # Compute statistics for edges and nodes separately
        summary = self.metrics.compute_jaccard_statistics(baseline_results)
        
        # Create experiment result
        result = ExperimentResult(
            experiment_type="baseline_jaccard_similarity",
            run_ids=all_run_ids,
            results={
                "baseline_run_id": baseline_run_id,
                "baseline_jaccard_indices": baseline_results,
                "summary": summary
            },
            metadata={
                "baseline_run_id": baseline_run_id,
                "total_circuits": len(run_ids),
                "total_comparisons": len(baseline_results)
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config.to_dict()
        )
        
        if self.config.verbose:
            print(f"Baseline Jaccard analysis complete:")
            print(f"  Baseline: {baseline_run_id}")
            print(f"  Comparisons: {len(baseline_results)}")
            print(f"  Edges - Mean: {summary['edges']['mean']:.4f}, Std: {summary['edges']['std']:.4f}")
            print(f"  Nodes - Mean: {summary['nodes']['mean']:.4f}, Std: {summary['nodes']['std']:.4f}")
        
        return result
        
