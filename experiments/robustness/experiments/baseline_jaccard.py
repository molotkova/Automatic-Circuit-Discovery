import time
from typing import List, Dict, Any

import sys
sys.path.append('.')

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
    
    def get_most_similar_to_baseline(self, result: ExperimentResult, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the circuits most similar to the baseline.
        
        Args:
            result: ExperimentResult from baseline Jaccard analysis
            top_k: Number of top similar circuits to return
            
        Returns:
            List of dictionaries containing circuit information and similarity scores
        """
        baseline_results = result.results["baseline_jaccard_indices"]
        circuits = []
        
        # Collect all circuits with their similarity scores
        for run_id, jaccard_data in baseline_results.items():
            circuits.append({
                "run_id": run_id,
                "edges_similarity": jaccard_data["edges"],
                "nodes_similarity": jaccard_data["nodes"]
            })
        
        # Sort by edges similarity (descending)
        circuits.sort(key=lambda x: x["edges_similarity"], reverse=True)
        
        return circuits[:top_k]
    
    def get_least_similar_to_baseline(self, result: ExperimentResult, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the circuits least similar to the baseline.
        
        Args:
            result: ExperimentResult from baseline Jaccard analysis
            top_k: Number of least similar circuits to return
            
        Returns:
            List of dictionaries containing circuit information and similarity scores
        """
        baseline_results = result.results["baseline_jaccard_indices"]
        circuits = []
        
        # Collect all circuits with their similarity scores
        for run_id, jaccard_data in baseline_results.items():
            circuits.append({
                "run_id": run_id,
                "edges_similarity": jaccard_data["edges"],
                "nodes_similarity": jaccard_data["nodes"]
            })
        
        # Sort by edges similarity (ascending)
        circuits.sort(key=lambda x: x["edges_similarity"])
        
        return circuits[:top_k]
    
    def get_similarity_distribution(self, result: ExperimentResult) -> Dict[str, Dict[str, float]]:
        """
        Get similarity distribution statistics.
        
        Args:
            result: ExperimentResult from baseline Jaccard analysis
            
        Returns:
            Dictionary containing distribution statistics for edges and nodes
        """
        baseline_results = result.results["baseline_jaccard_indices"]
        
        edges_values = [data["edges"] for data in baseline_results.values()]
        nodes_values = [data["nodes"] for data in baseline_results.values()]
        
        return {
            "edges": self.metrics.compute_statistics(edges_values),
            "nodes": self.metrics.compute_statistics(nodes_values)
        }
    
    def compare_with_baseline(self, result: ExperimentResult, threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        Categorize circuits based on similarity threshold.
        
        Args:
            result: ExperimentResult from baseline Jaccard analysis
            threshold: Similarity threshold for categorization
            
        Returns:
            Dictionary containing categorized circuit lists
        """
        baseline_results = result.results["baseline_jaccard_indices"]
        
        similar = []
        dissimilar = []
        
        for run_id, jaccard_data in baseline_results.items():
            if jaccard_data["edges"] >= threshold:
                similar.append(run_id)
            else:
                dissimilar.append(run_id)
        
        return {
            "similar": similar,
            "dissimilar": dissimilar,
            "threshold": threshold
        }
