import time
from typing import List, Dict, Any

import sys


from experiments.robustness.config import ExperimentConfig, ExperimentResult
from experiments.robustness.core import CircuitLoader, MetricComputer


class PairwiseJaccardAnalysis:
    """
    Experiment 2: Compute pairwise Jaccard indices between all circuits.

    This experiment takes a list of run IDs, loads the corresponding circuits,
    computes pairwise Jaccard indices for nodes and edges, and reports indices
    for each pair along with average and standard deviation.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the pairwise Jaccard analysis experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.loader = CircuitLoader(config)
        self.metrics = MetricComputer(config)

    def run(self, run_ids: List[str]) -> ExperimentResult:
        """
        Run the pairwise Jaccard similarity analysis experiment.

        Args:
            run_ids: List of W&B run IDs to analyze

        Returns:
            ExperimentResult containing pairwise Jaccard indices and statistics
        """
        if self.config.verbose:
            print(f"Running pairwise Jaccard analysis for {len(run_ids)} circuits...")

        # Load circuits in batch
        circuit_batch = self.loader.load_circuits_batch(run_ids)

        # Compute pairwise Jaccard indices
        pairwise_results = self.metrics.compute_pairwise_jaccard_indices(circuit_batch)

        # Compute statistics for edges and nodes separately
        summary = self.metrics.compute_pairwise_jaccard_statistics(pairwise_results)

        # Create experiment result
        result = ExperimentResult(
            experiment_type="pairwise_jaccard_similarity",
            run_ids=run_ids,
            results={"pairwise_jaccard_indices": pairwise_results, "summary": summary},
            metadata={
                "total_circuits": len(run_ids),
                "total_pairs": len(run_ids) * (len(run_ids) - 1) // 2,
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config.to_dict(),
        )

        if self.config.verbose:
            print(f"Pairwise Jaccard analysis complete:")
            print(f"  Total pairs: {result.metadata['total_pairs']}")
            print(
                f"  Edges - Mean: {summary['edges']['mean']:.4f}, Std: {summary['edges']['std']:.4f}"
            )
            print(
                f"  Nodes - Mean: {summary['nodes']['mean']:.4f}, Std: {summary['nodes']['std']:.4f}"
            )

        return result
