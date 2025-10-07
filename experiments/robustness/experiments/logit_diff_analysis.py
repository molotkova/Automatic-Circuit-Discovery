import time
from typing import List, Dict, Any

import sys

from experiments.robustness.config import ExperimentConfig, ExperimentResult, CircuitBatch
from experiments.robustness.core import CircuitLoader, MetricComputer


class LogitDiffAnalysis:
    """
    Analyze logit differences across multiple circuits.

    This experiment takes a list of run IDs, loads the corresponding circuits,
    computes the logit difference metric for each circuit, and reports the
    individual logit differences, average, and standard deviation.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the logit difference analysis experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.loader = CircuitLoader(config)
        self.metrics = MetricComputer(config)

    def run(self, run_ids: List[str], circuit_batch: CircuitBatch = None) -> ExperimentResult:
        """
        Run the logit difference calibration experiment.
        For the batch of circuits, compute logit difference produced by each circuit and aggregate statistics.

        Args:
            run_ids: List of W&B run IDs to analyze
            circuit_batch: Optional pre-loaded circuit batch for caching

        Returns:
            ExperimentResult containing individual logit differences and statistics
        """
        if self.config.verbose:
            print(f"Running logit difference analysis for {len(run_ids)} circuits...")

        # Load circuits in batch if not provided
        if circuit_batch is None:
            circuit_batch = self.loader.load_circuits_batch(run_ids)

        # Compute logit differences
        logit_diffs = self.metrics.compute_logit_differences(circuit_batch)

        # Compute statistics
        values = list(logit_diffs.values())
        summary = self.metrics.compute_statistics(values)

        # Create experiment result
        result = ExperimentResult(
            experiment_type="logit_difference_analysis",
            run_ids=run_ids,
            results={"individual_logit_diffs": logit_diffs, "summary": summary},
            metadata={
                "total_circuits": len(run_ids),
                "successfully_loaded": len(logit_diffs),
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config.to_dict(),
        )

        if self.config.verbose:
            print(f"Logit difference analysis complete:")
            print(f"  Individual results: {len(logit_diffs)} circuits")
            print(f"  Mean: {summary['mean']:.4f}")
            print(f"  Std: {summary['std']:.4f}")
            print(f"  Min: {summary['min']:.4f}")
            print(f"  Max: {summary['max']:.4f}")

        return result

    def run_with_baseline(
        self, run_ids: List[str], baseline_run_id: str
    ) -> ExperimentResult:
        """
        Run logit difference analysis with baseline comparison.

        Args:
            run_ids: List of W&B run IDs to analyze
            baseline_run_id: Baseline run ID for comparison

        Returns:
            ExperimentResult containing analysis with baseline context
        """
        if self.config.verbose:
            print(
                f"Running logit difference analysis with baseline {baseline_run_id}..."
            )

        # Include baseline in run_ids for loading
        all_run_ids = [baseline_run_id] + run_ids

        # Run standard analysis
        result = self.run(all_run_ids)

        # Update metadata to include baseline information
        result.metadata["baseline_run_id"] = baseline_run_id
        result.metadata["baseline_logit_diff"] = result.results[
            "individual_logit_diffs"
        ].get(baseline_run_id, 0.0)

        # Update experiment type
        result.experiment_type = "logit_difference_analysis_with_baseline"

        if self.config.verbose:
            baseline_diff = result.metadata["baseline_logit_diff"]
            print(f"Baseline logit difference: {baseline_diff:.4f}")

        return result
