from typing import List, Optional, Dict, Any
from pathlib import Path

import sys

from experiments.robustness.config import ExperimentConfig, ExperimentResult, CircuitBatch
from experiments.robustness.core import CircuitLoader, MetricComputer, ResultsManager
from experiments.robustness.experiments import (
    LogitDiffAnalysis,
    PairwiseJaccardAnalysis,
    BaselineJaccardAnalysis,
    BaselineLogitDiffAnalysis,
)


class RobustnessExperimentRunner:
    """
    Main orchestrator for all robustness experiments.

    This class provides a unified interface for running individual experiments
    or all experiments in sequence. It manages the lifecycle of experiments,
    handles resource allocation, and coordinates result collection.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the robustness experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.loader = CircuitLoader(config)
        self.metrics = MetricComputer(config)
        self.results_manager = ResultsManager(config.output_dir)

        # Initialize experiment classes
        self.logit_diff_analysis = LogitDiffAnalysis(config)
        self.pairwise_jaccard = PairwiseJaccardAnalysis(config)
        self.baseline_jaccard = BaselineJaccardAnalysis(config)
        self.baseline_logit_diff = BaselineLogitDiffAnalysis(config)

        # Circuit caching for reuse across experiments
        self._cached_circuit_batch: Optional[CircuitBatch] = None

        if self.config.verbose:
            print(f"Initialized RobustnessExperimentRunner with config:")
            print(f"  Project: {self.config.project_name}")
            print(f"  Device: {self.config.device}")
            print(f"  Output dir: {self.config.output_dir}")

    def _load_circuits_once(self, run_ids: List[str]) -> CircuitBatch:
        """
        Load circuits once and cache them for reuse across all experiments.
        
        Args:
            run_ids: List of run IDs to load
            
        Returns:
            CircuitBatch object containing the loaded circuits
        """
        if self._cached_circuit_batch is None:
            if self.config.verbose:
                print(f"Loading circuits for {len(run_ids)} runs...")
            
            self._cached_circuit_batch = self.loader.load_circuits_batch(run_ids)
            
            if self.config.verbose:
                print(f"Circuits loaded and cached for reuse across all experiments.")
        else:
            if self.config.verbose:
                print(f"Reusing cached circuits for {len(run_ids)} runs.")
        
        return self._cached_circuit_batch

    def clear_circuit_cache(self):
        """Clear the cached circuit batch to free memory."""
        self._cached_circuit_batch = None
        if self.config.verbose:
            print("Circuit cache cleared.")

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

    def run_logit_difference_analysis(self, run_ids: List[str]) -> ExperimentResult:
        """Run Logit Difference Analysis experiment."""
        if self.config.verbose:
            print(f"Running Logit Difference Analysis...")

        # Load circuits once and reuse
        circuit_batch = self._load_circuits_once(run_ids)
        
        # Filter to only include perturbed circuits
        filtered_circuit_batch = self._filter_circuit_batch(circuit_batch, run_ids)
        
        # Use the individual experiment class with filtered circuits
        result = self.logit_diff_analysis.run(run_ids, filtered_circuit_batch)
        
        self.results_manager.save_results(result)
        return result

    def run_pairwise_jaccard_similarity(self, run_ids: List[str]) -> ExperimentResult:
        """Run Pairwise Jaccard Similarity experiment."""
        if self.config.verbose:
            print(f"Running Pairwise Jaccard Similarity...")

        # Load circuits once and reuse
        circuit_batch = self._load_circuits_once(run_ids)
        
        # Filter to only include perturbed circuits
        filtered_circuit_batch = self._filter_circuit_batch(circuit_batch, run_ids)
        
        # Use the individual experiment class with filtered circuits
        result = self.pairwise_jaccard.run(run_ids, filtered_circuit_batch)
        
        self.results_manager.save_results(result)
        return result

    def run_baseline_jaccard_similarity(
        self, baseline_run_id: str, run_ids: List[str]
    ) -> ExperimentResult:
        """Run Baseline Jaccard Similarity experiment."""
        if self.config.verbose:
            print(f"Running Baseline Jaccard Similarity...")

        # Load circuits for all runs (baseline + run_ids)
        all_run_ids = [baseline_run_id] + run_ids
        circuit_batch = self._load_circuits_once(all_run_ids)
        
        # Filter to only include circuits for the requested run_ids (including baseline)
        filtered_circuit_batch = self._filter_circuit_batch(circuit_batch, all_run_ids)
        
        # Use the individual experiment class with filtered circuits
        result = self.baseline_jaccard.run(baseline_run_id, run_ids, filtered_circuit_batch)
        
        self.results_manager.save_results(result)
        return result

    def run_baseline_logit_diff_robustness(
        self, baseline_run_id: str, run_ids: List[str]
    ) -> ExperimentResult:
        """Run Baseline Logit Difference Robustness experiment."""
        if self.config.verbose:
            print(f"Running Baseline Logit Difference Robustness...")

        # Load circuits for all runs (baseline + run_ids)
        all_run_ids = [baseline_run_id] + run_ids
        circuit_batch = self._load_circuits_once(all_run_ids)
        
        # Filter to only include circuits for the requested run_ids (including baseline)
        filtered_circuit_batch = self._filter_circuit_batch(circuit_batch, all_run_ids)
        
        # Use the individual experiment class with filtered circuits
        result = self.baseline_logit_diff.run(baseline_run_id, run_ids, filtered_circuit_batch)
        
        self.results_manager.save_results(result)
        return result

    def run_all_experiments(
        self, run_ids: List[str], baseline_run_id: Optional[str] = None
    ) -> List[ExperimentResult]:
        """
        Run all robustness experiments in sequence.

        Args:
            run_ids: List of W&B run IDs to analyze
            baseline_run_id: Optional baseline run ID for experiments 3 & 4

        Returns:
            List of ExperimentResult objects from all experiments
        """
        if self.config.verbose:
            print(f"Running all experiments with {len(run_ids)} runs...")

        # Load all circuits once (including baseline if provided)
        all_run_ids = run_ids.copy()
        if baseline_run_id:
            all_run_ids = [baseline_run_id] + run_ids
        
        if self.config.verbose:
            print(f"Loading circuits for {len(all_run_ids)} runs (including baseline)...")
        
        circuit_batch = self._load_circuits_once(all_run_ids)

        results = []

        # Run experiments 1 & 2 (no baseline needed)
        results.append(self.run_logit_difference_analysis(run_ids))
        results.append(self.run_pairwise_jaccard_similarity(run_ids))

        # Run experiments 3 & 4 (require baseline)
        if baseline_run_id:
            results.append(
                self.run_baseline_jaccard_similarity(baseline_run_id, run_ids)
            )
            results.append(
                self.run_baseline_logit_diff_robustness(baseline_run_id, run_ids)
            )

        if self.config.verbose:
            print(f"Completed {len(results)} experiments")

        return results
