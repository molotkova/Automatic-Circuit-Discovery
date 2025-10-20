from typing import List, Optional, Dict, Any
from pathlib import Path

import sys

from experiments.robustness.config import ExperimentConfig, ExperimentResult, CircuitBatch
from experiments.robustness.core import CircuitLoader, MetricComputer, ResultsManager
from experiments.robustness.experiments import (
    LogitDiffAnalysis,
    PairwiseJaccardAnalysis,
    BaselineJaccardAnalysis,
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


    def run_logit_difference_analysis(self, run_ids: List[str]) -> ExperimentResult:
        """Run Logit Difference Analysis experiment."""
        if self.config.verbose:
            print(f"Running Logit Difference Analysis...")

        # Load circuits once and reuse
        circuit_batch = self._load_circuits_once(run_ids)
        
        # Use the individual experiment class
        result = self.logit_diff_analysis.run(run_ids, circuit_batch)
        
        self.results_manager.save_results(result)
        return result

    def run_pairwise_jaccard_similarity(self, run_ids: List[str]) -> ExperimentResult:
        """Run Pairwise Jaccard Similarity experiment."""
        if self.config.verbose:
            print(f"Running Pairwise Jaccard Similarity...")

        # Load circuits once and reuse
        circuit_batch = self._load_circuits_once(run_ids)
        
        # Use the individual experiment class
        result = self.pairwise_jaccard.run(run_ids, circuit_batch)
        
        self.results_manager.save_results(result)
        return result

    def run_baseline_jaccard_similarity(
        self, baseline_run_ids: List[str], run_ids: List[str]
    ) -> ExperimentResult:
        """Run Baseline Jaccard Similarity experiment."""
        if self.config.verbose:
            print(f"Running Baseline Jaccard Similarity...")

        # Load circuits for all runs (baselines + run_ids)
        all_run_ids = baseline_run_ids + run_ids
        circuit_batch = self._load_circuits_once(all_run_ids)
        
        # Use the individual experiment class
        result = self.baseline_jaccard.run(baseline_run_ids, run_ids, circuit_batch)
        
        self.results_manager.save_results(result)
        return result


    def run_all_experiments(
        self, run_ids: List[str], baseline_run_ids: Optional[List[str]] = None
    ) -> List[ExperimentResult]:
        """
        Run all robustness experiments in sequence.

        Args:
            run_ids: List of W&B run IDs to analyze
            baseline_run_ids: List of baseline run IDs for experiment 3

        Returns:
            List of ExperimentResult objects from all experiments
        """
        if self.config.verbose:
            print(f"Running all experiments with {len(run_ids)} runs...")

        # Load all circuits once (including baselines if provided)
        all_run_ids = run_ids.copy()
        if baseline_run_ids:
            all_run_ids = baseline_run_ids + run_ids
        
        if self.config.verbose:
            print(f"Loading circuits for {len(all_run_ids)} runs (including baselines)...")
        
        circuit_batch = self._load_circuits_once(all_run_ids)

        results = []

        # Run experiments 1 & 2 (no baseline needed)
        results.append(self.run_logit_difference_analysis(run_ids))
        results.append(self.run_pairwise_jaccard_similarity(run_ids))

        # Run experiment 3 (requires baseline)
        if baseline_run_ids:
            results.append(
                self.run_baseline_jaccard_similarity(baseline_run_ids, run_ids)
            )

        if self.config.verbose:
            print(f"Completed {len(results)} experiments")

        return results
