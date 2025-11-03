from dotenv import load_dotenv

load_dotenv()

import os

import torch
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from experiments.robustness.config import ExperimentConfig
from experiments.robustness.core import RobustnessExperimentRunner

torch.autograd.set_grad_enabled(False)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ACDC robustness experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments with single baseline
  python run_experiments.py --experiment all --run-ids run1 run2 run3 --baseline-ids run1

  # Run all experiments with multiple baselines
  python run_experiments.py --experiment all --run-ids run1 run2 run3 --baseline-ids baseline1 baseline2 baseline3

  # Run baseline experiment with single baseline
  python run_experiments.py --experiment jaccard-cross-similarity --run-ids run1 run2 run3 --baseline-ids run1

  # Run baseline experiment with multiple baselines
  python run_experiments.py --experiment jaccard-cross-similarity --run-ids run1 run2 run3 --baseline-ids baseline1 baseline2 baseline3

  # Run specific experiment (no baseline needed)
  python run_experiments.py --experiment logit-diff-calibration --run-ids run1 run2 run3
  
  # Run IO-S distribution experiment
  python run_experiments.py --experiment io-s-distribution --run-ids run1 run2 run3

  # Run with custom configuration
  python run_experiments.py --experiment all --run-ids run1 run2 run3 --baseline-ids run1 --device cpu --num-examples 50

  # Run with custom output directory
  python run_experiments.py --experiment jaccard-calibration --run-ids run1 run2 --output-dir ./my_results
        """,
    )

    # Required arguments
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "all",
            "logit-diff-calibration",
            "jaccard-calibration",
            "jaccard-cross-similarity",
            "io-s-distribution",
        ],
        help="Experiment to run",
    )

    parser.add_argument(
        "--run-ids", required=True, nargs="+", help="W&B run IDs to analyze"
    )

    # Optional arguments
    parser.add_argument(
        "--baseline-ids",
        nargs="+",
        type=str,
        help="Baseline run ID(s) - can be single ID or multiple IDs (required for baseline experiments)",
    )

    # Configuration arguments
    parser.add_argument(
        "--project-name",
        type=str,
        default="personal-14/acdc-robustness",
        help="W&B project name",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        help="Device to use for computation",
    )

    parser.add_argument(
        "--num-examples", type=int, default=100, help="Number of examples to use"
    )

    parser.add_argument(
        "--metric-name", type=str, default="logit_diff", help="Metric name to use"
    )

    parser.add_argument(
        "--perturbation",
        type=str,
        choices=["shuffle_abc_prompts", "add_random_prefixes", "swap_dataset_roles", "None"],
        default="None",
        help="Perturbation type to use (or None for no perturbation)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/robustness/results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Enable verbose output"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output (overrides --verbose)",
    )

    return parser.parse_args()


def create_config(args: argparse.Namespace) -> ExperimentConfig:
    """Create experiment configuration from command line arguments."""
    # Convert "None" string to None for perturbation
    perturbation = None if args.perturbation == "None" else args.perturbation
    
    return ExperimentConfig(
        project_name=args.project_name,
        device=args.device,
        num_examples=args.num_examples,
        metric_name=args.metric_name,
        perturbation=perturbation,
        output_dir=Path(args.output_dir),
        verbose=args.verbose and not args.quiet,
    )


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check if baseline is required
    if (
        args.experiment in ["jaccard-cross-similarity"]
        and not args.baseline_ids
    ):
        print(f"Error: Experiment '{args.experiment}' requires --baseline-ids")
        sys.exit(1)

    # Check if baseline is provided when not needed
    if args.experiment in ["logit-diff-calibration", "jaccard-calibration", "io-s-distribution"] and args.baseline_ids:
        print(
            f"Warning: Experiment '{args.experiment}' does not use baseline, ignoring --baseline-ids"
        )

    # Validate run IDs
    if not args.run_ids:
        print("Error: At least one run ID must be provided")
        sys.exit(1)

    # Check for duplicate run IDs
    if len(args.run_ids) != len(set(args.run_ids)):
        print("Error: Duplicate run IDs found")
        sys.exit(1)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Validate arguments
    validate_arguments(args)

    # Create configuration
    config = create_config(args)

    # Print configuration
    if config.verbose:
        print("=" * 80)
        print("ACDC ROBUSTNESS EXPERIMENTS")
        print("=" * 80)
        print(f"Experiment: {args.experiment}")
        print(f"Run IDs: {', '.join(args.run_ids)}")
        if args.baseline_ids:
            if len(args.baseline_ids) == 1:
                print(f"Baseline: {args.baseline_ids[0]}")
            else:
                print(f"Baselines: {', '.join(args.baseline_ids)}")
        print(f"Project: {config.project_name}")
        print(f"Device: {config.device}")
        print(f"Perturbation: {config.perturbation or 'None'}")
        print(f"Output: {config.output_dir}")
        print("=" * 80)

    # Create experiment runner
    runner = RobustnessExperimentRunner(config)

    # Run experiments
    if args.experiment == "all":
        # For 'all' experiments, use all baseline IDs if provided
        runner.run_all_experiments(args.run_ids, args.baseline_ids)
    elif args.experiment == "logit-diff-calibration":
        runner.run_logit_difference_analysis(args.run_ids)
    elif args.experiment == "jaccard-calibration":
        runner.run_pairwise_jaccard_similarity(args.run_ids)
    elif args.experiment == "jaccard-cross-similarity":
        runner.run_jaccard_cross_similarity(args.baseline_ids, args.run_ids)
    elif args.experiment == "io-s-distribution":
        runner.run_io_s_distribution(args.run_ids)

    print(f"\nExperiments completed successfully!")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
