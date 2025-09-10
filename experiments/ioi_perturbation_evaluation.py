#!/usr/bin/env python3
"""
IOI Perturbation Evaluation Script

This script loads two ACDC runs from Weights & Biases and computes Jaccard indices
for both edges and nodes between the two graphs to evaluate their similarity.

Usage:
    python ioi_perturbation_evaluation.py --run1-id <run1_id> --run2-id <run2_id>
"""

import gc
from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
from pathlib import Path
from typing import Set, Tuple, Dict, Any, Optional
import wandb
import torch

from utils.utils import get_acdc_runs
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCEdge import TorchIndex, Edge
from acdc.acdc_utils import get_present_nodes, filter_nodes
from acdc.ioi.utils import get_all_ioi_things
from acdc.TLACDCExperiment import TLACDCExperiment

torch.autograd.set_grad_enabled(False)


def load_single_acdc_run(
    run_id: str,
    project_name: str = "personal-14/acdc-robustness",
    device: str = "cuda"
) -> TLACDCCorrespondence:
    """
    Load a single ACDC run from Weights & Biases by run ID.
    
    Args:
        run_id: The Weights & Biases run ID
        project_name: The W&B project name
        device: Device to run computations on
        
    Returns:
        TLACDCCorrespondence object containing the graph from the run
    """
    # Setup IOI things for the experiment
    num_examples = 100  # Use fewer examples for faster loading
    things = get_all_ioi_things(num_examples=num_examples, device=device, metric_name="kl_div")
    
    tl_model = things.tl_model
    tl_model.reset_hooks()

    # Save some mem
    gc.collect()
    torch.cuda.empty_cache()

    # Create the experiment object (similar to roc_plot_generator.py)
    tl_model.reset_hooks()

    exp = TLACDCExperiment(
        model=tl_model,
        threshold=100_000,
        early_exit=False,
        using_wandb=False,
        zero_ablation=False,
        ds=things.test_data,
        ref_ds=things.test_patch_data,
        metric=things.validation_metric,
        second_metric=None,
        verbose=True,
        use_pos_embed=False,
        online_cache_cpu=False,
        corrupted_cache_cpu=False,
    )
    
    # Create a filter to get the specific run
    pre_run_filter = {
        "name": run_id
    }
    
    # Load the run using get_acdc_runs
    corrs, ids = get_acdc_runs(
        exp=exp,
        things=things,
        project_name=project_name,
        pre_run_filter=pre_run_filter,
        run_filter=None,
        clip=1,  # Only get one run
        return_ids=True,
        ignore_missing_score=True
    )
    
    if not corrs:
        raise ValueError(f"No ACDC run found with ID: {run_id}")
    
    correspondence, score_dict = corrs[0]
    print(f"Loaded run {run_id} with {correspondence.count_no_edges()} edges")
    
    return correspondence


def get_present_edges_from_correspondence(correspondence: TLACDCCorrespondence) -> Set[Tuple[str, TorchIndex, str, TorchIndex]]:
    """
    Extract all present edges from a correspondence object.
    
    Args:
        correspondence: TLACDCCorrespondence object
        
    Returns:
        Set of tuples representing present edges (child_name, child_index, parent_name, parent_index)
    """
    present_edges = set()
    
    for edge_tuple, edge in correspondence.all_edges().items():
        if edge.present:
            present_edges.add(edge_tuple)
    
    return present_edges


def get_present_nodes_from_correspondence(correspondence: TLACDCCorrespondence) -> Set[Tuple[str, TorchIndex]]:
    """
    Extract all present nodes from a correspondence object.
    
    Args:
        correspondence: TLACDCCorrespondence object
        
    Returns:
        Set of tuples representing present nodes (name, index)
    """
    present_nodes, _ = get_present_nodes(correspondence)
    return present_nodes


def compute_jaccard_index_edges(
    corr1: TLACDCCorrespondence, 
    corr2: TLACDCCorrespondence
) -> float:
    """
    Compute the Jaccard index for edges between two correspondence objects.
    
    The Jaccard index is defined as |A * B| / |A + B| where A and B are sets of edges.
    
    Args:
        corr1: First correspondence object
        corr2: Second correspondence object
        
    Returns:
        Jaccard index for edges (float between 0 and 1)
    """
    edges1 = get_present_edges_from_correspondence(corr1)
    edges2 = get_present_edges_from_correspondence(corr2)
    
    intersection = edges1.intersection(edges2)
    union = edges1.union(edges2)
    
    if len(union) == 0:
        return 1.0  # Both graphs have no edges
    
    jaccard_index = len(intersection) / len(union)
    return jaccard_index


def compute_jaccard_index_nodes(
    corr1: TLACDCCorrespondence, 
    corr2: TLACDCCorrespondence
) -> float:
    """
    Compute the Jaccard index for nodes between two correspondence objects.
    
    The Jaccard index is defined as |A * B| / |A + B| where A and B are sets of nodes.
    
    Args:
        corr1: First correspondence object
        corr2: Second correspondence object
        
    Returns:
        Jaccard index for nodes (float between 0 and 1)
    """
    nodes1 = get_present_nodes_from_correspondence(corr1)
    nodes2 = get_present_nodes_from_correspondence(corr2)
    
    intersection = nodes1.intersection(nodes2)
    union = nodes1.union(nodes2)
    
    if len(union) == 0:
        return 1.0  # Both graphs have no nodes
    
    jaccard_index = len(intersection) / len(union)
    return jaccard_index


def compute_logit_diff_relative_change(
    corr1: TLACDCCorrespondence,
    corr2: TLACDCCorrespondence,
    exp: TLACDCExperiment,
    things: Any,
    device: str = "cuda"
) -> float:
    """
    Compute the relative change in logit difference between two circuit graphs.
    
    The relative change is defined as:
    |logit_diff_2 - logit_diff_1| / |logit_diff_1|
    
    Args:
        corr1: First correspondence object (baseline circuit)
        corr2: Second correspondence object (comparison circuit)
        exp: TLACDCExperiment object with proper setup
        things: Object containing test_metrics and test_data
        device: Device to run computations on
        
    Returns:
        Relative change in logit difference (float >= 0)
    """
    # Ensure corrupted cache is set up before computing metrics
    exp.setup_corrupted_cache()
    
    # Define the logit difference metric function
    def get_logit_diff_metric(data: torch.Tensor) -> dict[str, float]:
        """Get logit difference metric using the test_metrics from things"""
        return {f"test_{name}": fn(data).item() for name, fn in things.test_metrics.items()}
    
    # Compute logit difference for first circuit
    print("Computing logit difference for circuit 1...")
    metrics1 = exp.call_metric_with_corr(corr1, get_logit_diff_metric, things.test_data)
    logit_diff_1 = metrics1["test_logit_diff"]
    
    # Compute logit difference for second circuit  
    print("Computing logit difference for circuit 2...")
    metrics2 = exp.call_metric_with_corr(corr2, get_logit_diff_metric, things.test_data)
    logit_diff_2 = metrics2["test_logit_diff"]
    
    # Compute relative change
    absolute_diff = abs(logit_diff_2 - logit_diff_1)
    absolute_baseline = abs(logit_diff_1)
    
    if absolute_baseline == 0:
        # If baseline is zero, return the absolute difference
        relative_change = absolute_diff
        print(f"Warning: Baseline logit difference is 0, returning absolute difference: {absolute_diff}")
    else:
        relative_change = absolute_diff / absolute_baseline
    
    print(f"Logit diff 1: {logit_diff_1:.6f}")
    print(f"Logit diff 2: {logit_diff_2:.6f}")
    print(f"Absolute difference: {absolute_diff:.6f}")
    print(f"Relative change: {relative_change:.6f}")
    
    return relative_change

def main():
    """Main function to run the IOI perturbation evaluation."""
    parser = argparse.ArgumentParser(
        description="Compare two ACDC runs using Jaccard indices for edges and nodes"
    )
    parser.add_argument(
        "--run1-id", 
        type=str, 
        required=True,
        help="Weights & Biases run ID for the first graph"
    )
    parser.add_argument(
        "--run2-id", 
        type=str, 
        required=True,
        help="Weights & Biases run ID for the second graph"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="personal-14/acdc-robustness",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run computations on"
    )
    parser.add_argument(
        "--compute-logit-diff",
        action="store_true",
        help="Compute logit difference relative change between circuits"
    )
    
    args = parser.parse_args()
    
    print("IOI Perturbation Evaluation")
    print("=" * 50)
    print(f"Run 1 ID: {args.run1_id}")
    print(f"Run 2 ID: {args.run2_id}")
    print(f"Project: {args.project_name}")
    print(f"Device: {args.device}")
    
    # Load the two ACDC runs
    print("\nLoading ACDC runs...")
    corr1 = load_single_acdc_run(args.run1_id, args.project_name, args.device)
    corr2 = load_single_acdc_run(args.run2_id, args.project_name, args.device)
    
    # Compute Jaccard indices
    print("\nComputing Jaccard indices...")
    edge_jaccard = compute_jaccard_index_edges(corr1, corr2)
    node_jaccard = compute_jaccard_index_nodes(corr1, corr2)
    
    # Print results
    print("\nResults:")
    print("=" * 30)
    print(f"Edge Jaccard Index: {edge_jaccard:.4f}")
    print(f"Node Jaccard Index: {node_jaccard:.4f}")
    
    # Compute logit difference relative change if requested
    if args.compute_logit_diff:
        print("\nSetting up experiment for logit difference computation...")
        num_examples = 100
        things = get_all_ioi_things(num_examples=num_examples, device=args.device, metric_name="logit_diff")
        
        exp = TLACDCExperiment(
            model=things.tl_model,
            threshold=100_000,
            early_exit=False,
            using_wandb=False,
            zero_ablation=False,
            ds=things.test_data,
            ref_ds=things.test_patch_data,
            metric=things.validation_metric,
            second_metric=None,
            verbose=True,
            use_pos_embed=False,
            online_cache_cpu=False,
            corrupted_cache_cpu=False,
        )
        
        print("\nComputing logit difference relative change...")
        logit_diff_relative_change = compute_logit_diff_relative_change(
            corr1, corr2, exp, things, args.device
        )
        
        print(f"Logit Difference Relative Change: {logit_diff_relative_change:.6f}")


if __name__ == "__main__":
    main()

