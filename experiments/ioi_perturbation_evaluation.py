#!/usr/bin/env python3
"""
IOI Perturbation Evaluation Script

This script loads two ACDC runs from Weights & Biases and computes Jaccard indices
for both edges and nodes between the two graphs to evaluate their similarity.

Usage:
    python ioi_perturbation_evaluation.py --run1-id <run1_id> --run2-id <run2_id>
"""

import argparse
import sys
from pathlib import Path
from typing import Set, Tuple, Dict, Any, Optional
import wandb

# Add the project root to the path
# project_root = Path(__file__).resolve().parent.parent
# sys.path.append(str(project_root))

from utils.utils import get_acdc_runs
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCEdge import TorchIndex, Edge
from acdc.acdc_utils import get_present_nodes, filter_nodes
from acdc.ioi.utils import get_all_ioi_things


def load_single_acdc_run(
    run_id: str,
    project_name: str = "remix_school-of-rock/acdc",
    device: str = "gpu"
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
    num_examples = 10  # Use fewer examples for faster loading
    things = get_all_ioi_things(num_examples=num_examples, device=device, metric_name="kl_div")
    
    # Create a filter to get the specific run
    pre_run_filter = {
        "display_name": run_id,
        "state": "finished"
    }
    
    # Load the run using get_acdc_runs
    corrs, ids = get_acdc_runs(
        exp=None,  # We'll create the experiment later if needed
        things=None,
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


def get_present_edges(correspondence: TLACDCCorrespondence) -> Set[Tuple[str, TorchIndex, str, TorchIndex]]:
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
    
    The Jaccard index is defined as |A ∩ B| / |A ∪ B| where A and B are sets of edges.
    
    Args:
        corr1: First correspondence object
        corr2: Second correspondence object
        
    Returns:
        Jaccard index for edges (float between 0 and 1)
    """
    edges1 = get_present_edges(corr1)
    edges2 = get_present_edges(corr2)
    
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
    
    The Jaccard index is defined as |A ∩ B| / |A ∪ B| where A and B are sets of nodes.
    
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


def print_graph_statistics(correspondence: TLACDCCorrespondence, name: str):
    """
    Print statistics about a correspondence graph.
    
    Args:
        correspondence: TLACDCCorrespondence object
        name: Name to display for this graph
    """
    present_edges = get_present_edges(correspondence)
    present_nodes = get_present_nodes_from_correspondence(correspondence)
    
    print(f"\n{name} Graph Statistics:")
    print(f"  Present edges: {len(present_edges)}")
    print(f"  Present nodes: {len(present_nodes)}")
    print(f"  Total edges in graph: {len(correspondence.all_edges())}")


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
        default="remix_school-of-rock/acdc",
        help="Weights & Biases project name (default: remix_school-of-rock/acdc)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run computations on (default: cpu)"
    )
    
    args = parser.parse_args()
    
    print("IOI Perturbation Evaluation")
    print("=" * 50)
    print(f"Run 1 ID: {args.run1_id}")
    print(f"Run 2 ID: {args.run2_id}")
    print(f"Project: {args.project_name}")
    print(f"Device: {args.device}")
    
    try:
        # Load the two ACDC runs
        print("\nLoading ACDC runs...")
        corr1 = load_single_acdc_run(args.run1_id, args.project_name, args.device)
        corr2 = load_single_acdc_run(args.run2_id, args.project_name, args.device)
        
        # Print statistics for both graphs
        print_graph_statistics(corr1, "First")
        print_graph_statistics(corr2, "Second")
        
        # Compute Jaccard indices
        print("\nComputing Jaccard indices...")
        edge_jaccard = compute_jaccard_index_edges(corr1, corr2)
        node_jaccard = compute_jaccard_index_nodes(corr1, corr2)
        
        # Print results
        print("\nResults:")
        print("=" * 30)
        print(f"Edge Jaccard Index: {edge_jaccard:.4f}")
        print(f"Node Jaccard Index: {node_jaccard:.4f}")
        
        # Interpretation
        print("\nInterpretation:")
        print(f"Edge similarity: {edge_jaccard:.1%}")
        print(f"Node similarity: {node_jaccard:.1%}")
        
        if edge_jaccard > 0.8:
            print("High edge similarity - graphs are very similar in structure")
        elif edge_jaccard > 0.5:
            print("Moderate edge similarity - graphs share significant structure")
        else:
            print("Low edge similarity - graphs are quite different")
            
        if node_jaccard > 0.8:
            print("High node similarity - graphs use very similar nodes")
        elif node_jaccard > 0.5:
            print("Moderate node similarity - graphs share many nodes")
        else:
            print("Low node similarity - graphs use quite different nodes")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

