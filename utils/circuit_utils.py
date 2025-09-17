import gc
from typing import Set, Tuple, Dict, Any
from dotenv import load_dotenv

load_dotenv()

import torch
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import TorchIndex
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import get_present_nodes
from acdc.ioi.utils import get_all_ioi_things
from .utils import get_acdc_runs


def get_present_edges_from_correspondence(
    correspondence: TLACDCCorrespondence,
) -> Set[Tuple[str, TorchIndex, str, TorchIndex]]:
    """
    Extract all present edges from a correspondence object.

    Args:
        correspondence: TLACDCCorrespondence object

    Returns:
        Set of tuples representing present edges (child_name, child_index, parent_name, parent_index)
    """
    present_edges: Set[Tuple[str, TorchIndex, str, TorchIndex]] = set()

    for edge_tuple, edge in correspondence.all_edges().items():
        if edge.present:
            present_edges.add(edge_tuple)

    return present_edges


def get_present_nodes_from_correspondence(
    correspondence: TLACDCCorrespondence,
) -> Set[Tuple[str, TorchIndex]]:
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
    corr1: TLACDCCorrespondence, corr2: TLACDCCorrespondence
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
    edges1 = get_present_edges_from_correspondence(corr1)
    edges2 = get_present_edges_from_correspondence(corr2)

    intersection = edges1.intersection(edges2)
    union = edges1.union(edges2)

    if len(union) == 0:
        return 1.0  # Both graphs have no edges

    jaccard_index = len(intersection) / len(union)
    return jaccard_index


def compute_jaccard_index_nodes(
    corr1: TLACDCCorrespondence, corr2: TLACDCCorrespondence
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


def load_single_acdc_run(
    run_id: str,
    exp: TLACDCExperiment,
    things: Any,
    project_name: str = "personal-14/acdc-robustness",
) -> TLACDCCorrespondence:
    """
    Load a single ACDC run from Weights & Biases by run ID.

    Args:
        run_id: The Weights & Biases run ID
        exp: TLACDCExperiment object (pre-configured)
        things: AllDataThings object (pre-configured)
        project_name: The W&B project name

    Returns:
        TLACDCCorrespondence object containing the graph from the run
    """
    # Create a filter to get the specific run
    pre_run_filter = {"name": run_id}

    # Load the run using get_acdc_runs
    corrs, ids = get_acdc_runs(
        exp=exp,
        things=things,
        project_name=project_name,
        pre_run_filter=pre_run_filter,
        run_filter=None,
        clip=1,  # Only get one run
        return_ids=True,
        ignore_missing_score=True,
    )

    if not corrs:
        raise ValueError(f"No ACDC run found with ID: {run_id}")

    correspondence, score_dict = corrs[0]
    print(f"Loaded run {run_id} with {correspondence.count_no_edges()} edges")

    return correspondence
