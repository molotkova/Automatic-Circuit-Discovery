import os
import re
import json
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import wandb
import requests
from tqdm import tqdm

from acdc.TLACDCInterpNode import parse_interpnode
from acdc.TLACDCCorrespondence import TLACDCCorrespondence


@dataclass(frozen=True)
class AcdcRunCandidate:
    threshold: float
    steps: int
    run: wandb.apis.public.Run
    score_d: dict
    corr: TLACDCCorrespondence


def get_acdc_runs(
    exp,
    things,
    project_name: str,
    pre_run_filter: dict,
    run_filter: Optional[Callable[[Any], bool]] = None,
    clip: Optional[int] = None,
    return_ids: bool = False,
    ignore_missing_score: bool = False,
    root_cache_dir: Optional[Path] = None,
):
    """
    Load ACDC runs from Weights & Biases and return correspondence objects with scores.

    Args:
        exp: TLACDCExperiment object
        things: Object containing test_metrics and test_data
        project_name: W&B project name
        pre_run_filter: Filter for W&B runs
        run_filter: Optional function to filter runs further
        clip: Maximum number of runs to process
        return_ids: Whether to return run IDs along with correspondences
        ignore_missing_score: Whether to ignore runs missing score information
        root_cache_dir: Directory for caching artifacts (defaults to ~/.cache/artifacts_for_plot)

    Returns:
        List of (correspondence, score_dict) tuples, optionally with run IDs
    """
    if clip is None:
        clip = 100_000  # so we don't clip anything

    if root_cache_dir is None:
        root_cache_dir = Path(os.environ["HOME"]) / ".cache" / "artifacts_for_plot"

    root_cache_dir.mkdir(exist_ok=True)

    api = wandb.Api()
    runs = api.runs(project_name, filters=pre_run_filter)
    if run_filter is None:
        filtered_runs = list(runs)[:clip]
    else:
        filtered_runs = list(filter(run_filter, tqdm(list(runs)[:clip])))
    print(
        f"loading {len(filtered_runs)} runs with filter {pre_run_filter} and {run_filter}"
    )

    threshold_to_run_map: dict[float, AcdcRunCandidate] = {}

    def add_run_for_processing(candidate: AcdcRunCandidate):
        if candidate.threshold not in threshold_to_run_map:
            threshold_to_run_map[candidate.threshold] = candidate
        else:
            if candidate.steps > threshold_to_run_map[candidate.threshold].steps:
                threshold_to_run_map[candidate.threshold] = candidate

    for run in filtered_runs:
        score_d = {k: v for k, v in run.summary.items() if k.startswith("test")}
        try:
            score_d["steps"] = run.summary["_step"]
        except KeyError:
            continue  # Run has crashed too much

        try:
            score_d["score"] = run.config["threshold"]
        except KeyError:
            try:
                score_d["score"] = float(run.name)
            except ValueError:
                try:
                    score_d["score"] = float(run.name.split("_")[-1])
                except ValueError as e:
                    if ignore_missing_score:
                        continue
                    else:
                        raise e

        threshold = score_d["score"]

        if "num_edges" in run.summary:
            print("This run n edges:", run.summary["num_edges"])
        # Try to find `edges.pth`
        edges_artifact = None
        for art in run.logged_artifacts():
            if "edges.pth" in art.name:
                edges_artifact = art
                break

        if edges_artifact is None:
            # We'll have to parse the run
            print(f"Edges.pth not found for run {run.name}, falling back to plotly")
            corr = deepcopy(exp.corr)

            # Find latest plotly file which contains the `result` for all edges
            files = run.files(per_page=100_000)
            regexp = re.compile(r"^media/plotly/results_([0-9]+)_[^.]+\.plotly\.json$")
            assert len(files) > 0

            latest_file = None
            latest_fname_step = -1
            for f in files:
                if m := regexp.match(f.name):
                    fname_step = int(m.group(1))
                    if fname_step > latest_fname_step:
                        latest_fname_step = fname_step
                        latest_file = f

            try:
                if latest_file is None:
                    raise wandb.CommError("a")
                # replace=False because these files are never modified. Save them in a unique location, ROOT/run.id
                with latest_file.download(
                    root_cache_dir / run.id, replace=False, exist_ok=True
                ) as f:
                    d = json.load(f)

                data = d["data"][0]
                assert len(data["text"]) == len(data["y"])

                # Mimic an ACDC run
                for edge, result in zip(data["text"], data["y"]):
                    parent, child = map(parse_interpnode, edge.split(" to "))
                    current_node = child

                    if result < threshold:
                        corr.edges[child.name][child.index][parent.name][
                            parent.index
                        ].present = False
                        corr.remove_edge(
                            current_node.name,
                            current_node.index,
                            parent.name,
                            parent.index,
                        )
                    else:
                        corr.edges[child.name][child.index][parent.name][
                            parent.index
                        ].present = True
                print("Before copying: n_edges=", corr.count_no_edges())

                corr_all_edges = corr.all_edges().items()

                corr_to_copy = deepcopy(exp.corr)
                new_all_edges = corr_to_copy.all_edges()
                for edge in new_all_edges.values():
                    edge.present = False

                for tupl, edge in corr_all_edges:
                    new_all_edges[tupl].present = edge.present

                print("After copying: n_edges=", corr_to_copy.count_no_edges())

                # Correct score_d to reflect the actual number of steps that we are collecting
                score_d["steps"] = latest_fname_step
                add_run_for_processing(
                    AcdcRunCandidate(
                        threshold=threshold,
                        steps=score_d["steps"],
                        run=run,
                        score_d=score_d,
                        corr=corr_to_copy,
                    )
                )

            except (wandb.CommError, requests.exceptions.HTTPError) as e:
                print(f"Error {e}, falling back to parsing output.log")
                try:
                    with run.file("output.log").download(
                        root=root_cache_dir / run.id, replace=False, exist_ok=True
                    ) as f:
                        log_text = f.read()
                    exp.load_from_wandb_run(log_text)
                    add_run_for_processing(
                        AcdcRunCandidate(
                            threshold=threshold,
                            steps=score_d["steps"],
                            run=run,
                            score_d=score_d,
                            corr=deepcopy(exp.corr),
                        )
                    )
                except Exception:
                    print(
                        f"Loading run {run.name} with state={run.state} config={run.config} totally failed."
                    )
                    continue

        else:
            corr = deepcopy(exp.corr)
            all_edges = corr.all_edges()
            for edge in all_edges.values():
                edge.present = False

            this_root = root_cache_dir / edges_artifact.name
            # Load the edges
            for f in edges_artifact.files():
                with f.download(root=this_root, replace=True, exist_ok=True) as fopen:
                    # Sadly f.download opens in text mode
                    with open(fopen.name, "rb") as fopenb:
                        edges_pth = pickle.load(fopenb)

            for (n_to, idx_to, n_from, idx_from), _effect_size in edges_pth:
                n_to = n_to.replace("hook_resid_mid", "hook_mlp_in")
                n_from = n_from.replace("hook_resid_mid", "hook_mlp_in")
                all_edges[(n_to, idx_to, n_from, idx_from)].present = True

            add_run_for_processing(
                AcdcRunCandidate(
                    threshold=threshold,
                    steps=score_d["steps"],
                    run=run,
                    score_d=score_d,
                    corr=corr,
                )
            )

    # Now add the test_fns to the score_d of the remaining runs
    def all_test_fns(data: torch.Tensor) -> dict[str, float]:
        return {
            f"test_{name}": fn(data).item() for name, fn in things.test_metrics.items()
        }

    all_candidates = list(threshold_to_run_map.values())
    for candidate in all_candidates:
        test_metrics = exp.call_metric_with_corr(
            candidate.corr, all_test_fns, things.test_data
        )
        candidate.score_d.update(test_metrics)
        print(
            f"Added run with threshold={candidate.threshold}, n_edges={candidate.corr.count_no_edges()}"
        )

    corrs = [(candidate.corr, candidate.score_d) for candidate in all_candidates]
    if return_ids:
        return corrs, [candidate.run.id for candidate in all_candidates]
    return corrs
