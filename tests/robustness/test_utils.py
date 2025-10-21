import time
from typing import Any, Callable, Dict, List, Mapping, Set

import torch

from experiments.robustness.config import ExperimentConfig, CircuitBatch


class FakeExperiment:
    def __init__(self, run_id_to_logit_diff: Mapping[str, float]) -> None:
        self.run_id_to_logit_diff: Mapping[str, float] = run_id_to_logit_diff
        self._setup_called: bool = False

    def setup_corrupted_cache(self) -> None:
        self._setup_called = True

    def call_metric_with_corr(
        self,
        circuit: Any,
        metric_fn: Callable[[Any], Dict[str, float]],
        data: Any,
    ) -> Dict[str, float]:
        # Return deterministic value for the circuit if provided, else fall back to metric_fn
        if hasattr(circuit, "run_id") and circuit.run_id in self.run_id_to_logit_diff:
            return {"test_logit_diff": float(self.run_id_to_logit_diff[circuit.run_id])}
        return metric_fn(data)


class FakeThings:
    def __init__(self) -> None:
        self.test_data: object = object()
        # Not used by FakeExperiment when mapping provided, but keep shape for realism
        self.test_metrics: Dict[str, Callable[[Any], torch.Tensor]] = {
            "logit_diff": lambda data: torch.tensor(1.23),
        }


class MockCircuit:
    """Mock circuit for general testing."""
    def __init__(self, run_id: str) -> None:
        self.run_id: str = run_id


class MockJaccardCircuit:
    """Mock circuit that can be used with Jaccard index functions."""
    def __init__(self, run_id: str, edges: Set[tuple], nodes: Set[str]) -> None:
        self.run_id: str = run_id
        self.edges: Set[tuple] = edges
        self.nodes: Set[str] = nodes
    
    def get_present_edges(self) -> Set[tuple]:
        """Mock method for get_present_edges_from_correspondence."""
        return self.edges
    
    def get_present_nodes(self) -> Set[str]:
        """Mock method for get_present_nodes_from_correspondence."""
        return self.nodes


def _make_circuit_batch(run_ids: List[str], run_id_to_logit_diff: Mapping[str, float]) -> CircuitBatch:
    """Create a circuit batch for logit difference testing."""
    experiment = FakeExperiment(run_id_to_logit_diff)
    things = FakeThings()
    circuits: Dict[str, MockCircuit] = {rid: MockCircuit(rid) for rid in run_ids}
    run_metadata: Dict[str, Dict[str, Any]] = {rid: {"run_id": rid} for rid in run_ids}

    batch = CircuitBatch(
        circuits=circuits,
        experiment=experiment,
        things=things,
        run_metadata=run_metadata,
        config=ExperimentConfig(),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return batch


def _make_jaccard_circuit_batch(
    run_ids: List[str], 
    run_id_to_edges: Dict[str, Set[tuple]], 
    run_id_to_nodes: Dict[str, Set[str]]
) -> CircuitBatch:
    """Create a circuit batch with Jaccard-compatible circuits."""
    experiment = FakeExperiment({})  # Empty for Jaccard tests
    things = FakeThings()
    circuits: Dict[str, MockJaccardCircuit] = {}
    run_metadata: Dict[str, Dict[str, Any]] = {}
    
    for rid in run_ids:
        circuits[rid] = MockJaccardCircuit(
            rid, 
            run_id_to_edges.get(rid, set()), 
            run_id_to_nodes.get(rid, set())
        )
        run_metadata[rid] = {"run_id": rid}

    batch = CircuitBatch(
        circuits=circuits,
        experiment=experiment,
        things=things,
        run_metadata=run_metadata,
        config=ExperimentConfig(),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return batch
