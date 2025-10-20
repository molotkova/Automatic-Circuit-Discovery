import time
from typing import Any, Callable, Dict, List, Mapping

import pytest
import torch

from experiments.robustness.config import ExperimentConfig, CircuitBatch
from experiments.robustness.core.metrics import MetricComputer


class FakeCircuit:
    def __init__(self, run_id: str) -> None:
        self.run_id: str = run_id


class FakeExperiment:
    def __init__(self, run_id_to_logit_diff: Mapping[str, float]) -> None:
        self.run_id_to_logit_diff: Mapping[str, float] = run_id_to_logit_diff
        self._setup_called: bool = False

    def setup_corrupted_cache(self) -> None:
        self._setup_called = True

    def call_metric_with_corr(
        self,
        circuit: FakeCircuit,
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


def _make_circuit_batch(run_ids: List[str], run_id_to_logit_diff: Mapping[str, float]) -> CircuitBatch:
    experiment = FakeExperiment(run_id_to_logit_diff)
    things = FakeThings()
    circuits: Dict[str, FakeCircuit] = {rid: FakeCircuit(rid) for rid in run_ids}
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


def test__filter_circuit_batch() -> None:
    metric_computer = MetricComputer(ExperimentConfig(verbose=False))
    all_ids = ["a", "b", "c"]
    batch = _make_circuit_batch(all_ids, {"a": 0.1, "b": 0.2, "c": 0.3})

    filtered = metric_computer._filter_circuit_batch(batch, ["a", "c"])

    print("hellooo")

    assert set(filtered.run_ids) == {"a", "c"}
    assert set(filtered.circuits.keys()) == {"a", "c"}
    assert set(filtered.run_metadata.keys()) == {"a", "c"}
    

"""
def test__filter_circuit_batch_raises_on_missing_id() -> None:
    metric_computer = MetricComputer(ExperimentConfig(verbose=False))
    all_ids = ["a", "b"]
    batch = _make_circuit_batch(all_ids, {"a": 0.1, "b": 0.2})

    with pytest.raises(ValueError):
        metric_computer._filter_circuit_batch(batch, ["a", "missing"])  # missing id


def test__compute_logit_diff_for_circuit_returns_experiment_value() -> None:
    metric_computer = MetricComputer(ExperimentConfig(verbose=False))
    run_ids = ["x"]
    target_value = 0.456
    batch = _make_circuit_batch(run_ids, {"x": target_value})

    circuit = batch.get_circuit("x")
    value = metric_computer._compute_logit_diff_for_circuit(circuit, batch, run_id="x")

    assert pytest.approx(value, rel=1e-6) == target_value


def test_compute_logit_differences_end_to_end_and_sets_up_cache() -> None:
    # End-to-end test over public API, also ensures setup_corrupted_cache is invoked
    cfg = ExperimentConfig(verbose=False)
    metric_computer = MetricComputer(cfg)
    run_ids = ["r1", "r2"]
    mapping = {"r1": 1.0, "r2": 2.5}
    batch = _make_circuit_batch(run_ids, mapping)

    results = metric_computer.compute_logit_differences(batch, run_ids)

    assert set(results.keys()) == set(run_ids)
    assert pytest.approx(results["r1"], rel=1e-6) == 1.0
    assert pytest.approx(results["r2"], rel=1e-6) == 2.5

    # Verify corrupted cache setup was called
    assert batch.experiment._setup_called is True

"""