import pytest

from experiments.robustness.config import ExperimentConfig
from experiments.robustness.core.metrics import MetricComputer
from tests.robustness.test_utils import _make_circuit_batch


def test__filter_circuit_batch() -> None:
    metric_computer = MetricComputer(ExperimentConfig(verbose=False))
    all_ids = ["a", "b", "c"]
    batch = _make_circuit_batch(all_ids, {"a": 0.1, "b": 0.2, "c": 0.3})

    filtered = metric_computer._filter_circuit_batch(batch, ["a", "c"])

    assert set(filtered.run_ids) == {"a", "c"}
    assert set(filtered.circuits.keys()) == {"a", "c"}
    assert set(filtered.run_metadata.keys()) == {"a", "c"}


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


def test_compute_statistics_calculates_correct_values() -> None:
    """Test that compute_statistics returns correct mean, std, min, max, and count."""
    metric_computer = MetricComputer(ExperimentConfig(verbose=False))
    values = [1.0, 2.0, 3.0]
    
    result = metric_computer.compute_statistics(values)
    
    # Expected values for [1.0, 2.0, 3.0]
    expected_mean = 2.0  # (1.0 + 2.0 + 3.0) / 3
    expected_std = 1.0   # sqrt(((1-2)² + (2-2)² + (3-2)²) / 2) = sqrt(2/2) = 1.0
    expected_min = 1.0
    expected_max = 3.0
    expected_count = 3
    
    assert pytest.approx(result["mean"], rel=1e-6) == expected_mean
    assert pytest.approx(result["std"], rel=1e-6) == expected_std
    assert pytest.approx(result["min"], rel=1e-6) == expected_min
    assert pytest.approx(result["max"], rel=1e-6) == expected_max
    assert result["count"] == expected_count