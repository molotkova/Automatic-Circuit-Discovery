import pytest

from experiments.robustness.config import ExperimentConfig
from experiments.robustness.core.metrics import MetricComputer
from tests.robustness.test_utils import _make_jaccard_circuit_batch


def test_compute_pairwise_jaccard_indices_returns_symmetric_matrix() -> None:
    """Test that pairwise Jaccard indices returns symmetric results."""
    # Mock the Jaccard functions to return predictable values
    import experiments.robustness.core.metrics as metrics_module
    
    def mock_jaccard_edges(circuit1, circuit2):
        # Return different values based on circuit IDs for testing
        if circuit1.run_id == circuit2.run_id:
            return 1.0  # Self-similarity
        elif (circuit1.run_id, circuit2.run_id) in [("a", "b"), ("b", "a")]:
            return 0.5  # Symmetric similarity
        else:
            return 0.2  # Other pairs
    
    def mock_jaccard_nodes(circuit1, circuit2):
        # Return different values based on circuit IDs for testing
        if circuit1.run_id == circuit2.run_id:
            return 1.0  # Self-similarity
        elif (circuit1.run_id, circuit2.run_id) in [("a", "b"), ("b", "a")]:
            return 0.6  # Symmetric similarity
        else:
            return 0.3  # Other pairs
    
    # Patch the functions
    original_edges = metrics_module.compute_jaccard_index_edges
    original_nodes = metrics_module.compute_jaccard_index_nodes
    metrics_module.compute_jaccard_index_edges = mock_jaccard_edges
    metrics_module.compute_jaccard_index_nodes = mock_jaccard_nodes
    
    try:
        metric_computer = MetricComputer(ExperimentConfig(verbose=False))
        run_ids = ["a", "b", "c"]
        
        # Create simple circuit batch (edges/nodes don't matter since we're mocking)
        batch = _make_jaccard_circuit_batch(
            run_ids,
            {"a": set(), "b": set(), "c": set()},
            {"a": set(), "b": set(), "c": set()}
        )
        
        result = metric_computer.compute_pairwise_jaccard_indices(batch, run_ids)
        
        # Check structure
        assert set(result.keys()) == {"a", "b", "c"}
        for run_id in run_ids:
            assert set(result[run_id].keys()) == {"a", "b", "c"}
        
        # Check symmetry
        assert result["a"]["b"]["edges"] == result["b"]["a"]["edges"]
        assert result["a"]["b"]["nodes"] == result["b"]["a"]["nodes"]
        assert result["a"]["c"]["edges"] == result["c"]["a"]["edges"]
        assert result["a"]["c"]["nodes"] == result["c"]["a"]["nodes"]
        assert result["b"]["c"]["edges"] == result["c"]["b"]["edges"]
        assert result["b"]["c"]["nodes"] == result["c"]["b"]["nodes"]
        
        # Check diagonal (self-similarity)
        assert result["a"]["a"]["edges"] == 1.0
        assert result["a"]["a"]["nodes"] == 1.0
        assert result["b"]["b"]["edges"] == 1.0
        assert result["b"]["b"]["nodes"] == 1.0
        assert result["c"]["c"]["edges"] == 1.0
        assert result["c"]["c"]["nodes"] == 1.0
        
        # Check specific values
        assert result["a"]["b"]["edges"] == 0.5
        assert result["a"]["b"]["nodes"] == 0.6
        
    finally:
        # Restore original functions
        metrics_module.compute_jaccard_index_edges = original_edges
        metrics_module.compute_jaccard_index_nodes = original_nodes

"""
def test_compute_pairwise_jaccard_indices_filters_correctly() -> None:
    # Test that pairwise Jaccard indices filters the circuit batch correctly.
    import experiments.robustness.core.metrics as metrics_module
    
    def mock_jaccard_edges(circuit1, circuit2):
        return 0.5
    
    def mock_jaccard_nodes(circuit1, circuit2):
        return 0.6
    
    # Patch the functions
    original_edges = metrics_module.compute_jaccard_index_edges
    original_nodes = metrics_module.compute_jaccard_index_nodes
    metrics_module.compute_jaccard_index_edges = mock_jaccard_edges
    metrics_module.compute_jaccard_index_nodes = mock_jaccard_nodes
    
    try:
        metric_computer = MetricComputer(ExperimentConfig(verbose=False))
        
        # Create batch with more circuits than we'll use
        all_run_ids = ["a", "b", "c", "d", "e"]
        batch = _make_jaccard_circuit_batch(
            all_run_ids,
            {rid: set() for rid in all_run_ids},
            {rid: set() for rid in all_run_ids}
        )
        
        # Only compute for subset
        requested_run_ids = ["a", "c", "e"]
        result = metric_computer.compute_pairwise_jaccard_indices(batch, requested_run_ids)
        
        # Should only contain requested run_ids
        assert set(result.keys()) == set(requested_run_ids)
        for run_id in requested_run_ids:
            assert set(result[run_id].keys()) == set(requested_run_ids)
        
    finally:
        # Restore original functions
        metrics_module.compute_jaccard_index_edges = original_edges
        metrics_module.compute_jaccard_index_nodes = original_nodes


def test_compute_pairwise_jaccard_indices_handles_empty_list() -> None:
    # Test that pairwise Jaccard indices handles empty run_ids list.
    metric_computer = MetricComputer(ExperimentConfig(verbose=False))
    batch = _make_jaccard_circuit_batch([], {}, {})
    
    result = metric_computer.compute_pairwise_jaccard_indices(batch, [])
    
    assert result == {}


def test_compute_pairwise_jaccard_indices_returns_correct_structure() -> None:
    # Test that pairwise Jaccard indices returns the correct nested structure.
    import experiments.robustness.core.metrics as metrics_module
    
    def mock_jaccard_edges(circuit1, circuit2):
        return 0.5
    
    def mock_jaccard_nodes(circuit1, circuit2):
        return 0.6
    
    # Patch the functions
    original_edges = metrics_module.compute_jaccard_index_edges
    original_nodes = metrics_module.compute_jaccard_index_nodes
    metrics_module.compute_jaccard_index_edges = mock_jaccard_edges
    metrics_module.compute_jaccard_index_nodes = mock_jaccard_nodes
    
    try:
        metric_computer = MetricComputer(ExperimentConfig(verbose=False))
        run_ids = ["x", "y"]
        batch = _make_jaccard_circuit_batch(
            run_ids,
            {"x": set(), "y": set()},
            {"x": set(), "y": set()}
        )
        
        result = metric_computer.compute_pairwise_jaccard_indices(batch, run_ids)
        
        # Check top-level structure
        assert isinstance(result, dict)
        assert set(result.keys()) == {"x", "y"}
        
        # Check second-level structure
        for run_id in run_ids:
            assert isinstance(result[run_id], dict)
            assert set(result[run_id].keys()) == {"x", "y"}
            
            # Check third-level structure (edges/nodes)
            for other_run_id in run_ids:
                jaccard_data = result[run_id][other_run_id]
                assert isinstance(jaccard_data, dict)
                assert "edges" in jaccard_data
                assert "nodes" in jaccard_data
                assert isinstance(jaccard_data["edges"], float)
                assert isinstance(jaccard_data["nodes"], float)
                assert jaccard_data["edges"] == 0.5
                assert jaccard_data["nodes"] == 0.6
        
    finally:
        # Restore original functions
        metrics_module.compute_jaccard_index_edges = original_edges
        metrics_module.compute_jaccard_index_nodes = original_nodes
"""