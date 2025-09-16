# Robustness Experiments Framework Implementation

## Overview

The Robustness Experiments Framework is a modular system designed to analyze the robustness of ACDC (Automatic Circuit Discovery) circuits loaded from Weights & Biases. The framework implements 4 specific experiments to evaluate circuit similarity, performance, and robustness across different runs.

## Architecture

### Core Design Principles

1. **Reuse Existing Code**: 90% of functionality leverages existing ACDC functions
2. **Efficient Resource Usage**: Single experiment setup per batch (not per circuit)
3. **Explicit Typing**: Comprehensive type annotations throughout
4. **Modular Design**: Easy to extend with new experiments
5. **Structured Results**: JSON format with clear metadata and statistics

### Framework Structure

```
experiments/robustness/
├── __init__.py                 # Main module exports
├── run_experiments.py          # CLI entry point
├── example_usage.py            # Usage examples
├── README.md                   # User documentation
├── IMPLEMENTATION.md           # This file
├── config/                     # Configuration management
│   ├── __init__.py
│   └── experiment_config.py    # Configuration classes
├── core/                       # Core framework components
│   ├── __init__.py
│   ├── experiment_runner.py    # Main orchestrator
│   ├── circuit_loader.py       # Efficient circuit loading
│   ├── metrics.py              # Metric computation utilities
│   └── results.py              # Results management
├── experiments/                # Individual experiment implementations
│   ├── __init__.py
│   ├── logit_diff_analysis.py  # Experiment 1
│   ├── pairwise_jaccard.py     # Experiment 2
│   ├── baseline_jaccard.py     # Experiment 3
│   └── baseline_logit_diff.py  # Experiment 4
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── data_structures.py      # Common data structures
└── results/                    # Results storage
    ├── logs/                   # Experiment logs
    └── data/                   # JSON results files
```

## Implementation Details

### 1. Reusable Functions (`utils/circuit_utils.py`)

The framework reuses existing functions from `ioi_perturbation_evaluation.py` by moving them to a centralized location:

#### Functions Moved:
- `get_present_edges_from_correspondence()` - Extract present edges from correspondence
- `get_present_nodes_from_correspondence()` - Extract present nodes from correspondence  
- `compute_jaccard_index_edges()` - Compute Jaccard index for edges
- `compute_jaccard_index_nodes()` - Compute Jaccard index for nodes
- `compute_logit_diff_relative_change()` - Compute relative change in logit difference
- `load_single_acdc_run()` - Load single ACDC run from W&B

#### Key Modification:
The `load_single_acdc_run()` function was modified to take `exp` and `things` as arguments instead of creating them internally:

```python
def load_single_acdc_run(
    run_id: str,
    exp: TLACDCExperiment,        # Pre-configured experiment
    things: Any,                  # Pre-configured things
    project_name: str = "personal-14/acdc-robustness"
) -> TLACDCCorrespondence:
```

This enables efficient batch loading by reusing the same experiment setup for multiple circuits.

### 2. Configuration System (`config/experiment_config.py`)

#### `ExperimentConfig` Class:
```python
@dataclass
class ExperimentConfig:
    # W&B settings
    project_name: str = "personal-14/acdc-robustness"
    device: str = "cuda"
    
    # Experiment settings
    num_examples: int = 100
    metric_name: str = "logit_diff"
    
    # Paths
    output_dir: Path = Path("experiments/robustness/results")
    cache_dir: Path = Path("~/.cache/acdc_robustness")
    log_dir: Path = Path("experiments/robustness/results/logs")
    
    # Performance settings
    memory_cleanup: bool = True
    verbose: bool = True
```

#### `CircuitBatch` Class:
```python
@dataclass
class CircuitBatch:
    circuits: Dict[str, TLACDCCorrespondence]  # run_id -> circuit
    experiment: TLACDCExperiment
    things: Any  # AllDataThings object
    run_metadata: Dict[str, Dict[str, Any]]  # run_id -> metadata
    config: ExperimentConfig
    timestamp: str
```

#### `ExperimentResult` Class:
```python
@dataclass
class ExperimentResult:
    experiment_type: str
    run_ids: List[str]
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str
    config: Dict[str, Any]
    computation_time: float = 0.0
```

### 3. Core Components

#### Circuit Loader (`core/circuit_loader.py`)

**Purpose**: Efficiently loads multiple circuits with shared experiment setup.

**Key Features**:
- Creates single `things` and `TLACDCExperiment` instance per batch
- Uses modified `load_single_acdc_run()` for individual circuit loading
- Handles memory cleanup and error recovery
- Returns `CircuitBatch` with all circuits and shared setup

**Implementation**:
```python
class CircuitLoader:
    def load_circuits_batch(self, run_ids: List[str]) -> CircuitBatch:
        # Create single experiment setup
        things = get_all_ioi_things(...)
        exp = TLACDCExperiment(...)
        
        # Load all circuits using shared setup
        circuits = {}
        for run_id in run_ids:
            correspondence = load_single_acdc_run(run_id, exp, things, ...)
            circuits[run_id] = correspondence
        
        return CircuitBatch(circuits, exp, things, ...)
```

#### Metrics Computer (`core/metrics.py`)

**Purpose**: Computes various metrics on circuits efficiently.

**Key Methods**:
- `compute_logit_differences()` - Compute logit diff for each circuit
- `compute_pairwise_jaccard_indices()` - Compute pairwise Jaccard indices
- `compute_baseline_jaccard_indices()` - Compute baseline vs circuits Jaccard
- `compute_baseline_logit_diff_relative_change()` - Compute relative changes
- `compute_statistics()` - Compute mean, std, min, max statistics

**Implementation**:
```python
class MetricComputer:
    def compute_logit_differences(self, circuit_batch: CircuitBatch) -> Dict[str, float]:
        # Ensure corrupted cache is set up
        circuit_batch.experiment.setup_corrupted_cache()
        
        # Define metric function
        def get_logit_diff_metric(data: torch.Tensor) -> Dict[str, float]:
            return {f"test_{name}": fn(data).item() for name, fn in circuit_batch.things.test_metrics.items()}
        
        # Compute for each circuit
        logit_diffs = {}
        for run_id, circuit in circuit_batch.circuits.items():
            metrics = circuit_batch.experiment.call_metric_with_corr(
                circuit, get_logit_diff_metric, circuit_batch.things.test_data
            )
            logit_diffs[run_id] = metrics["test_logit_diff"]
        
        return logit_diffs
```

#### Results Manager (`core/results.py`)

**Purpose**: Handles saving and loading experiment results.

**Key Features**:
- JSON serialization with structured format
- Automatic filename generation with timestamps
- Results validation and loading
- Summary report generation
- CSV export for analysis

**Implementation**:
```python
class ResultsManager:
    def save_results(self, result: ExperimentResult, filename: Optional[str] = None) -> Path:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.experiment_type}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        result_dict = result.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        return output_path
```

### 4. Individual Experiments

#### Experiment 1: Logit Difference Analysis (`experiments/logit_diff_analysis.py`)

**Purpose**: Analyze logit differences across multiple circuits.

**Implementation**:
```python
class LogitDiffAnalysis:
    def run(self, run_ids: List[str]) -> ExperimentResult:
        # Load circuits in batch
        circuit_batch = self.loader.load_circuits_batch(run_ids)
        
        # Compute logit differences
        logit_diffs = self.metrics.compute_logit_differences(circuit_batch)
        
        # Compute statistics
        values = list(logit_diffs.values())
        summary = self.metrics.compute_statistics(values)
        
        return ExperimentResult(
            experiment_type="logit_difference_analysis",
            run_ids=run_ids,
            results={
                "individual_logit_diffs": logit_diffs,
                "summary": summary
            },
            ...
        )
```

#### Experiment 2: Pairwise Jaccard Similarity (`experiments/pairwise_jaccard.py`)

**Purpose**: Compute pairwise Jaccard indices between all circuits.

**Implementation**:
```python
class PairwiseJaccardAnalysis:
    def run(self, run_ids: List[str]) -> ExperimentResult:
        # Load circuits in batch
        circuit_batch = self.loader.load_circuits_batch(run_ids)
        
        # Compute pairwise Jaccard indices
        pairwise_results = self.metrics.compute_pairwise_jaccard_indices(circuit_batch)
        
        # Compute statistics for edges and nodes separately
        summary = self.metrics.compute_jaccard_statistics(pairwise_results)
        
        return ExperimentResult(
            experiment_type="pairwise_jaccard_similarity",
            run_ids=run_ids,
            results={
                "pairwise_jaccard_indices": pairwise_results,
                "summary": summary
            },
            ...
        )
```

#### Experiment 3: Baseline Jaccard Similarity (`experiments/baseline_jaccard.py`)

**Purpose**: Compare baseline circuit similarity with other circuits.

**Implementation**:
```python
class BaselineJaccardAnalysis:
    def run(self, baseline_run_id: str, run_ids: List[str]) -> ExperimentResult:
        # Include baseline in the run_ids for loading
        all_run_ids = [baseline_run_id] + run_ids
        circuit_batch = self.loader.load_circuits_batch(all_run_ids)
        
        # Compute baseline Jaccard indices
        baseline_results = self.metrics.compute_baseline_jaccard_indices(
            baseline_run_id, circuit_batch
        )
        
        # Compute statistics
        summary = self.metrics.compute_jaccard_statistics(baseline_results)
        
        return ExperimentResult(
            experiment_type="baseline_jaccard_similarity",
            run_ids=all_run_ids,
            results={
                "baseline_run_id": baseline_run_id,
                "baseline_jaccard_indices": baseline_results,
                "summary": summary
            },
            ...
        )
```

#### Experiment 4: Baseline Logit Difference Robustness (`experiments/baseline_logit_diff.py`)

**Purpose**: Analyze baseline circuit robustness via relative logit difference changes.

**Implementation**:
```python
class BaselineLogitDiffAnalysis:
    def run(self, baseline_run_id: str, run_ids: List[str]) -> ExperimentResult:
        # Include baseline in the run_ids for loading
        all_run_ids = [baseline_run_id] + run_ids
        circuit_batch = self.loader.load_circuits_batch(all_run_ids)
        
        # Compute baseline logit difference relative changes
        relative_changes = self.metrics.compute_baseline_logit_diff_relative_change(
            baseline_run_id, circuit_batch
        )
        
        # Compute statistics
        values = list(relative_changes.values())
        summary = self.metrics.compute_statistics(values)
        
        return ExperimentResult(
            experiment_type="baseline_logit_diff_robustness",
            run_ids=all_run_ids,
            results={
                "baseline_run_id": baseline_run_id,
                "relative_changes": relative_changes,
                "summary": summary
            },
            ...
        )
```

### 5. Main Experiment Runner (`core/experiment_runner.py`)

**Purpose**: Orchestrates all robustness experiments.

**Key Methods**:
- `run_logit_difference_analysis()` - Run Experiment 1
- `run_logit_difference_similarity()` - Run Experiment 2
- `run_logit_difference_baseline_similarity()` - Run Experiment 3
- `run_logit_difference_baseline_robustness()` - Run Experiment 4
- `run_all_experiments()` - Run all experiments in sequence

**Implementation**:
```python
class RobustnessExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.loader = CircuitLoader(config)
        self.metrics = MetricComputer(config)
        self.results_manager = ResultsManager(config.output_dir)
        
        # Initialize experiment classes
        self.logit_diff_analysis = LogitDiffAnalysis(config)
        self.pairwise_jaccard = PairwiseJaccardAnalysis(config)
        self.baseline_jaccard = BaselineJaccardAnalysis(config)
        self.baseline_logit_diff = BaselineLogitDiffAnalysis(config)
    
    def run_all_experiments(self, run_ids: List[str], baseline_run_id: Optional[str] = None) -> List[ExperimentResult]:
        results = []
        
        # Experiment 1 & 2 (no baseline needed)
        results.append(self.run_logit_difference_analysis(run_ids))
        results.append(self.run_logit_difference_similarity(run_ids))
        
        # Experiments 3 & 4 (require baseline)
        if baseline_run_id:
            results.append(self.run_logit_difference_baseline_similarity(baseline_run_id, run_ids))
            results.append(self.run_logit_difference_baseline_robustness(baseline_run_id, run_ids))
        
        return results
```

## Results Format

### JSON Structure

```json
{
  "experiment_type": "logit_difference_analysis",
  "timestamp": "2024-01-15T10:30:00Z",
  "run_ids": ["run1", "run2", "run3"],
  "config": {
    "project_name": "personal-14/acdc-robustness",
    "device": "cuda",
    "num_examples": 100
  },
  "results": {
    "individual_logit_diffs": {
      "run1": 0.85,
      "run2": 0.92,
      "run3": 0.78
    },
    "summary": {
      "mean": 0.85,
      "std": 0.07,
      "min": 0.78,
      "max": 0.92,
      "count": 3
    }
  },
  "metadata": {
    "total_circuits": 3,
    "successfully_loaded": 3,
    "computation_time_seconds": 45.2
  }
}
```

### Statistics Computed

For each experiment, the framework computes:
- **Mean**: Average value across all measurements
- **Standard Deviation**: Measure of variability
- **Min/Max**: Range of values
- **Count**: Number of successful measurements

## Usage Examples

### Command Line Interface

```bash
# Run all experiments
python experiments/robustness/run_experiments.py --experiment all --run-ids run1 run2 run3 --baseline-id run1

# Run specific experiment
python experiments/robustness/run_experiments.py --experiment logit-diff --run-ids run1 run2 run3

# Run with custom configuration
python experiments/robustness/run_experiments.py --experiment all --run-ids run1 run2 run3 --baseline-id run1 --device cpu --num-examples 50
```

### Python API

```python
from experiments.robustness import RobustnessExperimentRunner, ExperimentConfig

# Create configuration
config = ExperimentConfig(
    project_name="personal-14/acdc-robustness",
    device="cuda",
    num_examples=100
)

# Create runner
runner = RobustnessExperimentRunner(config)

# Run individual experiments
result1 = runner.run_logit_difference_analysis(["run1", "run2", "run3"])
result2 = runner.run_logit_difference_similarity(["run1", "run2", "run3"])
result3 = runner.run_logit_difference_baseline_similarity("run1", ["run2", "run3"])
result4 = runner.run_logit_difference_baseline_robustness("run1", ["run2", "run3"])

# Run all experiments
results = runner.run_all_experiments(["run1", "run2", "run3"], baseline_run_id="run1")
```

## Key Implementation Benefits

### 1. Efficiency
- **Single Resource Creation**: Creates `things` and `TLACDCExperiment` once per batch
- **Batch Processing**: Loads all circuits together rather than one-by-one
- **Memory Management**: Proper cleanup and caching strategies

### 2. Reusability
- **90% Existing Code**: Leverages existing ACDC functions without modification
- **Centralized Functions**: All circuit analysis functions in `utils/circuit_utils.py`
- **Consistent Patterns**: Follows same patterns as existing codebase

### 3. Extensibility
- **Modular Design**: Easy to add new experiments
- **Clear Interfaces**: Well-defined interfaces between components
- **Configuration-Driven**: Easy to customize behavior

### 4. Maintainability
- **Explicit Typing**: Comprehensive type annotations
- **Error Handling**: Graceful handling of missing runs and failures
- **Structured Results**: Clear, processable output format

### 5. Usability
- **CLI Interface**: Easy command-line usage
- **Python API**: Programmatic access with detailed examples
- **Comprehensive Documentation**: Clear usage instructions

## Dependencies

The framework depends on existing ACDC components:
- `utils/utils.py` - W&B loading and `get_acdc_runs()`
- `utils/circuit_utils.py` - Circuit analysis functions
- `acdc/ioi/utils.py` - IOI experiment setup (`get_all_ioi_things()`)
- `acdc/TLACDCExperiment.py` - Experiment management
- `acdc/TLACDCCorrespondence.py` - Circuit representation

## Future Extensions

The framework is designed to be easily extensible:

1. **New Experiments**: Add new experiment classes following existing patterns
2. **New Metrics**: Add new metric computation functions to `MetricComputer`
3. **New Result Formats**: Extend `ResultsManager` for additional output formats
4. **New Circuit Types**: Extend `CircuitLoader` for different circuit sources
5. **New Analysis Tools**: Add analysis utilities to `utils/data_structures.py`

The modular architecture ensures that extensions can be added without modifying existing code.
