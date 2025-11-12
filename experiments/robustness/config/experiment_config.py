from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json


@dataclass
class ExperimentConfig:
    """
    Configuration parameters for robustness experiments.

    This class contains all configurable parameters for running robustness
    experiments, including W&B settings, experiment parameters, paths, and
    performance settings.
    """

    # W&B settings
    project_name: str = "personal-14/acdc-robustness"
    device: str = "cuda"

    # Experiment settings
    num_examples: int = 100
    metric_name: str = "logit_diff"
    perturbation: Optional[str] = None

    # Paths
    output_dir: Path = Path("experiments/robustness/results")
    cache_dir: Path = Path("~/.cache/acdc_robustness")
    log_dir: Path = Path("experiments/robustness/results/logs")

    # Performance settings
    memory_cleanup: bool = True
    verbose: bool = True

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir).expanduser()
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.device not in ["cuda", "cpu", "mps"]:
            raise ValueError(
                f"Unsupported device: {self.device}. Must be 'cuda', 'cpu', or 'mps'"
            )

        if self.num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {self.num_examples}")

    def to_dict(self) -> Dict[str, Any]:
        config_dict = asdict(self)

        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        for key in ["output_dir", "cache_dir", "log_dir"]:
            if key in config_dict and isinstance(config_dict[key], str):
                config_dict[key] = Path(config_dict[key])
        return cls(**config_dict)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class CircuitBatch:
    """
    Container for a batch of loaded circuits with shared experiment setup.

    This class holds multiple circuits loaded from W&B runs along with the
    shared experiment setup (TLACDCExperiment and AllDataThings) to enable
    efficient batch processing.
    """

    circuits: Dict[str, Any]  # run_id -> TLACDCCorrespondence
    experiment: Any  # TLACDCExperiment object
    things: Any  # AllDataThings object
    run_metadata: Dict[str, Dict[str, Any]]  # run_id -> metadata
    config: ExperimentConfig
    timestamp: str
    dataset_seeds: Optional[Dict[str, Any]] = None  # run_id -> dataset_seed
    perturbation_seeds: Optional[Dict[str, Any]] = None  # run_id -> perturbation_seed

    def __post_init__(self):
        if not self.circuits:
            raise ValueError("CircuitBatch must contain at least one circuit")

        if len(self.circuits) != len(self.run_metadata):
            raise ValueError("Number of circuits must match number of metadata entries")

        circuit_run_ids = set(self.circuits.keys())
        metadata_run_ids = set(self.run_metadata.keys())
        if circuit_run_ids != metadata_run_ids:
            raise ValueError("Circuit run_ids must match metadata run_ids")

    @property
    def run_ids(self) -> List[str]:
        """Get list of run IDs in this batch."""
        return list(self.circuits.keys())

    @property
    def num_circuits(self) -> int:
        """Get number of circuits in this batch."""
        return len(self.circuits)

    def get_circuit(self, run_id: str) -> Any:
        """Get circuit by run ID."""
        if run_id not in self.circuits:
            raise KeyError(f"Run ID {run_id} not found in batch")
        return self.circuits[run_id]

    def get_metadata(self, run_id: str) -> Dict[str, Any]:
        """Get metadata for a specific run ID."""
        if run_id not in self.run_metadata:
            raise KeyError(f"Run ID {run_id} not found in metadata")
        return self.run_metadata[run_id]

    def filter(self, run_ids: List[str]) -> "CircuitBatch":
        """
        Create a filtered CircuitBatch containing only the specified run_ids.
        
        Args:
            run_ids: List of run IDs to include in the filtered batch
            
        Returns:
            New CircuitBatch containing only the specified circuits
        """
        # Check that all requested run_ids exist
        missing_ids = [run_id for run_id in run_ids if run_id not in self.circuits]
        if missing_ids:
            raise ValueError(f"Run IDs not found in circuit batch: {missing_ids}")
        
        # Filter circuits, metadata, and seeds
        filtered_circuits = {run_id: self.circuits[run_id] for run_id in run_ids}
        filtered_metadata = {run_id: self.run_metadata[run_id] for run_id in run_ids}
        filtered_dataset_seeds = {run_id: self.dataset_seeds.get(run_id) for run_id in run_ids} if self.dataset_seeds else None
        filtered_perturbation_seeds = {run_id: self.perturbation_seeds.get(run_id) for run_id in run_ids} if self.perturbation_seeds else None
        
        return CircuitBatch(
            circuits=filtered_circuits,
            experiment=self.experiment,
            things=self.things,
            run_metadata=filtered_metadata,
            config=self.config,
            timestamp=self.timestamp,
            dataset_seeds=filtered_dataset_seeds,
            perturbation_seeds=filtered_perturbation_seeds,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excluding non-serializable objects)."""
        return {
            "run_ids": self.run_ids,
            "num_circuits": self.num_circuits,
            "run_metadata": self.run_metadata,
            "config": self.config.to_dict(),
            "timestamp": self.timestamp,
        }


@dataclass
class ExperimentResult:
    """
    Structured results from robustness experiments.

    This class provides a standardized format for storing and serializing
    experiment results with metadata and configuration information.
    """

    experiment_type: str
    run_ids: List[str]
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str
    config: Dict[str, Any]
    computation_time: float = 0.0
    dataset_seeds: Optional[Dict[str, Any]] = None
    perturbation_seeds: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.run_ids:
            raise ValueError("ExperimentResult must contain at least one run_id")

        if not self.results:
            raise ValueError("ExperimentResult must contain results")

        if self.computation_time < 0:
            raise ValueError("computation_time must be non-negative")

    @property
    def num_runs(self) -> int:
        """Get number of runs in this result."""
        return len(self.run_ids)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics if available in results."""
        if "summary" in self.results:
            return self.results["summary"]
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> "ExperimentResult":
        """Create ExperimentResult from dictionary."""
        return cls(**result_dict)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save result to JSON file."""
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentResult":
        """Load result from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            result_dict = json.load(f)
        return cls.from_dict(result_dict)

    def get_filename(self) -> str:
        """Generate filename for saving this result."""
        # Get perturbation from config, default to "none" if None
        perturbation = self.config.get("perturbation")
        perturbation_str = "none" if perturbation is None else perturbation
        
        return f"{self.experiment_type}_{perturbation_str}.json"

    def print_summary(self) -> None:
        """Print a summary of the experiment results."""
        print(f"\n=== {self.experiment_type.upper()} RESULTS ===")
        print(f"Timestamp: {self.timestamp}")
        print(f"Number of runs: {self.num_runs}")
        print(f"Computation time: {self.computation_time:.2f} seconds")
        print(f"Run IDs: {', '.join(self.run_ids)}")

        # Print summary statistics if available
        summary = self.get_summary_stats()
        if summary:
            print(f"\nSummary Statistics:")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        print("=" * 50)


# Type aliases for better code readability
ConfigDict = Dict[str, Any]
ResultDict = Dict[str, Any]
MetadataDict = Dict[str, Any]
CircuitDict = Dict[str, Any]
