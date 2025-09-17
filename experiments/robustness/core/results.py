import json
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime

import sys


from experiments.robustness.config import ExperimentResult


class ResultsManager:
    """
    Handles saving and loading experiment results.

    This class provides methods for serializing, saving, loading, and exporting
    experiment results with proper validation and error handling.
    """

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the results manager.

        Args:
            output_dir: Directory where results will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.logs_dir = self.output_dir / "logs"
        self.data_dir = self.output_dir / "data"
        self.logs_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

    def save_results(
        self, result: ExperimentResult, filename: Optional[str] = None
    ) -> Path:
        """
        Save experiment result to JSON file.

        Args:
            result: ExperimentResult object to save
            filename: Optional custom filename (defaults to auto-generated)

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = result.get_filename()

        output_path = self.data_dir / filename

        # Convert result to dictionary
        result_dict = result.to_dict()

        # Save to JSON file
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        return output_path

    def load_results(self, filepath: Union[str, Path]) -> ExperimentResult:
        """
        Load experiment result from JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            ExperimentResult object
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            result_dict = json.load(f)

        return ExperimentResult.from_dict(result_dict)

    def save_results_batch(
        self, results: List[ExperimentResult], prefix: str = "batch"
    ) -> List[Path]:
        """
        Save multiple results in a batch.

        Args:
            results: List of ExperimentResult objects
            prefix: Prefix for batch filenames

        Returns:
            List of paths to saved files
        """
        saved_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, result in enumerate(results):
            filename = f"{prefix}_{timestamp}_{i:03d}_{result.experiment_type}.json"
            path = self.save_results(result, filename)
            saved_paths.append(path)

        return saved_paths
