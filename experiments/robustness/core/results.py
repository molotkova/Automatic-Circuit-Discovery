import json
from pathlib import Path
from typing import Optional, Union, Any
from datetime import datetime

import sys


from experiments.robustness.config import ExperimentResult


class ResultsManager:
    """
    Handles saving and loading experiment results.

    This class provides methods for serializing, saving, loading, and exporting
    experiment results with proper validation and error handling.
    """

    def __init__(self, output_dir: Union[str, Path], run_timestamp: Optional[str] = None):
        """
        Initialize the results manager.

        Args:
            output_dir: Directory where results will be saved
            run_timestamp: Optional timestamp for this run (creates timestamped subdirectory)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create data directory
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Create timestamped run directory within data directory
        if run_timestamp:
            # Convert timestamp to directory-friendly format
            timestamp_str = run_timestamp.replace(":", "-").replace(" ", "_")
            self.run_dir = self.data_dir / f"run_{timestamp_str}"
        else:
            # Use current timestamp if none provided
            current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_dir = self.data_dir / f"run_{current_timestamp}"
        
        self.run_dir.mkdir(parents=True, exist_ok=True)

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

        output_path = self.run_dir / filename

        # Convert result to dictionary
        result_dict = result.to_dict()

        # Save to JSON file
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        return output_path

    def save_plot(
        self, 
        figure: Any, 
        filename: str, 
        dpi: int = 300,
        bbox_inches: str = 'tight'
    ) -> Path:
        """
        Save a matplotlib figure to file.

        Args:
            figure: Matplotlib figure object to save
            filename: Filename for the plot (e.g., "plot.png")
            dpi: Resolution for the saved figure (default: 300)
            bbox_inches: Bounding box inches setting (default: 'tight')

        Returns:
            Path to the saved plot file
        """
        output_path = self.run_dir / filename
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the figure
        figure.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
        
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

