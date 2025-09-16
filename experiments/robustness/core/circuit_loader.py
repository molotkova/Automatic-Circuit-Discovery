import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import traceback
import gc
import torch

import sys
sys.path.append('.')

from experiments.robustness.config import ExperimentConfig, CircuitBatch
from utils.circuit_utils import load_single_acdc_run
from utils.utils import get_acdc_runs
from acdc.ioi.utils import get_all_ioi_things
from acdc.TLACDCExperiment import TLACDCExperiment


class CircuitLoader:
    """
    Loads multiple circuits with shared experiment setup.
    
    This class creates a single experiment setup (things and TLACDCExperiment)
    per batch to minimize resource usage and improve performance.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the circuit loader.
        
        Args:
            config: Experiment configuration containing device, project settings, etc.
        """
        self.config = config
        self._experiment_cache: Optional[Tuple[Any, Any]] = None  # (things, exp)
        
    def _create_experiment_setup(self) -> Tuple[Any, Any]:
        """
        Create shared experiment setup (things and TLACDCExperiment).
        
        Returns:
            Tuple of (things, exp) objects
        """
        if self.config.verbose:
            print("Creating experiment setup...")
        
        things = get_all_ioi_things(
            num_examples=self.config.num_examples,
            device=self.config.device,
            metric_name=self.config.metric_name
        )

        tl_model = things.tl_model
        tl_model.reset_hooks()

        gc.collect()
        torch.cuda.empty_cache()

        tl_model.reset_hooks()

        exp = TLACDCExperiment(
        model=tl_model,
        threshold=100_000,
        early_exit=False,
        using_wandb=False,
        zero_ablation=False,
        ds=things.test_data,
        ref_ds=things.test_patch_data,
        metric=things.validation_metric,
        second_metric=None,
        verbose=True,
        use_pos_embed=False,
        online_cache_cpu=False,
        corrupted_cache_cpu=False,
        )
        
        return things, exp
    
    def _get_experiment_setup(self) -> Tuple[Any, Any]:
        """
        Get or create experiment setup with caching.
        
        Returns:
            Tuple of (things, exp) objects
        """
        if self._experiment_cache is None:
            self._experiment_cache = self._create_experiment_setup()
        return self._experiment_cache
    
    def _load_single_circuit(self, run_id: str, things: Any, exp: Any) -> Any:
        """
        Load a single circuit.
        
        Args:
            run_id: W&B run ID
            things: AllDataThings object
            exp: TLACDCExperiment object
            
        Returns:
            TLACDCCorrespondence object
        """
        if self.config.verbose:
            print(f"Loading circuit {run_id}...")
        
        correspondence = load_single_acdc_run(
            run_id=run_id,
            exp=exp,
            things=things,
            project_name=self.config.project_name
        )
        
        if self.config.verbose:
            print(f"Loaded circuit {run_id} with {correspondence.count_no_edges()} edges")
        
        return correspondence
    
    def _get_run_metadata(self, run_id: str, correspondence: Any) -> Dict[str, Any]:
        """
        Extract metadata from a loaded circuit.
        
        Args:
            run_id: W&B run ID
            correspondence: TLACDCCorrespondence object
            
        Returns:
            Dictionary containing run metadata
        """
        # Get basic circuit information
        num_edges = correspondence.count_no_edges()
        num_nodes = len(correspondence.nodes)
        
        metadata = {
            "run_id": run_id,
            "num_edges": num_edges,
            "num_nodes": num_nodes,
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.config.device,
            "project_name": self.config.project_name
        }
        
        # Get additional metadata from correspondence if available
        if hasattr(correspondence, 'metadata'):
            metadata.update(correspondence.metadata)
        
        return metadata
    
    def load_circuits_batch(self, run_ids: List[str]) -> CircuitBatch:
        """
        Load multiple circuits in a single batch with shared experiment setup.
        
        Args:
            run_ids: List of W&B run IDs to load
            
        Returns:
            CircuitBatch containing all successfully loaded circuits
            
        Raises:
            ValueError: If no circuits could be loaded successfully
        """
        if not run_ids:
            raise ValueError("No run IDs provided")
        
        if self.config.verbose:
            print(f"Loading {len(run_ids)} circuits in batch...")
        
        start_time = time.time()
        
        things, exp = self._get_experiment_setup()
        
        circuits: Dict[str, Any] = {}
        run_metadata: Dict[str, Dict[str, Any]] = {}
        
        for run_id in run_ids:
            correspondence = self._load_single_circuit(run_id, things, exp)
            circuits[run_id] = correspondence
            run_metadata[run_id] = self._get_run_metadata(run_id, correspondence)
        
        success_count = len(circuits)
        load_time = time.time() - start_time
        
        if self.config.verbose:
            print(f"Batch loading complete:")
            print(f"   Successfully loaded: {success_count}/{len(run_ids)} circuits")
        
        batch = CircuitBatch(
            circuits=circuits,
            experiment=exp,
            things=things,
            run_metadata=run_metadata,
            config=self.config,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return batch
    
    def load_single_circuit(self, run_id: str) -> CircuitBatch:
        """
        Load a single circuit (convenience method).
        
        Args:
            run_id: W&B run ID to load
            
        Returns:
            CircuitBatch containing the single circuit
        """
        return self.load_circuits_batch([run_id])
    
    def clear_cache(self) -> None:
        """
        Clear the experiment setup cache to free memory.
        
        This should be called when done with a batch to free up memory,
        especially if loading many different batches.
        """
        if self._experiment_cache is not None:
            if self.config.verbose:
                print("Clearing experiment setup cache...")
            
            # Clear the cache
            self._experiment_cache = None
            
            # Force garbage collection if enabled
            if self.config.memory_cleanup:
                gc.collect()
                
                # Clear CUDA cache if using GPU
                if self.config.device == "cuda":
                    torch.cuda.empty_cache()
            
            if self.config.verbose:
                print("Cache cleared")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get information about the current cache status.
        
        Returns:
            Dictionary containing cache information
        """
        return {
            "cached": self._experiment_cache is not None,
            "device": self.config.device,
            "memory_cleanup": self.config.memory_cleanup,
            "verbose": self.config.verbose
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clear cache on exit."""
        self.clear_cache()
    
    def __del__(self):
        """Destructor - ensure cache is cleared."""
        self.clear_cache()
