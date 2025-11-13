import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import torch

import sys

from experiments.robustness.config import ExperimentConfig, ExperimentResult, CircuitBatch
from experiments.robustness.core import CircuitLoader
from acdc.ioi.ioi_dataset import IOIDataset

torch.autograd.set_grad_enabled(False)


def get_logits_on_corr(exp, corr, data: torch.Tensor) -> torch.Tensor:
    """
    Get logits at last prediction token with a new correspondence.
    
    Args:
        exp: TLACDCExperiment object
        corr: The correspondence to use for the forward pass
        data: Input data tensor
    
    Returns:
        Logits over vocabulary for the last position [batch_size, vocab_size]
    """
    old_exp_corr = exp.corr
    try:
        exp.corr = corr
        exp.model.reset_hooks()
        exp.setup_model_hooks(
            add_sender_hooks=True,
            add_receiver_hooks=True,
            doing_acdc_runs=False,
        )
        logits = exp.model(data)
        # Extract logits for the last position
        last_pos_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        return last_pos_logits
    finally:
        exp.corr = old_exp_corr


class IOSDistributionAnalysis:
    """
    Analyze IO and S token logit distributions across multiple circuits.
    
    This experiment takes a list of run IDs, loads the corresponding circuits,
    computes logits for IO and S tokens at the last prediction token for each
    sample in the test set, and creates violin plots showing the distributions.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the IO-S distribution analysis experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.loader = CircuitLoader(config)

    def _extract_io_s_logits(
        self, 
        circuits_dict: Dict[str, Any],
        exp: Any,
        io_token_ids: List[int],
        s_token_ids: List[int],
        test_data: torch.Tensor
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Extract IO and S token logits for multiple circuits.
        
        Args:
            circuits_dict: Dictionary mapping run_id to circuit correspondence objects
            exp: TLACDCExperiment instance for evaluation
            io_token_ids: List of IO token IDs for test data samples
            s_token_ids: List of S token IDs for test data samples
            test_data: Test data tensor for evaluation
        
        Returns:
            Dictionary mapping run_id to dictionary with 'io_logits' and 's_logits' lists
        """
        all_results = {}
        batch_size = test_data.shape[0]
        
        # Validate token IDs length matches batch size
        assert len(io_token_ids) == batch_size, f"io_token_ids length {len(io_token_ids)} != batch_size {batch_size}"
        assert len(s_token_ids) == batch_size, f"s_token_ids length {len(s_token_ids)} != batch_size {batch_size}"
        
        for run_id, circuit in circuits_dict.items():
            # Get logits at last position using the circuit
            logits = get_logits_on_corr(exp, circuit, test_data)
            # logits shape: [batch_size, vocab_size]
            
            # Extract logits for IO and S tokens
            io_logits = []
            s_logits = []
            
            for batch_idx in range(batch_size):
                io_token_id = io_token_ids[batch_idx]
                s_token_id = s_token_ids[batch_idx]
                
                io_logit = logits[batch_idx, io_token_id].item()
                s_logit = logits[batch_idx, s_token_id].item()
                
                io_logits.append(io_logit)
                s_logits.append(s_logit)
            
            all_results[run_id] = {
                'io_logits': io_logits,
                's_logits': s_logits
            }
        
        return all_results

    def _create_violin_plot(
        self,
        all_io_logits: List[float],
        all_s_logits: List[float],
        perturbation_name: str,
        verbose_plot: bool = False,
        dataset_seeds: Optional[Dict[str, Any]] = None,
        perturbation_seeds: Optional[Dict[str, Any]] = None
    ) -> Figure:
        """
        Create violin plot for IO and S token logit distributions.
        
        Args:
            all_io_logits: List of all IO logit values across all circuits
            all_s_logits: List of all S logit values across all circuits
            perturbation_name: Name of the perturbation (e.g., "add random prefixes")
            verbose_plot: If True, add text box with unique seed values
            dataset_seeds: Dictionary mapping run_id to dataset_seed (already filtered to runs in plot)
            perturbation_seeds: Dictionary mapping run_id to perturbation_seed (already filtered to runs in plot)
        
        Returns:
            Matplotlib figure object
        """
        # Set font to OpenSans for all text elements
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Open Sans', 'DejaVu Sans', 'Arial', 'sans-serif']
        
        # Explicit color mapping: IO = #aef0c7, S = #dfdaff
        color_mapping = {
            'IO': '#aef0c7',
            'S': '#dfdaff'
        }
        
        # Prepare data for plotting using pandas DataFrame
        data = pd.DataFrame({
            'Token': ['IO'] * len(all_io_logits) + ['S'] * len(all_s_logits),
            'Logit Value': all_io_logits + all_s_logits
        })
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        
        # Create violin plot
        ax = sns.violinplot(
            data=data,
            x='Token',
            y='Logit Value',
            hue='Token',
            palette=[color_mapping['IO'], color_mapping['S']],
            inner='box',  # Show box plot inside violin
            cut=0,  # Extend to include min/max values
            legend=False  # Hide legend since we're using hue for coloring only
        )
        
        # Customize plot
        plt.xlabel('Token', fontsize=12, fontfamily='sans-serif')
        plt.ylabel('Logit Value', fontsize=12, fontfamily='sans-serif')
        plt.title(f'IO and S Token Logit Distributions - {perturbation_name}', fontsize=14, fontfamily='sans-serif')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add text box with unique seed values if verbose_plot is True
        if verbose_plot and (dataset_seeds is not None or perturbation_seeds is not None):
            # Get unique values (seeds are already filtered to only include runs in the plot)
            unique_dataset_seeds = sorted(set(v for v in (dataset_seeds or {}).values() if v is not None))
            unique_perturbation_seeds = sorted(set(v for v in (perturbation_seeds or {}).values() if v is not None))
            
            # Build text box content
            text_lines = []
            if unique_dataset_seeds:
                text_lines.append(f"Dataset seeds: {', '.join(map(str, unique_dataset_seeds))}")
            if unique_perturbation_seeds:
                text_lines.append(f"Perturbation seeds: {', '.join(map(str, unique_perturbation_seeds))}")
            
            if text_lines:
                text_content = '\n'.join(text_lines)
                # Add text box in upper left corner
                ax.text(0.02, 0.98, text_content, 
                       transform=ax.transAxes,
                       fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontfamily='sans-serif')
        
        # Set font for tick labels
        ax.tick_params(labelsize=10)
        for label in ax.get_xticklabels():
            label.set_fontfamily('sans-serif')
        for label in ax.get_yticklabels():
            label.set_fontfamily('sans-serif')
        
        # Tight layout to avoid label cutoff
        plt.tight_layout()
        
        return fig

    def run(self, run_ids: List[str], circuit_batch: CircuitBatch = None, verbose_plot: bool = False) -> tuple[ExperimentResult, Figure]:
        """
        Run the IO-S distribution analysis experiment.

        Args:
            run_ids: List of W&B run IDs to analyze
            circuit_batch: Optional pre-loaded circuit batch for caching
            verbose_plot: If True, add text box with unique seed values to the plot

        Returns:
            Tuple of (ExperimentResult, matplotlib figure) - ResultsManager will handle saving the figure
        """
        if self.config.verbose:
            print(f"Running IO-S distribution analysis for {len(run_ids)} circuits...")

        # Load circuits in batch if not provided
        if circuit_batch is None:
            circuit_batch = self.loader.load_circuits_batch(run_ids)

        # Reuse things and experiment from CircuitBatch (already set up during circuit loading)
        things = circuit_batch.things
        exp = circuit_batch.experiment
        
        # Create IOIDataset for token analysis
        # Note: test_data uses num_examples*2 samples, but only num_examples to num_examples*2 are test
        # So we need to create dataset with N=num_examples*2 to match the test_data size
        ioi_dataset = IOIDataset(
            prompt_type="ABBA",
            N=self.config.num_examples * 2,
            nb_templates=1,
            seed=0
        )
        
        # Assert that all test_data matches ioi_dataset tokens
        # test_data = default_data[num_examples:, :] where default_data = ioi_dataset.toks[:num_examples*2, :seq_len-1]
        # So test_data corresponds to ioi_dataset.toks[num_examples:, :seq_len-1]
        seq_len = ioi_dataset.toks.shape[1]
        ioi_dataset_test_toks = ioi_dataset.toks.long()[self.config.num_examples:, :seq_len-1].to(self.config.device)
        assert torch.equal(things.test_data, ioi_dataset_test_toks), \
            f"test_data does not match ioi_dataset.toks[{self.config.num_examples}:, :{seq_len-1}]. " \
            f"Shapes: test_data={things.test_data.shape}, ioi_dataset={ioi_dataset_test_toks.shape}"
        
        # Extract circuits from circuit_batch into a dictionary
        circuits_dict = {run_id: circuit_batch.get_circuit(run_id) for run_id in run_ids}
        
        # Get IO and S token IDs for test data (test_data = default_data[num_examples:, :])
        # So we need token IDs from num_examples onwards
        io_token_ids = ioi_dataset.io_tokenIDs[self.config.num_examples:]
        s_token_ids = ioi_dataset.s_tokenIDs[self.config.num_examples:]

        # Extract logits for all circuits
        all_results = self._extract_io_s_logits(
            circuits_dict=circuits_dict,
            exp=exp,
            io_token_ids=io_token_ids,
            s_token_ids=s_token_ids,
            test_data=things.test_data
        )
        
        # Flatten results and maintain run_id mapping
        all_io_logits: List[float] = []
        all_s_logits: List[float] = []
        run_results: Dict[str, Dict[str, List[float]]] = {}

        for run_id in run_ids:
            if run_id in all_results:
                logits_dict = all_results[run_id]
                all_io_logits.extend(logits_dict['io_logits'])
                all_s_logits.extend(logits_dict['s_logits'])
                run_results[run_id] = logits_dict
                
                if self.config.verbose:
                    print(f"  Extracted {len(logits_dict['io_logits'])} logits for circuit {run_id}")

        # Format perturbation name for display
        perturbation_display_map = {
            "add_random_prefixes": "add random prefixes",
            "shuffle_abc_prompts": "shuffle abc prompts",
            "swap_dataset_roles": "swap dataset roles",
            None: "no perturbation"
        }
        perturbation_name = perturbation_display_map.get(self.config.perturbation, self.config.perturbation or "no perturbation")
        
        # Filter dataset_seeds and perturbation_seeds to only include successfully processed run_ids
        # (only runs that are in run_results, i.e., runs that contributed to the plot)
        successfully_processed_ids = list(run_results.keys())
        filtered_dataset_seeds = {rid: circuit_batch.dataset_seeds.get(rid) for rid in successfully_processed_ids if circuit_batch.dataset_seeds and rid in circuit_batch.dataset_seeds} if circuit_batch.dataset_seeds else None
        filtered_perturbation_seeds = {rid: circuit_batch.perturbation_seeds.get(rid) for rid in successfully_processed_ids if circuit_batch.perturbation_seeds and rid in circuit_batch.perturbation_seeds} if circuit_batch.perturbation_seeds else None
        
        # Create violin plot figure (ResultsManager will handle saving)
        plot_filename = f"io_s_distribution_{self.config.perturbation or 'none'}.png"
        plot_figure = self._create_violin_plot(
            all_io_logits, 
            all_s_logits,
            perturbation_name=perturbation_name,
            verbose_plot=verbose_plot,
            dataset_seeds=filtered_dataset_seeds,
            perturbation_seeds=filtered_perturbation_seeds
        )

        # Compute statistics
        summary_stats = {
            'io_mean': float(np.mean(all_io_logits)),
            'io_std': float(np.std(all_io_logits)),
            'io_min': float(np.min(all_io_logits)),
            'io_max': float(np.max(all_io_logits)),
            's_mean': float(np.mean(all_s_logits)),
            's_std': float(np.std(all_s_logits)),
            's_min': float(np.min(all_s_logits)),
            's_max': float(np.max(all_s_logits)),
            'num_samples': len(all_io_logits),
            'num_circuits': len(run_results)
        }

        # Create experiment result
        result = ExperimentResult(
            experiment_type="io_s_distribution",
            run_ids=run_ids,
            results={
                'all_io_logits': all_io_logits,
                'all_s_logits': all_s_logits,
                'per_circuit_logits': run_results,
                'summary': summary_stats,
                'plot_filename': plot_filename  # Store filename for reference
            },
            metadata={
                'total_circuits': len(run_ids),
                'successfully_processed': len(run_results),
                'num_test_samples': len(all_io_logits),
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config.to_dict(),
            dataset_seeds=circuit_batch.dataset_seeds,
            perturbation_seeds=circuit_batch.perturbation_seeds,
        )

        if self.config.verbose:
            print(f"IO-S distribution analysis complete:")
            print(f"  Processed: {len(run_results)}/{len(run_ids)} circuits")
            print(f"  Total samples: {len(all_io_logits)}")
            print(f"  IO logit mean: {summary_stats['io_mean']:.4f} ± {summary_stats['io_std']:.4f}")
            print(f"  S logit mean: {summary_stats['s_mean']:.4f} ± {summary_stats['s_std']:.4f}")

        # Return both result and figure (experiment runner will handle saving the figure)
        return result, plot_figure

