from dotenv import load_dotenv

load_dotenv()

import argparse
import sys
import torch
from typing import Dict, Any
from pathlib import Path

from acdc.ioi.utils import get_all_ioi_things
from acdc.ioi.ioi_dataset import IOIDataset
from acdc.TLACDCExperiment import TLACDCExperiment
from utils.circuit_utils import load_single_acdc_run

torch.autograd.set_grad_enabled(False)



def get_ioi_average_token_probabilities(probabilities: torch.Tensor, ioi_dataset) -> Dict[str, Any]:
    """
    Get average probabilities for IO and S tokens across the entire IOI dataset.
    
    Args:
        probabilities: torch.Tensor of shape [batch_size, vocab_size]
        ioi_dataset: IOIDataset instance
    
    Returns:
        Dictionary with average IO and S token probabilities and statistics
    """
    batch_size = probabilities.shape[0]
    
    # Get all IO and S token IDs for the batch
    io_token_ids = ioi_dataset.io_tokenIDs[:batch_size]
    s_token_ids = ioi_dataset.s_tokenIDs[:batch_size]
    
    # Extract probabilities for IO and S tokens
    io_probs = []
    s_probs = []
    
    for batch_idx in range(batch_size):
        io_token_id = io_token_ids[batch_idx]
        s_token_id = s_token_ids[batch_idx]
        
        io_prob = probabilities[batch_idx, io_token_id].item()
        s_prob = probabilities[batch_idx, s_token_id].item()
        
        io_probs.append(io_prob)
        s_probs.append(s_prob)
    
    # Calculate averages
    avg_io_prob = sum(io_probs) / len(io_probs)
    avg_s_prob = sum(s_probs) / len(s_probs)
    
    # Calculate additional statistics
    io_vs_s_ratios = [io_prob / s_prob if s_prob > 0 else float('inf') for io_prob, s_prob in zip(io_probs, s_probs)]
    avg_io_vs_s_ratio = sum(io_vs_s_ratios) / len(io_vs_s_ratios) if all(r != float('inf') for r in io_vs_s_ratios) else float('inf')
    
    # Get sample token texts (from first example)
    io_text = ioi_dataset.tokenizer.decode([io_token_ids[0]])
    s_text = ioi_dataset.tokenizer.decode([s_token_ids[0]])
    
    return {
        'average_IO_probability': avg_io_prob,
        'average_S_probability': avg_s_prob,
        'average_IO_vs_S_ratio': avg_io_vs_s_ratio,
        'IO_token_text': io_text,
        'S_token_text': s_text,
        'num_examples': batch_size,
        'IO_probabilities': io_probs,
        'S_probabilities': s_probs,
        'IO_vs_S_ratios': io_vs_s_ratios
    }


def main():
    """Main function to load circuit and calculate probabilities."""
    parser = argparse.ArgumentParser(
        description="Calculate average probabilities for IO and S tokens using a loaded circuit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_probs.py --run-id "my-circuit-run-123"
        """
    )
    
    parser.add_argument(
        "--run-id",
        required=True,
        help="W&B run ID of the circuit to load"
    )
    
    args = parser.parse_args()
    
    # Default values
    device = "cuda"
    num_examples = 100
    project_name = "personal-14/acdc-robustness"
    metric_name = "logit_diff"
    verbose = True
    
    print(f"Loading circuit {args.run_id}...")
    print(f"Device: {device}")
    print(f"Number of examples: {num_examples}")
    print(f"Project: {project_name}")
    print(f"Metric: {metric_name}")
    print("-" * 50)
    
    # Create IOI things
    print("Creating IOI dataset and model...")
    things = get_all_ioi_things(
        num_examples=num_examples,
        device=device,
        metric_name=metric_name,
    )
    
    # Create IOIDataset for token analysis
    print("Creating IOIDataset for token analysis...")
    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=num_examples*2,
        nb_templates=1,
        seed=0
    )
    
    # Create TLACDCExperiment
    print("Creating TLACDCExperiment...")
    tl_model = things.tl_model
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
        verbose=verbose,
        use_pos_embed=False,
        online_cache_cpu=False,
        corrupted_cache_cpu=False,
    )
    
    # Load the circuit
    print(f"Loading circuit {args.run_id}...")
    correspondence = load_single_acdc_run(
        run_id=args.run_id,
        exp=exp,
        things=things,
        project_name=project_name
    )
    
    print(f"Loaded circuit with {correspondence.count_no_edges()} edges")
    
    # Get probabilities using get_probs_on_corr
    print("Calculating probabilities...")
    probabilities = exp.get_probs_on_corr(correspondence, things.test_data)
    
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Calculate average probabilities for IO and S tokens
    print("Analyzing IO and S token probabilities...")
    results = get_ioi_average_token_probabilities(probabilities, ioi_dataset)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Circuit ID: {args.run_id}")
    print(f"Number of examples: {results['num_examples']}")
    print(f"IO token text: '{results['IO_token_text']}'")
    print(f"S token text: '{results['S_token_text']}'")
    print()
    print(f"Average IO probability: {results['average_IO_probability']:.6f}")
    print(f"Average S probability: {results['average_S_probability']:.6f}")
    print(f"Average IO/S ratio: {results['average_IO_vs_S_ratio']:.2f}")
    print()
    
    # Show some individual examples
    print("Individual example probabilities (first 2):")
    print("Example | IO Prob | S Prob | IO/S Ratio")
    print("-" * 40)
    for i in range(min(10, len(results['IO_probabilities']))):
        io_prob = results['IO_probabilities'][i]
        s_prob = results['S_probabilities'][i]
        ratio = results['IO_vs_S_ratios'][i]
        print(f"   {i:2d}   | {io_prob:.4f} | {s_prob:.4f} | {ratio:.2f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()