import random
import torch
from typing import Tuple
from .ioi_dataset import IOIDataset


class DatasetPerturbation:
    """Base class for all dataset perturbations"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def apply(self, ioi_dataset: IOIDataset, abc_dataset: IOIDataset, **kwargs) -> Tuple[IOIDataset, IOIDataset]:
        """Apply perturbation and return modified datasets"""
        raise NotImplementedError
    
    def __str__(self):
        return f"{self.name}: {self.description}"


class ShuffleABCPrompts(DatasetPerturbation):
    """Shuffle the order of prompts in abc_dataset while keeping ioi_dataset unchanged"""
    
    def __init__(self):
        super().__init__(
            name="shuffle_abc_prompts",
            description="Randomly shuffle abc_dataset prompts while keeping ioi_dataset unchanged"
        )
    
    def apply(self, ioi_dataset: IOIDataset, abc_dataset: IOIDataset, **kwargs) -> Tuple[IOIDataset, IOIDataset]:
        seed = kwargs.get("seed", 42)
        
        # Keep ioi_dataset unchanged
        modified_ioi = ioi_dataset
        
        # Create shuffled abc_dataset
        if seed is not None:
            random.seed(seed)
        
        # Get all prompts and shuffle them
        shuffled_prompts = abc_dataset.ioi_prompts.copy()
        random.shuffle(shuffled_prompts)
        
        # Create new dataset with shuffled prompts
        modified_abc = IOIDataset(
            prompt_type=abc_dataset.prompt_type,
            N=abc_dataset.N,
            tokenizer=abc_dataset.tokenizer,
            prompts=shuffled_prompts,  # Use shuffled prompts
            prefixes=abc_dataset.prefixes,
            ioi_prompts_for_word_idxs=abc_dataset.ioi_prompts,  # Keep original for word indices
            prepend_bos=abc_dataset.prepend_bos,
            seed=seed
        )
        
        return modified_ioi, modified_abc


class AddRandomPrefixes(DatasetPerturbation):
    """Add random prefixes to both ioi_dataset and abc_dataset prompts"""
    
    def __init__(self):
        super().__init__(
            name="add_random_prefixes",
            description="Add random prefixes to both datasets, ensuring corresponding prompts get same prefix"
        )
        self.prefixes = [
            "It was a quiet afternoon.",
            "Everything seemed ordinary at first.",
            "The day had just begun.",
            "Nothing unusual had happened so far.",
            "It was time for a short break."
        ]
    
    def apply(self, ioi_dataset: IOIDataset, abc_dataset: IOIDataset, **kwargs) -> Tuple[IOIDataset, IOIDataset]:
        seed = kwargs.get("seed", 42)
        
        # Set seed for reproducible prefix selection
        if seed is not None:
            random.seed(seed)
        
        # Generate random prefix indices for each prompt
        # Same index for corresponding prompts in both ioi and abc datasets
        num_prompts = len(ioi_dataset.ioi_prompts)
        prefix_indices = [random.randint(0, len(self.prefixes) - 1) for _ in range(num_prompts)]
        
        # Apply prefixes to ioi_dataset
        modified_ioi_prompts = []
        for i, prompt in enumerate(ioi_dataset.ioi_prompts):
            prefix = self.prefixes[prefix_indices[i]]
            modified_prompt = prompt.copy()
            modified_prompt["text"] = prefix + " " + prompt["text"]
            modified_ioi_prompts.append(modified_prompt)
        
        # Apply prefixes to abc_dataset
        modified_abc_prompts = []
        for i, prompt in enumerate(abc_dataset.ioi_prompts):
            prefix = self.prefixes[prefix_indices[i]]  # Same prefix index as ioi_dataset
            modified_prompt = prompt.copy()
            modified_prompt["text"] = prefix + " " + prompt["text"]
            modified_abc_prompts.append(modified_prompt)
        
        # Create new datasets with prefixed prompts
        modified_ioi = IOIDataset(
            prompt_type=ioi_dataset.prompt_type,
            N=ioi_dataset.N,
            tokenizer=ioi_dataset.tokenizer,
            prompts=modified_ioi_prompts,
            prefixes=ioi_dataset.prefixes,
            ioi_prompts_for_word_idxs=ioi_dataset.ioi_prompts,  # Keep original for word indices
            prepend_bos=ioi_dataset.prepend_bos,
            seed=seed
        )
        
        modified_abc = IOIDataset(
            prompt_type=abc_dataset.prompt_type,
            N=abc_dataset.N,
            tokenizer=abc_dataset.tokenizer,
            prompts=modified_abc_prompts,
            prefixes=abc_dataset.prefixes,
            ioi_prompts_for_word_idxs=abc_dataset.ioi_prompts,  # Keep original for word indices
            prepend_bos=abc_dataset.prepend_bos,
            seed=seed
        )
        
        return modified_ioi, modified_abc


class SwapDatasetRoles(DatasetPerturbation):
    """Swap the roles of ioi_dataset and abc_dataset"""
    
    def __init__(self):
        super().__init__(
            name="swap_dataset_roles",
            description="Swap ioi_dataset and abc_dataset roles: ioi_dataset becomes abc_dataset (with random names) and abc_dataset becomes ioi_dataset (with consistent names)"
        )
    
    def apply(self, ioi_dataset: IOIDataset, abc_dataset: IOIDataset, **kwargs) -> Tuple[IOIDataset, IOIDataset]:
        # Simply swap the references
        # The ACDC algorithm will now treat:
        # - abc_dataset as the "clean" dataset (ioi_dataset role)
        # - ioi_dataset as the "corrupted" dataset (abc_dataset role)
        
        return abc_dataset, ioi_dataset  # Swapped return order


def get_perturbation(perturbation_name: str) -> DatasetPerturbation:
    """Get perturbation by name"""
    perturbations = {
        "shuffle_abc_prompts": ShuffleABCPrompts(),
        "add_random_prefixes": AddRandomPrefixes(),
        "swap_dataset_roles": SwapDatasetRoles(),
    }
    
    if perturbation_name not in perturbations:
        raise ValueError(f"Unknown perturbation: {perturbation_name}. Available: {list(perturbations.keys())}")
    
    return perturbations[perturbation_name]
