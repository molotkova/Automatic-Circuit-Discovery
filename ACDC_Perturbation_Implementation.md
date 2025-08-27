# ACDC Dataset Perturbation Implementation

This document describes how to implement dataset perturbation support in the ACDC main script to test algorithm robustness.

## Overview

The implementation adds support for applying a single dataset perturbation per ACDC run, allowing researchers to systematically test how different types of data corruption affect circuit discovery performance.

## 1. Modify the Argument Parser in main.py

Add these new arguments after the existing ones:

```python
parser.add_argument('--perturbation', type=str, required=False, default=None, 
                   help='Dataset perturbation to apply (e.g., shuffle_abc_prompts)')
parser.add_argument('--perturbation-seed', type=int, required=False, default=42,
                   help='Seed for perturbation randomization')

```

## 2. Process the New Arguments

Add these lines after processing other args:

```python
PERTURBATION = args.perturbation
PERTURBATION_SEED = args.perturbation_seed

```

## 3. Modify the IOI Task Setup

Update the IOI task setup section:

```python
elif TASK == "ioi":
    num_examples = 100
    things = get_all_ioi_things(
        num_examples=num_examples, 
        device=DEVICE, 
        metric_name=args.metric,
        perturbation_name=PERTURBATION,
        perturbation_kwargs={
            "seed": PERTURBATION_SEED
        } if PERTURBATION else None
    )
    print("Dataset and model ready")
```

## 4. Update the Notebook Command String

Modify the notebook command string to include perturbation options:

```python
if ipython is not None:
    # We are in a notebook
    # you can put the command you would like to run as the ... in r"""..."""
    args = parser.parse_args(
        [line.strip() for line in r"""--task=ioi\
--threshold=0.71\
--indices-mode=reverse\
--first-cache-cpu=False\
--second-cache-cpu=False\
--max-num-epochs=100000\
--perturbation=shuffle_abc_prompts\
--perturbation-seed=42""".split("\\\n")]
    )
```

## 5. Update the WandB Run Name

Modify the wandb run name to include perturbation info:

```python
# Modify the wandb run name to include perturbation info
if WANDB_RUN_NAME is None or IPython.get_ipython() is not None:
    perturbation_suffix = f"_pert_{PERTURBATION}" if PERTURBATION else ""
    WANDB_RUN_NAME = f"{ct()}{'_randomindices' if INDICES_MODE=='random' else ''}_{THRESHOLD}{'_zero' if ZERO_ABLATION else ''}{perturbation_suffix}"
else:
    assert WANDB_RUN_NAME is not None, "I want named runs, always"
```

## 6. Create the Perturbation Framework

Create a new file `acdc/ioi/perturbations.py`:

```python
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
        # Same index for corresponding prompts in both datasets
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
        # Simply swap the references - much simpler and more efficient
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
```

## 7. Modify the utils.py File

Add import at the top of `acdc/ioi/utils.py`:

```python
# Add import at the top
from .perturbations import get_perturbation
```

Modify the `get_all_ioi_things` function:

```python
def get_all_ioi_things(
    num_examples, 
    device, 
    metric_name, 
    kl_return_one_element=True,
    perturbation_name: str = None,
    perturbation_kwargs: dict = None
):
    tl_model = get_gpt2_small(device=device)
    
    # Create base datasets
    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=num_examples*2,
        nb_templates=1,
        seed=0,
    )

    abc_dataset = (
        ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )
    
    # Apply perturbation if specified
    if perturbation_name is not None:
        perturbation = get_perturbation(perturbation_name)
        kwargs = perturbation_kwargs or {}
        print(f"Applying perturbation: {perturbation}")
        ioi_dataset, abc_dataset = perturbation.apply(ioi_dataset, abc_dataset, **kwargs)
    
    # Rest of the function remains the same...
    seq_len = ioi_dataset.toks.shape[1]
    assert seq_len == 16, f"Well, I thought ABBA #1 was 16 not {seq_len} tokens long..."
    
    # ... existing code continues unchanged ...
```

## 8. Usage Examples

### Command Line Usage

```bash
# Run with shuffle perturbation
python main.py --task=ioi --threshold=0.71 --perturbation=shuffle_abc_prompts --perturbation-seed=42



# Run with prefix perturbation
python main.py --task=ioi --threshold=0.71 --perturbation=add_random_prefixes --perturbation-seed=42

# Run with role swap perturbation
python main.py --task=ioi --threshold=0.71 --perturbation=swap_dataset_roles
```

### Notebook Usage

```python
# In the notebook command string, you can specify:
r"""--task=ioi\
--threshold=0.71\
--perturbation=shuffle_abc_prompts\
--perturbation-seed=42"""

# Or with prefix perturbation:
r"""--task=ioi\
--threshold=0.71\
--perturbation=add_random_prefixes\
--perturbation-seed=42"""

# Or with role swap perturbation:
r"""--task=ioi\
--threshold=0.71\
--perturbation=swap_dataset_roles"""
```

## 9. Available Perturbations

### shuffle_abc_prompts
- **Description**: Randomly shuffle the order of prompts in abc_dataset
- **Parameters**: `seed` (default: 42)
- **Effect**: Changes prompt ordering while maintaining individual prompt structure



### add_random_prefixes
- **Description**: Add random prefixes to both ioi_dataset and abc_dataset prompts
- **Parameters**: `seed` (default: 42)
- **Effect**: Adds one of five predefined prefixes to each prompt, ensuring corresponding prompts in both datasets get the same prefix
- **Prefixes**:
  - "It was a quiet afternoon."
  - "Everything seemed ordinary at first."
  - "The day had just begun."
  - "Nothing unusual had happened so far."
  - "It was time for a short break."

### swap_dataset_roles
- **Description**: Swap the roles of ioi_dataset and abc_dataset
- **Parameters**: None (operation is deterministic)
- **Effect**: 
  - ioi_dataset becomes abc_dataset (with random names but consistent structure)
  - abc_dataset becomes ioi_dataset (with consistent names but corrupted structure)
- **Purpose**: Tests how the model performs when the "clean" and "corrupted" datasets are swapped

## 10. Key Benefits

1. **Single Perturbation Per Run**: Clean, focused experiments
2. **Command Line Control**: Easy to script multiple runs
3. **Reproducible**: Seeds ensure consistent results
4. **Extensible**: Easy to add new perturbations
5. **Backward Compatible**: Original behavior unchanged when no perturbation specified
6. **WandB Integration**: Perturbation info included in run names

## 11. Adding New Perturbations

To add a new perturbation:

1. Create a new class inheriting from `DatasetPerturbation`
2. Implement the `apply` method
3. Register it in the `get_perturbation` function
4. Add any new command line arguments if needed

Example:

```python
class MyNewPerturbation(DatasetPerturbation):
    def __init__(self):
        super().__init__(
            name="my_new_perturbation",
            description="Description of what this perturbation does"
        )
    
    def apply(self, ioi_dataset: IOIDataset, abc_dataset: IOIDataset, **kwargs) -> Tuple[IOIDataset, IOIDataset]:
        # Implementation here
        return modified_ioi, modified_abc
```

## 12. Testing the Implementation

After implementing the changes:

1. Test without perturbation (should work as before):
   ```bash
   python main.py --task=ioi --threshold=0.71
   ```

2. Test with perturbation:
   ```bash
   python main.py --task=ioi --threshold=0.71 --perturbation=shuffle_abc_prompts
   ```

3. Test with prefix perturbation:
   ```bash
   python main.py --task=ioi --threshold=0.71 --perturbation=add_random_prefixes
   ```

4. Test with role swap perturbation:
   ```bash
   python main.py --task=ioi --threshold=0.71 --perturbation=swap_dataset_roles
   ```

3. Verify that the perturbation is applied and logged correctly

This implementation allows you to run **one perturbation per ACDC run** while maintaining the existing workflow and making it easy to systematically test different types of dataset corruption for robustness analysis.
