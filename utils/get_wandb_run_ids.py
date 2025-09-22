import argparse
import wandb

from dotenv import load_dotenv
load_dotenv()

def get_run_ids(entity, project, filters=None, max_runs=None):
    """
    Fetches run IDs from a Weights & Biases project according to the given filters.

    Args:
        entity (str): The wandb entity (user or team).
        project (str): The wandb project name.
        filters (dict, optional): Filters to apply to the runs.
        max_runs (int, optional): Maximum number of runs to fetch.

    Returns:
        List[str]: List of run IDs matching the filter.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters)
    run_ids = []
    for i, run in enumerate(runs):
        run_ids.append(run.id)
        if max_runs is not None and i + 1 >= max_runs:
            break
    return run_ids

def parse_filter(filter_str):
    """
    Parses a filter string of the form key1=value1,key2=value2 into a dict.
    Supports comma-separated values for a single key: key1=value1,value2,value3
    
    Examples:
    - 'config.perturbation=swap_dataset_roles,config.dataset_seed=2'
    - 'config.dataset_seed=2,3,4,5'
    - 'config.perturbation=shuffle_abc_prompts,add_random_prefixes'
    """
    if not filter_str:
        return None
    filter_dict = {}
    
    # Split by comma, but be careful about commas within values
    items = []
    current_item = ""
    paren_count = 0
    
    for char in filter_str:
        if char == "(":
            paren_count += 1
        elif char == ")":
            paren_count -= 1
        elif char == "," and paren_count == 0:
            items.append(current_item.strip())
            current_item = ""
            continue
        current_item += char
    
    if current_item.strip():
        items.append(current_item.strip())
    
    for item in items:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        
        # Check if value contains comma-separated values
        if "," in v:
            # Parse as list of values
            value_list = []
            for val in v.split(","):
                val = val.strip()
                parsed_val = _parse_value(val)
                value_list.append(parsed_val)
            filter_dict[k] = value_list
        else:
            # Single value
            filter_dict[k] = _parse_value(v)
    
    return filter_dict

def _parse_value(val):
    """Helper function to parse a single value as int, float, bool, or string."""
    if val.lower() == "true":
        return True
    elif val.lower() == "false":
        return False
    else:
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

def main():
    parser = argparse.ArgumentParser(description="Get W&B run IDs according to filter. Defaults to personal-14/acdc-robustness project.")
    parser.add_argument("--entity", default="personal-14", help="WandB entity (user or team) [default: personal-14]")
    parser.add_argument("--project", default="acdc-robustness", help="WandB project name [default: acdc-robustness]")
    parser.add_argument("--filter", default=None, help="Filter string, e.g. 'config.perturbation=swap_dataset_roles,config.dataset_seed=2,3,4'")
    parser.add_argument("--max-runs", type=int, default=None, help="Maximum number of runs to fetch")
    parser.add_argument("--output", default=None, help="Output file to write run IDs (one per line). If not set, prints to stdout.")

    args = parser.parse_args()

    filters = parse_filter(args.filter)
    run_ids = get_run_ids(args.entity, args.project, filters=filters, max_runs=args.max_runs)

    if args.output:
        with open(args.output, "w") as f:
            for run_id in run_ids:
                f.write(run_id + "\n")
        print(f"Wrote {len(run_ids)} run IDs to {args.output}")
    else:
        for run_id in run_ids:
            print(run_id)

if __name__ == "__main__":
    main()

