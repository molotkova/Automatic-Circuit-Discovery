from dotenv import load_dotenv
load_dotenv()

import wandb

api = wandb.Api()

# Filter for exact run ID
filters = {
    "state": "finished",
    "name": "sjr6k1ip"  # No $in needed for single ID
}

runs = api.runs("personal-14/acdc-robustness", filters=filters)

for run in runs:
    print(f"Found run: {run.id}, {run.name}, {run.state}")