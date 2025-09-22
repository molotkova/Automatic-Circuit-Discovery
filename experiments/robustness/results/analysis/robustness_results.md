# Robustness Experiment Results

## shuffle_abc_prompts

| Metric | Edges | Nodes | Overall |
|--------|-------|-------|---------|
| baseline jaccard | $0.864 \pm 0.091$ | $0.875 \pm 0.095$ | - |
| jaccard (only perturbed circuits) | $0.843 \pm 0.079$ | $0.857 \pm 0.084$ | - |
| baseline $\Delta_l$ | - | - | $0.078 \pm 0.096$ |
| $\Delta_l$ (only perturbed circuits) | - | - | $1.000 \pm 0.104$ |

## add_random_prefixes

| Metric | Edges | Nodes | Overall |
|--------|-------|-------|---------|
| baseline jaccard | $0.740 \pm 0.014$ | $0.721 \pm 0.021$ | - |
| jaccard (only perturbed circuits) | $0.933 \pm 0.093$ | $0.922 \pm 0.099$ | - |
| baseline $\Delta_l$ | - | - | $0.235 \pm 0.000$ |
| $\Delta_l$ (only perturbed circuits) | - | - | $1.158 \pm 0.068$ |

## swap_dataset_roles

| Metric | Edges | Nodes | Overall |
|--------|-------|-------|---------|
| baseline jaccard | $0.358 \pm 0.034$ | $0.340 \pm 0.030$ | - |
| jaccard (only perturbed circuits) | $0.730 \pm 0.201$ | $0.737 \pm 0.207$ | - |
| baseline $\Delta_l$ | - | - | $0.226 \pm 0.025$ |
| $\Delta_l$ (only perturbed circuits) | - | - | $1.150 \pm 0.069$ |
