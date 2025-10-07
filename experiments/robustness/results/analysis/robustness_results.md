# Robustness Experiment Results

## baseline - metric calibration

| Metric | Edges | Nodes | Overall |
|--------|-------|-------|---------|
| jaccard: calibration | $0.821 \pm 0.071$ | $0.840 \pm 0.075$ | - |
| jaccard: circuit to baseline circuit | $0.866 \pm 0.079$ | $0.878 \pm 0.075$ | - |
| $LD$: calibration | - | - | $0.907 \pm 0.107$ |
| $\Delta_{LD}$: circuit minus single baseline circuit (averaging over multiple other circuits) | - | - | $0.060 \pm 0.106$ |

## shuffle_abc_prompts

| Metric | Edges | Nodes | Overall |
|--------|-------|-------|---------|
| jaccard: calibration | $0.838 \pm 0.077$ | $0.853 \pm 0.081$ | - |
| jaccard: circuit to baseline circuit | $0.864 \pm 0.091$ | $0.875 \pm 0.095$ | - |
| $LD$: calibration | - | - | $1.004 \pm 0.109$ |
| $\Delta_{LD}$: circuit minus single baseline circuit (averaging over multiple other circuits) | - | - | $$0.078 \pm 0.096$$ |

## add_random_prefixes

| Metric | Edges | Nodes | Overall |
|--------|-------|-------|---------|
| jaccard: calibration | $0.976 \pm 0.018$ | $0.966 \pm 0.026$ | - |
| jaccard: circuit to baseline circuit | $0.740 \pm 0.014$ | $0.721 \pm 0.021$ | - |
| $LD$: calibration | - | - | $1.179 \pm 0.000$ |
| $\Delta_{LD}$: circuit minus single baseline circuit (averaging over multiple other circuits) | - | - |$$0.235 \pm 0.000$$ |

## swap_dataset_roles

| Metric | Edges | Nodes | Overall |
|--------|-------|-------|---------|
| jaccard: calibration | $0.813 \pm 0.103$ | $0.826 \pm 0.093$ | - |
| jaccard: circuit to baseline circuit | $0.358 \pm 0.034$ | $0.340 \pm 0.030$ | - |
| $LD$: calibration | - | - | $1.170 \pm 0.024$ |
| $\Delta_{LD}$: circuit minus single baseline circuit (averaging over multiple other circuits) | - | - | $0.226 \pm 0.025$ |
