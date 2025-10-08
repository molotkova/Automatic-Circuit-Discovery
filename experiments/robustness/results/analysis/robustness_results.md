# Robustness Experiment Results

## baseline - metric calibration

| Metric | Edges | Vertices | Overall |
|--------|-------|-------|---------|
| jaccard: calibration | $0.821 \pm 0.071$ | $0.840 \pm 0.075$ | - |
| jaccard: circuit to baseline circuit (averaging over other than baseline circuits) | $0.866 \pm 0.079$ | $0.878 \pm 0.075$ | - |
| $LD$: calibration | - | - | $0.907 \pm 0.107$ |
| $\Delta_{LD}$: circuit minus baseline circuit (averaging over other than baseline circuits) | - | - | $0.060 \pm 0.106$ |
| $\Delta_{LD}$: circuit minus baseline circuit (averaging over all circuits) | - | - | $0.111 \pm 0.152$ |

## shuffle_abc_prompts

| Metric | Edges | Vertices | Overall |
|--------|-------|-------|---------|
| jaccard: calibration | $0.838 \pm 0.077$ | $0.853 \pm 0.081$ | - |
| jaccard: circuit to baseline circuit (averaging over other than baseline circuits) | $0.864 \pm 0.091$ | $0.875 \pm 0.095$ | - |
| jaccard: circuit to baseline circuit (averaging over all circuits) | $0.808 \pm 0.080$ | $0.826 \pm 0.083$ | - |
| $LD$: calibration | - | - | $1.004 \pm 0.109$ |
| $\Delta_{LD}$: circuit minus baseline circuit (averaging over other than baseline circuits) | - | - | $$0.078 \pm 0.096$$ |
| $\Delta_{LD}$: circuit minus baseline circuit (averaging over all circuits) | - | - | $$ 0.150 \pm 0.184 $$ |

## add_random_prefixes

| Metric | Edges | Vertices | Overall |
|--------|-------|-------|---------|
| jaccard: calibration | $0.976 \pm 0.018$ | $0.966 \pm 0.026$ | - |
| jaccard: circuit to baseline circuit (averaging over other than baseline circuits) | $0.740 \pm 0.014$ | $0.721 \pm 0.021$ | - |
| $LD$: calibration | - | - | $1.179 \pm 0.000$ |
| $\Delta_{LD}$: circuit minus baseline circuit (averaging over other than baseline circuits) | - | - |$$0.235 \pm 0.000$$ |

## swap_dataset_roles

| Metric | Edges | Vertices | Overall |
|--------|-------|-------|---------|
| jaccard: calibration | $0.813 \pm 0.103$ | $0.826 \pm 0.093$ | - |
| jaccard: circuit to baseline circuit (averaging over other than baseline circuits) | $0.358 \pm 0.034$ | $0.340 \pm 0.030$ | - |
| $LD$: calibration | - | - | $1.170 \pm 0.024$ |
| $\Delta_{LD}$: circuit minus baseline circuit (averaging over other than baseline circuits) | - | - | $0.226 \pm 0.025$ |
