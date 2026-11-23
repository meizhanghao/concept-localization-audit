# Supplemental sensitivity tables

All analyses are exploratory. The original and frozen cohorts are not pooled.

## Native Loc-CBM adapters

Original intervals condition on fixed heads/donors and resample images. Frozen intervals jointly resample images/heads with fixed donors. These are pointwise intervals, not simultaneous tests.

| Cohort | Setting | Excess | 95% interval |
|---|---|---:|---|
| Original | own_box__bilinear__row_major | 0.039237 | [-0.001889, 0.080757] |
| Original | own_box__bilinear__center | 0.039138 | [-0.001914, 0.080995] |
| Original | own_box__bicubic__row_major | 0.028032 | [-0.014269, 0.071339] |
| Original | own_box__bicubic__center | 0.029568 | [-0.012448, 0.072756] |
| Original | full_image__bilinear__row_major | 0.000529 | [-0.042915, 0.044516] |
| Original | full_image__bilinear__center | 0.000663 | [-0.042848, 0.044554] |
| Original | full_image__bicubic__row_major | 0.004572 | [-0.037796, 0.049502] |
| Original | full_image__bicubic__center | 0.004706 | [-0.037542, 0.049602] |
| Frozen | own_box__bilinear__row_major | 0.021382 | [0.005979, 0.037007] |
| Frozen | own_box__bilinear__center | 0.021890 | [0.006358, 0.037621] |
| Frozen | own_box__bicubic__row_major | 0.025572 | [0.009733, 0.042001] |
| Frozen | own_box__bicubic__center | 0.026445 | [0.010179, 0.042787] |
| Frozen | full_image__bilinear__row_major | 0.041915 | [0.024071, 0.058888] |
| Frozen | full_image__bilinear__center | 0.042306 | [0.024670, 0.059196] |
| Frozen | full_image__bicubic__row_major | 0.042188 | [0.025024, 0.059043] |
| Frozen | full_image__bicubic__center | 0.042579 | [0.025695, 0.059004] |

## All six paired width comparisons

One fixed family of six wider-minus-narrower differences. The simultaneous intervals use the 95th percentile of the common-bootstrap maximum absolute centered deviation divided by each contrast's bootstrap standard deviation. These approximate bands do not correct exploratory choice of the four widths.

| Wider - narrower | Difference | Pointwise 95% interval | Six-comparison simultaneous interval |
|---|---:|---|---|
| 24 - 16 | 0.004778 | [-0.002590, 0.012215] | [-0.004770, 0.014326] |
| 36 - 16 | 0.019490 | [0.006971, 0.031746] | [0.003503, 0.035478] |
| 45 - 16 | 0.015318 | [0.001078, 0.029657] | [-0.002854, 0.033489] |
| 36 - 24 | 0.014712 | [0.004501, 0.025176] | [0.001575, 0.027850] |
| 45 - 24 | 0.010539 | [-0.002511, 0.023138] | [-0.005869, 0.026948] |
| 45 - 36 | -0.004173 | [-0.012214, 0.003651] | [-0.014447, 0.006101] |

## Anatomical influence

All ten groups follow the scorer's fixed attribute-prefix/keypoint mapping. Attribute-macro weights are retained; leave-out renormalizes over remaining attributes. Intervals below are descriptive and pointwise. All groups remain in the main estimate.

| Part | Attributes | Queries | Excess [95% interval] | Signed macro contribution | Leave-out excess [95% interval] |
|---|---:|---:|---|---:|---|
| back | 5 | 798 | 0.028812 [-0.012032, 0.068938] | 0.003601 | 0.020320 [0.003737, 0.037691] |
| beak | 6 | 1778 | -0.011657 [-0.047061, 0.024458] | -0.001749 | 0.027212 [0.009601, 0.045125] |
| belly | 4 | 942 | 0.032376 [-0.016303, 0.082298] | 0.003238 | 0.020160 [0.002525, 0.038069] |
| breast | 8 | 1452 | 0.012905 [-0.029902, 0.056959] | 0.002581 | 0.023501 [0.008334, 0.039508] |
| crown | 2 | 407 | 0.043059 [-0.009708, 0.098145] | 0.002153 | 0.020241 [0.004599, 0.036341] |
| eye | 1 | 660 | 0.225053 [0.105180, 0.336235] | 0.005626 | 0.016160 [-0.000242, 0.032303] |
| forehead | 3 | 533 | -0.029583 [-0.091574, 0.033588] | -0.002219 | 0.025514 [0.008339, 0.042903] |
| leg | 3 | 479 | 0.007378 [-0.046559, 0.063133] | 0.000553 | 0.022517 [0.006840, 0.038367] |
| nape | 3 | 507 | 0.003283 [-0.046841, 0.050920] | 0.000246 | 0.022849 [0.006477, 0.039485] |
| throat | 5 | 767 | 0.058802 [-0.004736, 0.119615] | 0.007350 | 0.016036 [0.000626, 0.031585] |

Full precision and all 40 attribute rows: [cub_robustness_results.json](cub_robustness_results.json).
