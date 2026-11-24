# Renderer-response results accompanying Section 6.3

These tables format `data/expected.json`; they add no experiment,
change no input and make no independent-confirmation claim. All
comparisons are exploratory and conditional on fixed models, heads
and donor schedules. The 300-image cohort previously supplied layout
calibration. Visible means at least one decoded recipient part pixel;
donors are unchanged and all test parts are annotated present.

## Complementary 300 images: gradient association differences

Each cell is rho(delta_g) minus rho(s), averaged over all five part
correlations, with a paired image-cluster 95% percentile interval.
All 3,000 draws are defined for these gradient contrasts. Intervals
are pointwise, not multiplicity-adjusted.

| Model | All recipients | Visible recipients |
|---|---|---|
| resnet18_seed0 | -0.0571 [-0.1035, -0.0097] | 0.0083 [-0.0370, 0.0551] |
| resnet18_seed1 | -0.0259 [-0.0707, 0.0170] | 0.0462 [0.0005, 0.0886] |
| resnet50_seed2 | -0.0505 [-0.0956, -0.0071] | 0.0293 [-0.0158, 0.0739] |

The visible R18 seed 1 interval is positive pointwise; the other two
visible intervals cross zero. A universal zero-effect claim does not
follow. Comparing recipient populations is not a causal decomposition.

## Complementary 300 images: CAM point estimates

| Model | rho(s) | rho(delta_g) | Defined bootstrap draws |
|---|---:|---:|---:|
| resnet18_seed0 | 0.2369 | 0.0926 | 98.07% |
| resnet18_seed1 | 0.2563 | 0.0955 | 94.83% |
| resnet50_seed2 | 0.2513 | 0.1167 | 86.47% |

No unconditional CAM interval is claimed. A constant part score makes
the fixed-five-part statistic undefined in that draw; we do not average
a smaller set of parts. Conditional defined-draw diagnostics in the JSON
are not unconditional confidence intervals.

## Part-selection utility and oracle regret

Only R18 seed 0 has the frozen species readout used here. Choose the
highest-scoring part, averaging exact ties. Utility is the signed
true-class/original-best-rival logit-margin drop after deleting that
part, minus the response from uniform selection among eligible parts.
Regret is the largest eligible deletion response minus the selected
response. Both are in readout logit-margin units; no normalization,
human utility or semantic ACE is implied. Concept-map and class-task
targets differ. All utility/regret draws are defined.

### original200, all recipients

| Explainer / selector | Selected minus uniform [95% interval] | Oracle regret [95% interval] | Mean ties |
|---|---|---|---:|
| cam / S | 0.0928 [0.0281, 0.1573] | 2.1382 [2.0041, 2.2785] | 4.1350 |
| cam / G | -0.1452 [-0.3251, 0.0477] | 2.3762 [2.1335, 2.6062] | 1.0400 |
| cam / deltaG | -0.1203 [-0.3027, 0.0697] | 2.3513 [2.1099, 2.5900] | 1.0000 |
| gradient / S | 0.0761 [0.0086, 0.1407] | 2.1549 [2.0177, 2.3078] | 3.6400 |
| gradient / G | -0.1089 [-0.2897, 0.0795] | 2.3399 [2.1010, 2.5818] | 1.0100 |
| gradient / deltaG | 0.0421 [-0.1506, 0.2496] | 2.1889 [1.9376, 2.4294] | 1.0000 |

Uniform-selection oracle regret: 2.2310.

| Explainer | Paired selected-response difference: delta_g minus s [95% interval] |
|---|---|
| cam | -0.2131 [-0.4044, -0.0198] |
| gradient | -0.0340 [-0.2167, 0.1563] |

### original200, visible recipients

| Explainer / selector | Selected minus uniform [95% interval] | Oracle regret [95% interval] | Mean ties |
|---|---|---|---:|
| cam / S | 0.0507 [-0.0074, 0.1097] | 2.1350 [1.9994, 2.2766] | 4.1350 |
| cam / G | -0.1790 [-0.3606, 0.0204] | 2.3647 [2.1210, 2.5948] | 1.0250 |
| cam / deltaG | -0.1630 [-0.3483, 0.0265] | 2.3488 [2.1089, 2.5900] | 1.0000 |
| gradient / S | 0.0340 [-0.0307, 0.0978] | 2.1517 [2.0147, 2.3039] | 3.6400 |
| gradient / G | -0.1509 [-0.3323, 0.0361] | 2.3367 [2.0988, 2.5790] | 1.0100 |
| gradient / deltaG | -0.0031 [-0.1929, 0.2007] | 2.1888 [1.9364, 2.4309] | 1.0000 |

Uniform-selection oracle regret: 2.1857.

| Explainer | Paired selected-response difference: delta_g minus s [95% interval] |
|---|---|
| cam | -0.2137 [-0.4058, -0.0218] |
| gradient | -0.0371 [-0.2222, 0.1532] |

### complement300, all recipients

| Explainer / selector | Selected minus uniform [95% interval] | Oracle regret [95% interval] | Mean ties |
|---|---|---|---:|
| cam / S | 0.1134 [0.0653, 0.1612] | 2.1503 [2.0523, 2.2549] | 4.0833 |
| cam / G | -0.0078 [-0.1782, 0.1625] | 2.2715 [2.0767, 2.4739] | 1.0300 |
| cam / deltaG | -0.0013 [-0.1668, 0.1645] | 2.2650 [2.0570, 2.4718] | 1.0100 |
| gradient / S | 0.0911 [0.0335, 0.1523] | 2.1726 [2.0657, 2.2807] | 3.6800 |
| gradient / G | -0.0771 [-0.2449, 0.0914] | 2.3407 [2.1467, 2.5307] | 1.0067 |
| gradient / deltaG | 0.0210 [-0.1392, 0.1892] | 2.2427 [2.0516, 2.4284] | 1.0100 |

Uniform-selection oracle regret: 2.2637.

| Explainer | Paired selected-response difference: delta_g minus s [95% interval] |
|---|---|
| cam | -0.1148 [-0.2786, 0.0537] |
| gradient | -0.0701 [-0.2300, 0.0997] |

### complement300, visible recipients

| Explainer / selector | Selected minus uniform [95% interval] | Oracle regret [95% interval] | Mean ties |
|---|---|---|---:|
| cam / S | 0.0606 [0.0135, 0.1081] | 2.1457 [2.0485, 2.2518] | 4.0833 |
| cam / G | -0.0594 [-0.2289, 0.1158] | 2.2658 [2.0683, 2.4688] | 1.0233 |
| cam / deltaG | -0.0639 [-0.2329, 0.1041] | 2.2702 [2.0629, 2.4798] | 1.0033 |
| gradient / S | 0.0384 [-0.0147, 0.0969] | 2.1680 [2.0635, 2.2766] | 3.6800 |
| gradient / G | -0.1298 [-0.2942, 0.0365] | 2.3362 [2.1396, 2.5238] | 1.0067 |
| gradient / deltaG | 0.0003 [-0.1611, 0.1695] | 2.2061 [2.0199, 2.3924] | 1.0033 |

Uniform-selection oracle regret: 2.2064.

| Explainer | Paired selected-response difference: delta_g minus s [95% interval] |
|---|---|
| cam | -0.1245 [-0.2867, 0.0408] |
| gradient | -0.0381 [-0.1962, 0.1225] |

A positive s-selector advantage and an unresolved delta_g advantage
do not themselves establish a difference between selectors: use the
paired selector contrasts above. In the complementary all-recipient
cohort both paired contrasts are unresolved. These checks validate
neither spatial score as a universal response ranker.
