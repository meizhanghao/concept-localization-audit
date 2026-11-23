# Auditing Concept Localization

Code and numerical materials accompanying **Auditing Concept Localization:
Crossed Controls and Concept Eligibility**.

**Zhanghao Mei · Zuqiang Meng · Xuwei Tang**  
College of Computer, Electronics and Information, Guangxi University

## Overview

A concept map can hit the expected part without showing evidence specific to
its image or concept label. We audit localization scores by crossing two
controls: **own versus donor-image maps**, and **trained versus randomized
concept heads**. Every comparison keeps the recipient's target window fixed.

![Crossed-control audit framework](assets/framework.png)

The audit reports four quantities:

| Quantity | Meaning |
|---|---|
| **S** | Spatial agreement: the map's peak hits the target acceptance region. |
| **G** | Image correspondence: own-map score minus donor-map score. |
| **G_rand** | The same correspondence contrast averaged across matched random heads. |
| **ΔG** | Excess correspondence: G minus G_rand. |

Separate positive image and head checks need not resolve their crossed
interaction. A second audit examines how concept-presence eligibility and
donor rules affect same-part label contrasts. See the
[protocol](docs/protocol.md) for definitions, aggregation and uncertainty.

## Quick start

Use **Python 3.10+** with NumPy. The reference environment is Python 3.10.8 /
NumPy 2.2.6. The complete numerical inputs are included in `data/` (about 25 MB).
The audits run on CPU; no dataset download or GPU is needed.

Download or clone this repository, then run from its root:

```bash
git clone https://github.com/meizhanghao/concept-localization-audit.git
cd concept-localization-audit
python -m pip install -r requirements-reproduce.txt
python run_audit.py
```

Expected terminal output:

```text
Inputs verified: 51 files
core: passed (maximum error ...)
concepts: passed (maximum error ...)
bridge: passed (maximum error ...)
Results saved to .../results
```

All numerical differences must be at most **1e-12**. The runner validates input
SHA-256 hashes before scoring and exits with a nonzero status on failure.
Outputs include full-precision JSON reports, a CSV of the core scores, and a
combined verification report. See [reproduction instructions](docs/reproduction.md)
for individual audits and output fields.

## Included analyses

| Audit | Coverage |
|---|---|
| `core` | 36 localization score cases and 12 paired identity contrasts on FunnyBirds and CUB, including original/disjoint cohorts and eight frozen Loc-CBM adapter settings. |
| `concepts` | Primary same-part label contrast and a both-present conditional analysis, with paired intervals and random-head-adjusted contrasts. |
| `bridge` | CUB donor-rule comparison holding 133 ordered pairs, 7,081 recipient records, maps and macro weights fixed. Own/donor hits are recomputed from native peaks and keypoints. |

Selected reference results:

| Analysis | Estimate | 95% interval |
|---|---:|---:|
| Original CUB, native Loc-CBM, crossed ΔG | 0.039 | [−0.007, 0.084] |
| Disjoint CUB cohort, native Loc-CBM, crossed ΔG | 0.021 | [0.006, 0.037] |
| CUB primary same-part label contrast | 0.120 | [0.108, 0.133] |
| CUB both-present conditional label contrast | 0.003 | [−0.011, 0.019] |
| CUB fixed-recipient donor-rule difference | 0.026 | [0.024, 0.028] |

The primary and both-present contrasts use different conditional populations.
The bridge isolates the donor-rule effect on one fixed recipient set; it does
not decompose the original 0.120 contrast. An interval crossing zero remains
unresolved. Positive spatial correspondence does not establish semantic
correctness or causal faithfulness.

![Spatial agreement and image correspondence](assets/diagnostic.png)

*Paper diagnostic on FunnyBirds. Panel (a) varies acceptance geometry for fixed
maps. Panel (b) separates spatial agreement from correspondence; its reference
bands are conditional intervals. Paired ΔG intervals, rather than band overlap,
determine above-reference evidence.*

## Project structure

```text
.
├── run_audit.py                 # Command-line entry point
├── audit/                      # Scoring, aggregation and paired uncertainty
│   ├── core.py
│   ├── concepts.py
│   ├── bridge.py
│   ├── metrics.py
│   └── cli.py
├── data/                       # Frozen numerical inputs and expected results
│   ├── core_audit/
│   ├── concept_eligibility/
│   ├── donor_bridge/
│   └── checksums.json
├── assets/                     # Figures from the manuscript
├── docs/                       # Protocol, data schema and reproduction guide
├── tests/                      # Control-invariance and integrity checks
└── results/                    # Generated locally; excluded from Git
```

## Reproduction scope

This repository reproduces cached numerical audits. Core own/donor scores and
the donor bridge are recomputed from peaks and target geometry; concept-label
intervals use frozen per-query scores and resampling draws. Source-model
training, raw heatmap extraction, classification-quality experiments and
upper-envelope optimization are outside this package. Dataset images and
model checkpoints are obtained from their original releases when required
for those separate stages.

The [data description](docs/data.md) documents coordinates, arrays and source
projects. [Protocol details](docs/protocol.md) distinguish conditional intervals,
joint image/head intervals and exploratory follow-ups.

## Citation

The accompanying manuscript is in preparation. Repository citation metadata
is provided in [CITATION.cff](CITATION.cff); publication details will be added
when available.

## Acknowledgments

This work uses derived evaluation records from
[CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) and
[FunnyBirds](https://github.com/visinf/funnybirds), and builds on
[Loc-CBM](https://github.com/VIPL-VSU/Loc-CBM) and the
[ProtoCBM scorer](https://github.com/hqhQAQ/ProtoCBM). The released NumPy scorer
implements the cached peak predicate independently. Dataset and model terms
remain those of their respective owners.

For questions about the code or protocol, contact **Zuqiang Meng**
([zqmeng@gxu.edu.cn](mailto:zqmeng@gxu.edu.cn)).
