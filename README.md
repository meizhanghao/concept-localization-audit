# Auditing Concept Localization

CPU/NumPy rescoring materials for **Auditing Concept Localization: Crossed
Controls and Concept Eligibility** (research manuscript prepared for BlockSys
2027; not a claim of acceptance).

Authors: Zhanghao Mei, Zuqiang Meng (corresponding author), Xuwei Tang.
College of Computer, Electronics and Information, Guangxi University.

## Quick start

Python 3.10+ and NumPy 1.24+ are required. No GPU, PyTorch, network connection,
raw images, model checkpoints or imports from the original project are needed
to run the released rescoring checks.

```bash
python -m pip install -r requirements.txt
python -I -B run_all.py
```

The command verifies SHA-256 fingerprints, executes three independent scripts,
and writes `.results/verification.json` plus execution logs. Checks fail on a
numerical mismatch larger than 1e-12. The exact bootstrap draws and donor
schedules are frozen inputs, rather than freshly sampled runs selected to
recover the reported intervals.

## What is reproduced

| Directory | Inputs and outputs |
|---|---|
| `core_audit/` | Derived peaks, query windows/keypoints, donor schedules and bootstrap draws; recomputes 36 score cases and 12 paired identity contrasts, including the original/frozen CUB cohorts, FunnyBirds controls and all eight frozen Loc-CBM adapter settings. |
| `concept_eligibility/` | Cached query scores, native concept peaks, label assignments and geometry; reproduces the primary same-part label contrast (0.1203188044792221) and both-present conditional contrast (0.0032497229168705), with paired intervals and random-head-adjusted contrasts. |
| `donor_bridge/` | Recomputes own/donor hits from native peaks and keypoints, fixing the 133 ordered pairs and 7,081 recipient records. Target-positive-only versus both-present donors gives -0.022478228490025506 versus 0.0032497229168705; their paired difference is 0.025727951406896016 [0.02363788493129574, 0.02776533399932188]. |

**The 133-pair donor bridge is a CUB Loc-CBM analysis, not FunnyBirds.** It
isolates donor eligibility on one fixed recipient set. It neither decomposes
the primary 0.120 contrast nor estimates a causal effect of concept presence.
The primary and both-present analyses use different conditional populations.
Unresolved differences are not equivalence tests or evidence of semantic
failure. Positive G or excess Delta G does not certify semantic correctness or
causal faithfulness.

## Deliberate limits

This is a **cached numerical rescoring release**, not an end-to-end training
release. It does not regenerate source models, raw heatmaps, full-test model
quality, upper-envelope optimization, every exploratory experiment or paper
figures. The core bundle README gives additional interval and adapter scope.
The concept addendum starts from per-query scores for its fast interval checks;
the bridge independently recomputes the fixed-recipient own/donor hits from
the released native peaks and keypoints. Training uncertainty is not included.

## Data, models and antecedents

Only derived numerical evaluation records are distributed here; no raw bird
images or model weights are included. Dataset and model terms remain those of
their respective owners. This repository does not grant rights to third-party
datasets or checkpoints.

- [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [FunnyBirds](https://github.com/visinf/funnybirds)
- [ProtoCBM public scorer](https://github.com/hqhQAQ/ProtoCBM)
- [Loc-CBM](https://github.com/VIPL-VSU/Loc-CBM)

The NumPy predicate is independently implemented for cached peaks; it does not
replace the original method or dataset releases. The paper's public-scorer and
inference checks were performed separately from this portable rescoring run.

## Manuscript and citation

The manuscript is in preparation. A publication DOI and citation will be added
if available. The present repository provides the numerical audit materials;
it does not imply that the manuscript has been accepted or published.
