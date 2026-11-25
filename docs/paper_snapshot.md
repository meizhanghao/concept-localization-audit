# Base snapshot and later numerical extensions

The original immutable base snapshot is
[`342aeb6cb5af87f37277051d156ec339248cda71`](https://github.com/meizhanghao/concept-localization-audit/tree/342aeb6cb5af87f37277051d156ec339248cda71).
The `paper-v1` tag points to that exact commit. It identifies the review
baseline, rather than a claim that the manuscript has been accepted or frozen
against further author review. The full commit hash is authoritative.

## Included in the paper snapshot

- All 57 fingerprinted numerical inputs.
- The four original audit modules, new-input CLI and 12 unit tests.
- `sweep_cub_geometry.py` and `analyze_cub_robustness.py` for the width,
  simultaneous-interval and anatomical follow-ups.
- `audit_released_cem.py`, `audit/cem_model.py` and `regenerate_cem.py` for the
  author-released CEM's cached audit and raw-image regeneration.
- Full-precision reports, protocols and complete sensitivity tables.

## Added after the base snapshot

Commit [`69037b1`](https://github.com/meizhanghao/concept-localization-audit/commit/69037b1f4d48d6dbb7b10f64e4b3975794935caa)
adds `verify_cub_robustness.py` and `verify_released_cem.py`, their usage
instructions and CI steps. These are portable independent numerical checkers.
They do not add or change scientific inputs, estimates or manuscript claims.
The 12 unit tests already exist in the paper snapshot; they are distinct from
these two later verification commands and from the 12 paired identity
contrasts reported by the core audit.

The later `experiments/frontier/` directory adds exploratory finite-reference
comparisons, a frozen-encoder head-only optimization stress test, and within-part
deletion-response associations. Its 21-file SHA-256 manifest is separate from the 57-file base manifest; none of
the base inputs is replaced. The source modules are preserved for inspection,
and `reproduce.py` independently recomputes the new results from frozen arrays.
Unlike the two earlier verification utilities, this extension adds scientific
analyses. These analyses are exploratory, not preregistered confirmation.

The `paper-v1` tag continues to identify the original base. Do not use that
tag to claim the extension was included. For a manuscript citing a later
commit, check out that full commit to obtain both the base and the extension.

## Reproduce a declared version

```bash
git clone https://github.com/meizhanghao/concept-localization-audit.git
cd concept-localization-audit
git switch --detach 342aeb6cb5af87f37277051d156ec339248cda71
python -m pip install -r requirements-reproduce.txt
python -m unittest discover -s tests -v
python run_audit.py --verify-only
python run_audit.py
```

Use a later revision containing the extension for its portable numerical check:

```bash
python experiments/frontier/reproduce.py --out results/frontier.json
python experiments/renderer_validity/reproduce.py --out results/renderer_validity.json
python experiments/cub_prior/reproduce.py --out results/cub_prior.json
```

`main` can advance; citing it alone does not identify the paper's inputs or
code version. Follow the version-specific README for follow-up commands.
The optimization extension covers one encoder/training seed. Deletion-response endpoints
were examined after the primary optimization outcomes. The association analysis
uses the original R18 seed0 and 200 audit images, averaging five within-part
Spearman coefficients with paired image-cluster intervals. Its historical
bootstrap used `nanmean` over the nonconstant parts available in each draw.
That changes the number of parts in degenerate draws; it must not be described
as an unconditional interval for the fixed five-part mean.

The later `experiments/renderer_validity/` follow-up uses raw logits from
paired original/deletion batches for all three existing FunnyBirds models.
It checks the original 200 images and their complementary 300 images (the
latter had previously supplied geometry calibration). It keeps constant-part
statistics undefined, labels conditional defined-draw bands, and withholds
the simultaneous claim when the declared statistic is undefined. It also
reports visibility sensitivity, explicit part-selection ties and oracle regret.
The manuscript uses this strict follow-up instead of the historical available-part
correlation intervals. It is exploratory, not independent confirmation.
Its manifest is separate from both base and frontier manifests.

The later `experiments/cub_prior/` control reuses an existing training-mean
position rule on the frozen CUB cohort. Its ten centers use 4,794 training
images disjoint from the 800 audit photos. The separate manifest includes
the pre-evaluation protocol, a floating-point diagnostic amendment and
official-scorer checks. The portable command recomputes training centers,
scores and paired intervals from derived numerical arrays; verified frozen
scorer peaks are supplied. It does not regenerate the original datasets.
The manuscript uses this control to contextualize agreement, rather than
claiming a new explainer, a new adequacy threshold or causal grounding.

`experiments/figures/` contains the programmatic Figure 3 renderer and its
provenance manifest. Image files and model checkpoints must be obtained from
their original sources. The current test suite has 20 tests; this later count
must not be attributed to the 12-test base snapshot.

These query-level associations do not compare explainers' universal faithfulness. No result here
certifies semantic or causal grounding or constitutes full training regeneration.
