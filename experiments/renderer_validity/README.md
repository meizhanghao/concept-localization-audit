# Renderer-response criterion check

An exploratory follow-up to the crossed concept-localization audit. This is a
criterion check, not a new attribution method or a semantic-causality claim.

## Reproduce the numerical analysis

Python 3.10+ and NumPy are sufficient; the compressed arrays are included.

```bash
python reproduce.py --out reproduced.json
```

The command verifies the SHA-256 manifest, checks the donor identities and
no-op responses, recomputes spatial scores from peaks and windows, and reruns
the correlation and part-selection analyses. It compares the resulting numbers
with the archived full-precision results. It does not rerender FunnyBirds,
regenerate maps from images, or retrain the encoders.

## Design

The original 200 images are the first four official test images per class.
The complementary 300 images are the remaining six per class; they previously
supplied layout calibration, so this is **not independent confirmation**.
The three checkpoints are the existing ResNet-18 seeds 0/1 and ResNet-50 seed 2.
There are no new training runs or selected best seeds.

For each image and its five concept-versus-alternative queries, we use the
official same-scene image with exactly one part removed. The outcome is the
signed drop of that fixed concept-logit margin. Camera and scene parameters
are held fixed, but deletion can change occlusion and shadows. Original and
five deletions share one forward batch: an exactly unchanged rendering has
exactly unchanged logits. This is a model response to the specified renderer
operation, **not natural-world ACE or a `do(color)` experiment**.

Scores are query-level `s`, `g`, and `delta_g`, at the existing mask-dilation
radius 24. Each explainer has three matched Gaussian heads (seed 821, row norms
preserved). Exact query/alternative donors are deranged inside each cohort;
heads and 200 donor schedules remain fixed under the image bootstrap.

All test parts are annotated present. The visibility sensitivity requires at
least one decoded part pixel at 160x160, restricting recipients only. It does
not equate an empty mask with an absent concept and does not change donors.

## Estimation and limits

- The statistic is the equal-weight mean of **five within-part Spearman
  coefficients**. Bootstrap samples repeat images and keep their queries
  together; weighted midranks preserve ties exactly.
- A constant score has no correlation. We retain that undefined value rather
  than averaging fewer parts. Some CAM draws are degenerate: their reported
  defined-draw intervals are **conditional diagnostics, not unconditional
  95% intervals**. The simultaneous six-comparison claim is withheld.
- `analysis_amendment.json` records this reporting correction after the first
  analysis exposed degeneracy. It did not change cohorts, maps or donors.
- The historical `frontier` checker used `nanmean` over available parts in a
  bootstrap draw. Its old correlation intervals describe that different
  procedure; use this follow-up for the strict five-part statistic.
- Part-selection utility is secondary and only uses R18 seed 0's existing
  frozen class readout. It chooses the largest score, averages exact ties,
  and measures true-class/original-best-rival margin drop. The comparator is
  uniform choice; regret is the gap to the largest observed part-deletion
  response. Concept-map targets and class utility are different targets.
- Results are conditional on these models and reference heads. They neither
  rank explainers universally nor establish S or delta_g as a causal proxy.

## Provenance

`data/inputs.npz` contains integer peaks, packed windows, logits, fixed donors,
bootstrap counts, labels and frozen readout parameters. It contains no raw
images or full encoder weights. `data/protocol.json` records the pre-inference
plan; the inference checks reproduce the historical 200-image peaks exactly.

The executed inference/export sources are preserved in `source/`. They use
the author workspace's existing FunnyBirds caches and checkpoint layout;
they are provenance, not a standalone full-training pipeline. The portable
statistics above are self-contained.

Part-removal testing follows FunnyBirds (Hesse et al., ICCV 2023). The added
question is whether the crossed spatial scores predict the declared renderer
response, and how that answer changes with recipient visibility. No priority
claim is made for deletion tests, correlation analysis or regret.
