# Training-only CUB position prior

This follow-up completes the position-prior context of the frozen CUB audit
card. It reuses the existing training-mean layout rule, rather than proposing
a new explainer or tuning a new prior on test outcomes.

```bash
python experiments/cub_prior/reproduce.py --out results/cub_prior.json
```

Use Python 3.10+ and NumPy. The command verifies a separate SHA-256 manifest,
recomputes all ten training means and the prior/model score contrasts, then
compares the new computation with the archived results. It does not read
expected outcomes when calculating scores or intervals. The original 57
baseline inputs are unchanged.

Calibration uses the existing **4,794 training images** (split 0), whose image
identifiers do not overlap the 800 audit recipients. For each declared part,
the fixed center is the mean of its visible training keypoint coordinates in
the annotated bird-box frame. Bilateral landmarks are point-weighted, as in
the earlier experiment. No test keypoint, class, concept label or confidence
is used to choose a peak. Test bounding boxes still define the shared scoring
frame; this is not an annotation-free image-space method.

The research code renders each center as the pre-existing Gaussian on a
56-pixel grid (sigma .035), then uses the unchanged scorer's bicubic
224-pixel peak extraction. The portable bundle uses those verified frozen
peaks. Source annotation and checkpoint files are not required for numerical
reproduction; no RGB images are included.

## Interpretation

On the unchanged 800-photo / 8,323-query / 40-attribute cohort at half-width 45,
the prior has `S=0.3596 [0.3452,0.3738]` and `G=0` exactly. Native Loc-CBM
has `S=0.3072`; its paired agreement minus prior is
`-0.0524 [-0.0722,-0.0319]`. Exact uniform-peak chance is 0.1445.

The adapted CAM and gradient displays remain above the prior in agreement,
by 0.0243 and 0.1013 respectively. These secondary contrasts have pointwise,
not multiplicity-adjusted, intervals. The prior does not dominate every model.

Loc-CBM can be below a layout prior on agreement while retaining a small
positive trained-minus-random correspondence contrast. The two axes answer
different questions. A prior is not a random-head reference, and its
agreement is neither a new adequacy threshold nor a semantic/causal certificate.

All intervals reuse 3,000 image-cluster draws and condition on the fixed
training calibration. This is an exploratory check on a known cohort.

## Files and source scope

- `data/inputs.npz`: derived training coordinates/visibility and image IDs,
  verified fixed peaks, recipient keypoints, existing donor schedules,
  bootstrap weights and model hit vectors.
- `data/protocol.json`: rule and inclusion criterion frozen before new scores.
- `data/implementation_amendment.json`: a documented floating-point diagnostic
  tolerance change after an assertion, before any result table was produced;
  the chosen historical centers, maps and peaks were not changed.
- `data/verification.json`: official scorer, split separation and independent
  calibration checks in the full research environment.
- `source/cub_position_prior.py`: executed research source; relative imports
  require the author's full experiment environment.
- `cub_prior_statistics.py` and `reproduce.py`: portable numerical analysis.
