# Manuscript figures from cached audit inputs

Replot the manuscript's illustrative CUB and FunnyBirds cases using frozen
model-derived maps and the original dataset images. No generative image model,
inpainting, object removal or manual movement of peaks is involved.

```bash
python experiments/figures/render.py --cub-root /path/to/CUB_200_2011 --funnybirds-zip /path/to/FunnyBirds.zip --out-dir results/four_cells
```

Dependencies: Python 3.10+, NumPy, Pillow and Matplotlib. The checked environment
uses Matplotlib 3.10.9. Obtain the datasets separately under their own terms;
the numerical map file contains no source RGB images. Source-pixel hashes
verify the declared crop/resize operations before rendering.

Each row has six columns: recipient, donor source, trained/own, trained/donor,
random/own and random/donor. All four overlays use the recipient RGB image.
Donor maps retain their source coordinate frame; they are not aligned to the
recipient's keypoints or recomputed on the recipient.

- CUB uses the released Loc-CBM's pre-mask ReLU concept projection. The case is
  `has_breast_pattern::solid`; acceptance uses the official keypoint predicate
  at half-width 45 after the declared own-box projection.
- FunnyBirds uses the trained ResNet-18 seed 0 model's beak contrast, variant 4
  against variant 1. The images are the official dataset's rendered assets.
  Acceptance dilates the recipient's beak mask by Chebyshev radius 24.
- Cyan contours show the **acceptance region**, not an object segmentation or
  a concept ground-truth mask. White crosses are scored argmax positions.
- Each map is min/max scaled separately for display, with the magma colormap
  and opacity `0.65 * scaled_map`. Cross-panel color intensities do not measure
  relative evidence strength. Scores use the frozen peaks, not display colors.

Selection follows stored array order: the first eligible CUB query and the
first FunnyBirds audit image's first nonempty part; donor schedule 0 and
reference head 0. Selection does not inspect maps, hit scores or confidence.

The CUB hits are `[0, 0, 1, 0]`; FunnyBirds hits are `[1, 1, 0, 0]` in four-cell
order. These are individual illustrative observations. They cannot estimate
population `G`, establish a trained/random ranking, or verify the semantic
meaning of breast texture or beak identity.

`source/upgrade_examples.py` records the executed research generator; its
relative imports require the author's full research environment. The portable
`render.py` command instead verifies frozen maps and user-provided RGB images.
Full model map generation uses the research generator.

## Audit overview, scoring geometry, eligible concepts and donor ranges

From the public repository root, render the four additional PDFs with

```bash
python -m experiments.figures.render_extended --cub-root /path/to/CUB_200_2011 --funnybirds-zip /path/to/FunnyBirds.zip --out-dir results/manuscript_figures
```

This command also uses SciPy and checks the cached RGB, map and result hashes.
Pass `--preview-dir results/manuscript_figures/previews` to export PNG previews.

| Output | Inputs |
| --- | --- |
| `fig_framework.pdf` | A donor-map transfer and the frozen CUB cohort's four hit rates and attribute-calibrated donor range. The fixed-schedule excess and the uniform-policy baseline are computed separately. |
| `geometry_scoring_examples.pdf` | The two recipients from the four-cell figure, each with its cached annotation, scoring window and trained/random own-image peaks |
| `concept_eligibility_examples.pdf` | CUB image 28, its two projected Loc-CBM bill-attribute maps and the existing fixed-recipient donor-eligibility comparison |
| `donor_calibration_ranges.pdf` | The four attainable ranges from the common-support calibration analysis on 6,522 queries |

The co-present photo is the first qualifying cached record, using a shared
window, two hits, two available nonzero maps and a minimum peak separation of
16 pixels for readability. It illustrates one binary-score tie. The adjacent
plot summarizes all 7,081 recipient records from 133 ordered pairs. Its
interval belongs to the paired change in the contrast.

`data/extended_provenance.json` records the picture ID, coordinates, attribute
IDs, crop, selection rule and data hashes. The extended NPZ contains only the
scoring annotations and two projected maps used in the figures. The local
author workflow is recorded in `source/prepare_extended_figures.py`.
