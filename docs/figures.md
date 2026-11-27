# Manuscript figures

The figures use programmatic diagrams or numerical rendering. Dataset images
and model maps are not produced by a generative image model.

| Figure | Rendering | Evidence and scope |
| --- | --- | --- |
| Audit overview | Pillow and Matplotlib | Shows one CUB donor-map transfer alongside cohort-level four-cell scores and the attainable donor range. |
| Scoring geometry | Pillow and Matplotlib | Shows the annotations, acceptance regions and scored peaks for the CUB and FunnyBirds examples. |
| Diagnostic plots | Matplotlib | Plots spatial agreement, image-pairing gains and conditional reference intervals from archived audit results. |
| Four-cell examples | Pillow and Matplotlib, with cached PyTorch maps | Shows own/donor × trained/random hits for one CUB query and one FunnyBirds query. |
| Eligibility and calibrated ranges | Pillow and Matplotlib | Illustrates a co-present pair and visualizes the fixed-recipient bridge and the four calibrated policy ranges. |

## Replot the four-cell examples

The [figure bundle](../experiments/figures/) includes frozen model maps, scored
peaks, acceptance regions, original image identifiers, pixel checksums and the
executed research generator. The portable replot command reads separately
obtained source datasets; no source RGB arrays are bundled.

```bash
python experiments/figures/render.py --cub-root /path/to/CUB_200_2011 --funnybirds-zip /path/to/FunnyBirds.zip --out-dir results/four_cells
```

Read the bundle README for preprocessing and scope. This command regenerates
the **display from frozen maps**; it does not rerun model inference or training.
The author-side provenance audit additionally checks both datasets' example
maps against fresh checkpoint inference and matches all eight plotted peaks.

All four map panels overlay the **recipient** image. A donor panel deliberately
uses the donor's map without recipient-keypoint alignment. Cyan contours show
the scoring acceptance region, not a semantic attribute mask. Colors use
independent min/max scaling per map, so brightness is not comparable across
panels. The illustrated first-in-array cases are not selected for high scores.

The CUB row uses the released Loc-CBM, whereas the FunnyBirds row uses the
trained ResNet-18 seed 0 concept model. Neither row proves that a heatmap has
identified the semantic content of the named concept.
