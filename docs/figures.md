# Manuscript figures

The figures use programmatic diagrams or numerical rendering. Dataset images
and model maps are not produced by a generative image model.

| Figure | Rendering | Evidence and scope |
| --- | --- | --- |
| 1: crossed workflow | LaTeX/TikZ | A schematic of the declared audit. Boxes and arrows describe the procedure; they are not measured model evidence. |
| 2: diagnostic plots | Matplotlib, with the top two panels retained from a three-panel PDF | Plots fixed audit JSON results, including spatial agreement, image-pairing gains and conditional reference intervals. Paired excess intervals determine the diagnostic conclusion. |
| 3: four-cell examples | Pillow and Matplotlib; maps originally computed with PyTorch | Uses CUB photographs and the official FunnyBirds rendered dataset. Shows individual own/donor × trained/random hit predicates, not population gain estimates. |

## Replot Figure 3

The [figure bundle](../experiments/figures/) includes frozen model maps, scored
peaks, acceptance regions, original image identifiers, pixel checksums and the
executed research generator. The portable replot command reads separately
obtained source datasets; no source RGB arrays are bundled.

```bash
python experiments/figures/render.py --cub-root /path/to/CUB_200_2011 --funnybirds-zip /path/to/FunnyBirds.zip --out-dir results/figure3
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
