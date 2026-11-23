# Small Loc-CBM raw-image regeneration example

Start with user-owned CUB JPEGs and the author-released checkpoint. This example
regenerates native **pre-mask ReLU concept maps**, own-box projections and scored
peaks for eight fixed records and six heads. It does not reproduce training,
masked concept scores, species logits or every map in the paper. The full
numerical audit works without these downloads.

Obtain CUB from its [provider](https://www.vision.caltech.edu/datasets/cub_200_2011/)
and `cub_rn50_off.pth` from the [official Loc-CBM release](https://github.com/VIPL-VSU/Loc-CBM/releases/tag/v1-download).
Dataset and model terms remain those of their providers; neither is included.

```bash
python -m pip install torch torchvision Pillow numpy
python regenerate_loccbm.py --cub-root /path/to/CUB_200_2011 --checkpoint /path/to/cub_rn50_off.pth --device cuda --output-dir results/raw_example
python audit_peaks.py --input results/raw_example/peaks.npz --config results/raw_example/config.json --output-dir results/raw_example/audit
```

Use `--device cpu` if necessary. Verified environment: Python 3.10.8, torch
2.13.0+cu132, torchvision 0.28.0+cu132, NumPy 2.2.6. Other devices/libraries can
move near-tied bicubic maxima. The script reports mismatches and map errors;
it never substitutes cached peaks for regenerated ones.

Checkpoint SHA-256:

```text
69cf5a3d0b14d4d589099c641e99596c58940692ab856cd1c027e46c62e9e22a
```

`examples/loccbm/manifest.json` records selection, image paths, integer boxes
and keypoints: the first eight eligible records of attribute54 in saved order,
without map/score filtering. `reference_rows.npz` contains the five original
Gaussian projection rows, avoiding a cross-device RNG equivalence claim.
`verification.npz` contains the six-head native maps and peaks.

The implementation strictly loads the released ResNet-50 backbone state with
torchvision primitives, then applies Resize512/CenterCrop448/ImageNet
normalization and the released concept projection. The audited map precedes
the Gaussian mask. Inverse own-box projection is bilinear with zero padding,
followed by bicubic224 and uniform/row-major zero-map fallback.

The example audit uses one cyclic donor schedule among eight records. It is
not a replacement for full 800-image/8,323-query estimates with 200 schedules.
Its interval is a pipeline check, not a population finding.
