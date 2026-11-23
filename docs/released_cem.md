# Released CEM reproduction

This follow-up uses the CUB checkpoint linked by the authors of
[Understanding Inter-Concept Relationships in Concept-Based Models (ICML 2024)](https://proceedings.mlr.press/v235/raman24a.html).
The [author repository](https://github.com/naveenr414/Concept-Learning)
provides the [checkpoint](https://drive.google.com/file/d/1ILQkLwDw88bYRV7PnTnQZBcwASk0uA0m/view)
and [experiment configuration](https://drive.google.com/file/d/1rJDESQTqIorr_LANijN0QPAUuSl0gd0n/view).
Source revision: `8509549475caf2be1b057c3c3861e623cc522d5c`.

Checkpoint SHA-256:
`386a366087713da40a56c150c09d9f832a5a05faa748c04cb7490a996934d212`.
Weights and raw CUB photos are obtained from their sources; they are not
redistributed here. The model is loaded without training or interventions.

## CPU numerical audit

```bash
python audit_released_cem.py --output-dir results/released_cem
```

The included peaks use the existing 800-image, 8,323-query, 40-attribute
cohort and donor schedules. The trained row is followed by five reference
heads. The reference heads randomize both concept-context and shared
probability layers, matching each weight-row norm and setting biases to
zero. The ResNet backbone, including its 1,000-output fc, is retained.

The main result is G=0.111643, G_rand=0.072210 and excess=0.039433.
The image-only interval conditions on the five heads: [0.027630,0.051573].
Resampling image clusters and the finite head bank gives
[-0.015305,0.087317]. This is unresolved excess, not proof of absence or
semantic failure. Different architecture/preprocessing/reference mechanisms
preclude a direct quality ranking against native Loc-CBM.

## Raw-image regeneration

Install PyTorch, torchvision and Pillow, in addition to the numerical
requirements. The reference runtime is recorded in the
[eight-image report](released_cem_raw_example.json) and
[full-cohort report](released_cem_raw_full.json).
CUDA is required for the stored Gaussian-bank RNG sequence.

```bash
python regenerate_cem.py --cub-root /path/to/CUB_200_2011 --checkpoint /path/to/ConceptEmbeddingModel_resnet34_fold_1.pt --limit-images 8 --output-dir results/raw_cem
```

Use `--limit-images 800` for the full cohort. The script obtains fixed photo
IDs and concept indices from the included manifest, executes the network,
computes concept-logit Grad-CAM, reconstructs each source crop and bird-box
frame, and compares generated peaks against the cache. It never reads
expected scores to generate peaks.

The reference run reproduced all 49,938 head/query peaks; the eight-image
example checked 330. Near-tie behavior and other GPU/library versions can
affect exact peaks; a mismatch is reported explicitly.

## Model and adapter checks

- Strictly load all 447 checkpoint tensors.
- Reproduce the author's no-intervention forward on eight images: concept
  probabilities and species logits match exactly in the reference runtime.
- Reconstruct the author's 112-column labels from 4,796 author training IDs,
  original 312 instance labels/certainties, training majority, and invisible
  negative exclusion. All 5,794 test vectors match exactly. Continuous CUB
  class-attribute scores are a different label source and are not substituted.
- Follow the author loader: CenterCrop(299), no resize, mean 0.5/std 2.
  Full-test species accuracy is 78.84%; class112 concept accuracy is 96.44%.
- Verify 144 analytic Grad-CAM maps against autograd; all scored peaks agree.
- Check inverse crop origins independently for odd padding and rounding.
- Retain all queries, including 211 trained zero maps with row-major fallback.

Full-precision reports: [protocol](released_cem_protocol.json),
[quality](released_cem_quality.json), [results](released_cem_results.json),
[independent verification](released_cem_verification.json).
