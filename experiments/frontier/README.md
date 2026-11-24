# Exploratory extensions: numerical supplement

Finite-reference comparisons and a head-only optimization stress test for
**Auditing Concept Localization: Crossed Controls and Concept Eligibility**.

The base paper snapshot is `342aeb6cb5af87f37277051d156ec339248cda71`.
This supplement contains **later exploratory analyses**. It is included in
`experiments/frontier/` in the repository's later revisions; it is not part of
the base snapshot. Consult the repository's version-scope document before
reproducing a cited revision.

## Reproduce the reported numbers

Use Python 3.10+ and NumPy, then run:

```bash
python reproduce.py --out reproduced_frontier.json
```

The script uses frozen windows, peaks, image/head bootstrap indices and
inference outputs. It does not read expected-result JSON files. No dataset,
network access, GPU or model download is needed for this numerical check.
The response calculation fixes the original model's strongest nontrue species
as the rival in both training conditions.

The script verifies the SHA-256 hashes in `manifest.json` before loading inputs.
Full width and anatomical tables moved
out of the main PDF are retained as `width_details.tex` and
`anatomical_details.tex`; the main text keeps their principal paired findings.

## What the tests establish

- Four declared reference banks: three Gaussian direction banks and one
  shared channel-permutation bank; three heads per bank. The methods share
  image weights and head indices within a bank. All paired CAM-minus-gradient
  intervals cross zero. Point sign changes are not resolved reversals.
- The optimization test freezes the trained R18 s0 encoder and continues its
  concept head on 2,000 original training images. The 1,000 validation images
  are disjoint; the 200 audit images never provide training gradients. CE-only
  and regularized conditions start from the same trained head, with one seed,
  ten fixed epochs, Adam at 0.001 and no test-selected epoch or coefficient.
- The regularizer is five times the hinge with margin 0.1 on a differentiable
  spatial-mass surrogate. Native 5-by-5 contrast maps use softmax without ReLU;
  target weights are area fractions of the r=24 dilated masks. Same-concept
  batch donors rotate cyclically. Empty-window and singleton-donor cases are
  omitted from the auxiliary loss. Three fixed Gaussian heads provide the
  training reference. The three evaluation head seeds are 2026112421--23.
- Hard peak excess improves; probability and fixed-readout class-margin
  response increases remain unresolved. Response endpoints were inspected
  **after** primary training outcomes. Conditional intervals omit training
  variation and establish neither equivalence nor semantic causal effects.

## Regeneration scope

`source/` preserves the executed exploratory modules for inspection. In the
original project, run them from `main_python` as modules, for example:

```bash
python -m concept_null_audit.frontier_references
python -m concept_null_audit.frontier_training
python -m concept_null_audit.frontier_training_check
```

Those commands require the original FunnyBirds dataset, prepared feature/RGB
caches, model checkpoint, metadata and base numerical inputs. They are not
a standalone reconstruction of full encoder training. The source bundle
contains no dataset images or pretrained encoder weights. The NumPy script
above is the portable reproduction route for the new table and comparisons.

`data/protocol.json` is the historical gate recorded before the new outcomes.
Its manuscript hash identifies the earlier paper. Regenerating that gate after
editing the manuscript should not overwrite its recorded history.
