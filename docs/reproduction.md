# Reproducing the numerical audits

## Environment

The reference run uses Python 3.10.8 and NumPy 2.2.6 on CPU. Install the pinned
NumPy version with:

```bash
python -m pip install -r requirements-reproduce.txt
```

`requirements.txt` records the supported dependency range. No PyTorch, SciPy,
CUDA or network access is required during scoring. To avoid excessive BLAS
threading, the runner sets the requested thread limit before importing NumPy.

## Commands

Run all analyses:

```bash
python run_audit.py
```

Run one analysis:

```bash
python run_audit.py --audit core
python run_audit.py --audit concepts
python run_audit.py --audit bridge
python run_audit.py --audit extensions
```

Check the input files without computing scores:

```bash
python run_audit.py --verify-only
```

Use a custom output directory or thread limit:

```bash
python run_audit.py --output-dir results/cpu_run --threads 2
```

The script resolves its default `data/` directory relative to the project, so
it can also be invoked by absolute path from another working directory.
`--data-dir` accepts a directory with the same schema and matching
`checksums.json`. Outputs must be outside the input directory.

`python -m audit` is an equivalent entry point when run from the project root.

## Outputs

| File | Contents |
|---|---|
| `results/core.json` | 36 score cases and 12 paired identity contrasts; full-precision S, G, random reference, excess and declared intervals. |
| `results/core_scores.csv` | A convenient tabular view of the core numerical report. |
| `results/concepts.json` | Primary and both-present label contrasts, adjusted contrasts and paired intervals. |
| `results/bridge.json` | Target-only versus both-present donor contrasts and their paired difference; fixed pair/recipient counts. |
| `results/extensions.json` | Reference-family and fixed-support donor-species comparisons. |
| `results/verification.json` | Checks performed in this invocation, maximum errors, Python/NumPy versions and elapsed time. |

An individual audit replaces its own JSON and the invocation's verification
report; other result files in the same output directory are retained. Use a
fresh `--output-dir` to keep separate runs.

Every run first checks all 57 scientific input files against SHA-256
fingerprints. Each analysis then compares computed numbers with the expected
reference fields at absolute tolerance 1e-12. The expected files are regression
targets, not inputs used to set the estimates. Corrupt inputs, missing files,
non-finite outputs and numerical mismatches produce a nonzero exit status.

The frozen donor indices and resampling weights are reused exactly. Timing is
machine-dependent. The input hashes and reported numerical fields, rather
than elapsed time or serialized report bytes, are the reproducibility criteria.

## Control checks

```bash
python -m unittest discover -s tests -v
```

These checks cover a fixed prior's zero image-swap gain, an image-paired
positive control, inclusive/clipped CUB geometry, invalid keypoints and input
integrity failures. The full audit command is the numerical regression check
against the released experimental records.

Seven additional checks exercise the independent [new-input interface](new_inputs.md),
including a hand-computed positive control, mask/keypoint equivalence,
multiple concepts per macro group and rejection of invalid donor/bootstrap
designs. Twelve unit tests run in total. The optional
[raw-image example](regeneration.md) needs separate PyTorch/image dependencies.

## Follow-up commands

The complete numerical runner above reproduces the original four audit modules.
Additional commands reproduce the follow-ups from the same frozen inputs:

```bash
python sweep_cub_geometry.py --output-dir results/geometry
python analyze_cub_robustness.py --input-dir data/core_audit/data --attribute-manifest data/core_audit/attribute_manifest.json --output-dir results/robustness
python audit_released_cem.py --output-dir results/released_cem
```

These compute the four-width sweep, paired/simultaneous width and anatomical
analyses, and the released-CEM crossed audit, respectively. Reports under
`docs/` provide the reference outputs. The CEM's separate
[raw regeneration command](released_cem.md) executes its author checkpoint on
CUB photos, rather than rescoring stored peaks.
