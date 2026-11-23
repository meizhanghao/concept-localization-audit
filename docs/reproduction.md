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
| `results/verification.json` | Checks performed in this invocation, maximum errors, Python/NumPy versions and elapsed time. |

An individual audit replaces its own JSON and the invocation's verification
report; other result files in the same output directory are retained. Use a
fresh `--output-dir` to keep separate runs.

Every run first checks all 51 scientific input files against SHA-256
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
