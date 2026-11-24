# Paper snapshot and later verification utilities

The manuscript identifies the immutable scientific snapshot
[`342aeb6cb5af87f37277051d156ec339248cda71`](https://github.com/meizhanghao/concept-localization-audit/tree/342aeb6cb5af87f37277051d156ec339248cda71).
The `paper-v1` tag points to that exact commit. It identifies the review
baseline, rather than a claim that the manuscript has been accepted or frozen
against further author review. The full commit hash is authoritative.

## Included in the paper snapshot

- All 57 fingerprinted numerical inputs.
- The four original audit modules, new-input CLI and 12 unit tests.
- `sweep_cub_geometry.py` and `analyze_cub_robustness.py` for the width,
  simultaneous-interval and anatomical follow-ups.
- `audit_released_cem.py`, `audit/cem_model.py` and `regenerate_cem.py` for the
  author-released CEM's cached audit and raw-image regeneration.
- Full-precision reports, protocols and complete sensitivity tables.

## Added after the paper snapshot

Commit [`69037b1`](https://github.com/meizhanghao/concept-localization-audit/commit/69037b1f4d48d6dbb7b10f64e4b3975794935caa)
adds `verify_cub_robustness.py` and `verify_released_cem.py`, their usage
instructions and CI steps. These are portable independent numerical checkers.
They do not add or change scientific inputs, estimates or manuscript claims.
The 12 unit tests already exist in the paper snapshot; they are distinct from
these two later verification commands and from the 12 paired identity
contrasts reported by the core audit.

## Reproduce a declared version

```bash
git clone https://github.com/meizhanghao/concept-localization-audit.git
cd concept-localization-audit
git switch --detach 342aeb6cb5af87f37277051d156ec339248cda71
python -m pip install -r requirements-reproduce.txt
python -m unittest discover -s tests -v
python run_audit.py --verify-only
python run_audit.py
```

Use the later commit or current `main` for the two portable verification
commands. `main` can advance; citing it alone does not identify the paper's
inputs or code version. Follow the version-specific README for follow-up
commands. No result here certifies semantic or causal grounding.
