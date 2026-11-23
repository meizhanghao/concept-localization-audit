# Numerical inputs

All required numerical inputs are included under `data/`. No original bird
images or model checkpoints are distributed. The `.npz` files are ordinary
compressed NumPy archives and load with `allow_pickle=False`.

## Directory layout

### `core_audit/`

`manifest.json` declares score cases, shared geometry contexts, reference-head
resampling and paired identity comparisons. `expected_results.json` contains
full-precision regression targets. `protocol_summary.json` records cohort and
adapter scope. The nested `data/` directory contains context and peak archives.

| Array | Meaning |
|---|---|
| `peaks` | Per-head integer peak coordinates, trained head first. |
| `donors` | Frozen donor-query indices; columns correspond to recipient queries. |
| `image`, `group` | Recipient-image and macro-group indices per query. |
| `weights` | Saved recipient-image bootstrap multiplicities. |
| `head_indices` | Globally sampled reference-head indices for joint intervals. |
| `points_xy` | CUB target keypoints, with `(-1, -1)` for invalid points. |
| `windows_packed`, `side` | FunnyBirds acceptance regions and grid size. |

Core data contain 36 cases and 12 paired identity contrasts. For example,
`old_CUB_LocCBM`, `new_CUB_native_LocCBM`, `FB_mask_cam` and
`FB_mask_gradient` correspond to the main score families. The `frozen_LocCBM_*`
cases provide the eight adapter sensitivities.

### `concept_eligibility/`

`data.npz` contains derived own/donor query scores and original resampling
draws. Its primary population has 800 image indices, 40 attributes and 8,323
queries. Head axis 0 is trained; the remaining five entries are reference heads.

| Array | Meaning |
|---|---|
| `attribute`, `image` | Primary query attribute/image indices. |
| `weights`, `observed`, `heads` | Paired image weights, group coverage, global head draws. |
| `correct_own`, `correct_donor` | Matched-label own/donor scores. |
| `wrong_own`, `wrong_donor` | Scores for 32 same-part label assignments. |
| `pair_own`, `pair_donor` | Both-present correct/foil scores, with label role on axis 1. |
| `pair_id`, `pair_image`, `pair_weights` | Pair-image record indexing and declared macro weights. |

`native_peak_cache.npz` contains the native peaks for all concepts;
`query_geometry.npz` maps the eligible queries to native attribute IDs,
recipient keypoints and donor indices. `label_assignments.npz` records the
same-part assignment bank. `expected.json` contains the reference contrasts
and intervals.

### `donor_bridge/`

`pairs.json` declares target/foil IDs and eligible query indices for the 133
ordered pairs. `expected.json` records the paired donor-rule comparison.
The bridge loads native peaks and geometry from `concept_eligibility/` and
recomputes hits for both donor rules, rather than substituting expected scores.

## Integrity and provenance

`checksums.json` covers all 51 input files. The runner verifies every SHA-256
fingerprint before analysis. Expected-result files are checked inputs, but
their values are read only for comparison after scoring.

The input arrays were copied byte-for-byte from the verified research caches.
The standalone project reorganizes code and metadata without altering those
arrays, donor schedules, pair eligibility or saved bootstrap draws.

Source projects:

- [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [FunnyBirds](https://github.com/visinf/funnybirds)
- [Loc-CBM](https://github.com/VIPL-VSU/Loc-CBM)
- [ProtoCBM](https://github.com/hqhQAQ/ProtoCBM)

Dataset and model terms remain those of their original providers. The numerical
caches and included paper figures do not provide a license to redistribute the
original images or checkpoints.
