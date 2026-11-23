# Audit your own peak maps

This entry point evaluates new inputs. It does **not** load expected-result
files or require the paper's datasets. It implements the paper's paired crossed
audit, rather than a general replacement for [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus).

```bash
python audit_peaks.py --input examples/minimal/peaks.npz --config examples/minimal/config.json --output-dir results/minimal
```

The hand-checkable example has S = G = ΔG = 1 and G_rand = 0. It demonstrates
arithmetic, not statistical power. JSON and Markdown audit cards record the
declared protocol and coverage.

## Input archive

Supply a NumPy archive readable with `allow_pickle=False`:

| Array | Shape | Meaning |
|---|---|---|
| `peaks_xy` | `(1+H, Q, 2)` | Integer `(x,y)` peaks; trained head first, H references next. |
| `image_id` | `(Q,)` | Image IDs defining bootstrap clusters; integers or strings. |
| `concept_id` | `(Q,)` | Exact concept identity to preserve in donor assignments. |
| `group_id` | `(Q,)` | Equally weighted macro groups. Multiple concepts may share a group. |
| `donors` | `(K,Q)` | Query-record indices; every donor has a different image ID and the same concept ID. |
| `points_xy` | `(Q,P,2)` | Keypoint geometry: integer points; `(-1,-1)` denotes missing points. |
| `windows` | `(Q,L,L)` | Mask geometry: binary recipient acceptance regions. |
| `bootstrap_weights` | `(B,N)` | Optional integer cluster multiplicities. Columns follow sorted unique image IDs; each row sums to N. |
| `head_indices` | `(B,H)` | Optional reference resampling indices from 0 to H−1. |

Every `(image_id, concept_id)` must be unique. Peaks must share the declared
frame, grid and tie-breaking rule. The scorer cannot verify map semantics:
concept provenance and eligibility are caller-supplied requirements. Donor
maps retain source coordinates; do not align them to recipient annotations.

```json
{
  "geometry": {"type": "keypoints", "side": 224, "half_width": 45},
  "seed": 123,
  "draws": 3000,
  "s_adequacy_threshold": null,
  "eligibility": "Describe presence, certainty, visibility and selection.",
  "reference": "Describe backbone, head distribution, seeds and map extraction.",
  "donors": "Describe source pool, restrictions and schedules."
}
```

For masks, use `"geometry": {"type": "mask", "side": L}` and supply `windows`.
Keypoint squares have inclusive, clipped endpoints. Exact chance counts the
union of accepted integer peak positions; mask chance is the accepted fraction.
There is no default S adequacy threshold.

Absent supplied weights, image clusters are sampled with replacement. Draws
losing a macro group are rejected; the script reports retention and stops if
fewer than B valid draws occur within 6B proposals. Invalid supplied draws are
rejected, not silently changed. References are sampled globally per draw from
the finite bank. Donors and trained models remain fixed.

Report S, G, G_rand, ΔG and paired intervals together. Positive G can arise
from generic backbone structure. Above-reference evidence requires positive
G and joint image/head ΔG intervals. Unresolved is not equivalence. These
scores do not certify semantics or causal use.

See the [raw-image example](regeneration.md) and the separate frozen regression
entry point `python run_audit.py`.
