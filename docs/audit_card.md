# Worked audit card: frozen native Loc-CBM

## Declared setting

- Recipients: 800 CUB photos, 8,323 queries, 40 inherited attributes.
- Scorer: unchanged ProtoCBM keypoint predicate, half-width 45 at 224 pixels.
- Adapter: own bird-box frame, bilinear inverse crop, row-major zero fallback.
- Reference: five row-norm-matched Gaussian concept heads; trained backbone retained.
- Donors: 200 fixed concept-preserving schedules, shared across heads.
- Aggregation: attribute macro mean; paired image/head bootstrap, 3,000 draws.
- Status: exploratory disjoint cohort in an already explored dataset, not independent confirmation.

## Four primitive scores

| Head | Own-image maps | Donor-image maps |
|---|---:|---:|
| Trained | 0.3072109151 | 0.1844692880 |
| Reference, score-averaged | 0.2708187947 | 0.1694590111 |

The own-minus-donor difference is **G = 0.1227416271**. The matched reference
gives **G_rand = 0.1013597836**, leaving **delta_G = 0.0213818435** with a
joint 95% interval **[0.005979, 0.037007]**. Compute differences before rounding.

## Supported judgment

There is a small positive excess correspondence above this declared Gaussian
reference. A spatial-adequacy requirement was not specified, so this card does
not label S as adequate. The reference accounts for most of the raw pairing gain.

## Chance and independently calibrated position prior

Exact uniform-peak chance is 0.1445147333. The pre-existing layout rule,
calibrated on 4,794 disjoint training images, has
**S=0.3596138656 [0.345212,0.373794]** and **G=0 exactly** on the same audit
recipients. No test keypoint selects its peak; test bird boxes still define
the common scoring frame.

Native model agreement minus prior agreement is
**-0.0524029505 [-0.072204,-0.031896]**, with paired image-cluster draws,
conditional on the fixed calibration. Thus the native model has lower agreement
than this layout baseline while retaining a small positive correspondence excess.
The prior is neither a randomized-head reference nor an application-set adequacy
threshold. Adapted CAM and gradients remain above this prior in agreement.

Reproduce the new control with
`python experiments/cub_prior/reproduce.py --out results/cub_prior.json`.
Its [protocol and interpretation](../experiments/cub_prior/) preserve the
exploratory status and the inherited rule.

## Boundaries and next control

The card does not establish concept correctness, causal decision relevance or
robustness for every anatomical group. Omitting eye leaves an unresolved
interval; the complete [anatomical table](sensitivity_tables.md) is retained.
Changing reference families changes the comparison, not the trained model.
Concept-eligibility and donor-rule controls are needed before interpreting a
same-part label contrast as semantic specificity.

Reproduce the primitive scores with `python run_audit.py`, and the width/part
follow-up with the command in [the protocol](protocol.md).
