# Reference and donor sensitivity

Exploratory follow-ups, not new randomization or image-swap methods. Run
`python run_audit.py --audit extensions`. Previous input files are unchanged.

## Reference family

On the same 800-image/8,323-query Loc-CBM cohort, compare five original Gaussian
heads with five shared feature-channel permutations W′ = WP. A common column
permutation preserves row entries/norms and WWᵀ while changing orientation to
features. Channels need not be exchangeable; this is not a calibrated null.

Permutation seeds: 2026112301–2026112305; head resampling: 2026112306. Queries,
own-box/bilinear/row-major adapter, 200 donors and image weights stay fixed.
Reference scores are averaged after peak extraction; no family is selected.

| Reference | G_rand | ΔG | Joint 95% interval |
|---|---:|---:|---|
| Original Gaussian | 0.101360 | 0.021382 | [0.005979, 0.037007] |
| Shared channel permutation | −0.025958 | 0.148700 | [0.135483, 0.162183] |

Trained G stays 0.122742. Direction survives both, but magnitude depends strongly
on reference family. Gaussian remains the primary, more stringent comparator
here. Paired excess difference: 0.127318 [0.116776, 0.139030].

Zero-query counts (trained, then five references) are [68,2,3,3,0,1] for
Gaussian and [68,51,63,61,63,54] for channel permutation. Every zero map uses
the same uniform/row-major fallback; no query is removed. The families do not
match every induced map distribution, even though the permutation preserves
the stated weight statistics.

## Donor species

Common recipient support: 6,522 queries on all 800 images, 200 species and 40
attributes, each with another same-species/exact-concept image. The donor source
pool is the full frozen cohort. Enumerate all legal donors, rather than the
original 200 schedules. Both conditions exclude recipient image and preserve
concept; one adds equal species. Maps, windows and macro weights stay fixed.

| Donor rule | G | G_rand | ΔG | Joint 95% interval |
|---|---:|---:|---:|---|
| Unrestricted species | 0.153590 | 0.101496 | 0.052094 | [0.033635, 0.070638] |
| Same species | 0.137880 | 0.097209 | 0.040672 | [0.019647, 0.061894] |

S is 0.338205 under both. Paired ΔG change: −0.011422 [−0.023647, 0.000183],
unresolved. Same-species donors: 1–3/query; unrestricted: 80–659/query. All
3,000 image draws retain all40 groups. These fixed-support estimates are not
pooled with full-query/200-schedule estimates. Same species does not match pose,
geometry or background, and does not remove every possible confounder.
