# Audit protocol

## Crossed image/head control

For a recipient image and queried concept, fix its target acceptance region.
Score the peak of its own map and the peak of a donor image's map for the same
concept. Donor peaks are tested against the **recipient's** region; donor
annotations do not replace the target geometry.

- **S** is the macro-aggregated own-map hit rate.
- **G** is the macro-aggregated own hit minus mean donor hit.
- **G_rand** is the mean G of individually scored randomized concept heads,
  keeping the trained backbone, explainer, queries, geometry and donors fixed.
- **ΔG = G − G_rand** is the crossed excess.

Equivalently, with trained/random × own/donor scores, ΔG is
`(trained_own - trained_donor) - (random_own - random_donor)`.
A positive G and a positive own-score trained/random gap do not determine
whether this interaction is resolved. Random-head averaging occurs after peak
extraction; the historical random-gradient map ensemble is a different control.

## Coordinates and acceptance regions

FunnyBirds uses cached `(y, x)` integer peaks and bit-packed binary acceptance
regions. Its mask/point constructions have separate geometry contexts. CUB
uses `(x, y)` peaks in the 224-coordinate frame. The CUB predicate asks whether
any associated visible keypoint lies inside the inclusive peak-centered square
of half-width 45, clipped to `[0, 224]`. Invalid keypoints are `(-1, -1)`.

CUB maps are projected using each source image's own frame, not fitted to the
recipient's annotations. The eight frozen native adapters vary source frame,
interpolation and zero-map fallback. Frames alter physical tolerance and chance,
so adapter differences do not isolate a single causal mechanism.

## Populations and aggregation

Core FunnyBirds scores average within a part and then across five parts.
CUB scores average within an eligible attribute and then across attributes.
The original CUB 40-attribute audit uses 377 images / 1,018 queries. The disjoint
cohort uses 800 images / 8,323 eligible queries over the same 40 attributes.
Its query rule uses all eligible queries rather than the original sampling rule.
The cohorts are not pooled. They share an already explored dataset; the
disjoint cohort is not an external preregistration or an untouched dataset.

The primary same-part label audit uses 32 global assignments on native Loc-CBM
with five randomized reference heads. Only 7.1% of the mismatched concepts in
the primary query population are annotated present. Its positive contrast
therefore does not identify specificity between two known-present attributes.

The both-present analysis retains 36 target attributes, 133 ordered pairs,
776 distinct images and 7,081 pair-image observations. Both target and foil
are positive/certain in every recipient and donor. It averages within pair,
within target attribute, then across target attributes. All other jointly
eligible images are enumerated as donors.

The donor bridge holds those exact recipient records, pairs, maps and macro
weights fixed while changing only donor eligibility: target-positive-only
versus both-present. This is a CUB analysis. It does not decompose the primary
0.120 contrast or estimate the causal effect of concept presence.

## Uncertainty

Core intervals use saved paired recipient-image cluster weights. Where
`joint: true` is declared in the core manifest, reference heads are resampled
globally per draw. Conditional intervals instead retain the reference bank.
Donor schedules and the trained model remain fixed.

The primary label contrast additionally resamples the assignment bank, using
the original assignment-resampling seed. Random-adjusted intervals include
global head resampling. Both-present and bridge intervals condition on the pair
bank and enumerate eligible donors. The same image weights are reused so
contrasts remain paired. No interval here includes training uncertainty.

The eight adapter intervals are exploratory and pointwise, not simultaneous
confirmatory intervals. One same-part permutation is primary for the original
adapted-model identity control; five global controls are also included. This
is distinct from the later 32-assignment native concept-eligibility audit.

## Reading the outputs

A zero-crossing interval is unresolved, not evidence of equivalence. Positive
G or ΔG establishes a spatial correspondence contrast under the declared
protocol, not semantic correctness or causal faithfulness. Co-located true
attributes can legitimately share spatial evidence. The experiment's reference
head bank retains the trained backbone, so it is a matched reference rather
than a universal floor or a causal decomposition.
