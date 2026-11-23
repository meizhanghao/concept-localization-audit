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
The 40 attributes were selected by the original cohort's minimum-10-pairs
rule, then inherited unchanged. Photos, queries, primary adapter, heads and
donor schedules were fixed before this cohort's localization scores; the
attribute selection was not independently replicated. The positive disjoint
result therefore remains exploratory, not independent confirmation.
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

## Window sensitivity

`python sweep_cub_geometry.py` changes only the native CUB acceptance half-width
to 16, 24, 36 and 45. It fixes 800 photos, 8,323 queries, 40 inherited attributes,
the own-box/bilinear/row-major adapter, all heads, donor schedules and image/head
draws. The four results are exploratory pointwise comparisons. At 16 the excess
interval crosses zero; at 24 its lower bound is close to zero. Neither the
largest excess nor significance is used to select a width. The primary `h=45`
result is unchanged. This does not test every adapter or explanation method.

### Paired differences and simultaneous bands

`analyze_cub_robustness.py` reuses the same draws to compare widths directly.
It reports all six wider-minus-narrower differences rather than comparing
whether two separate intervals cross zero. Approximate 95% simultaneous bands
use the common-bootstrap maximum absolute centered deviation, divided by each
contrast's bootstrap standard deviation. The four width estimates and the six
paired differences are separate fixed families. These are exploratory bands,
not selective inference or a finite-sample coverage guarantee. They do not
undo the earlier choice of widths or change the primary `h=45` result.

### Anatomical influence

The same script reports all ten anatomical groups specified by the scorer's
attribute-prefix/keypoint mapping, plus every leave-one-group-out estimate.
The [attribute manifest](../data/core_audit/attribute_manifest.json) fixes this
mapping. Each part retains equal attribute weights; removing a part renormalizes
over the remaining attributes. Group intervals are descriptive and pointwise.
All groups remain in the main estimate. Eye and throat supply about 61% of
the signed macro excess, and omitting eye leaves a zero-crossing interval;
the overall result does not establish robustness for every part.

### Additional author-released CEM

The [CEM follow-up](released_cem.md) fixes the same photos, queries, windows
and donors. Its concept-logit Grad-CAM and five architecture-specific Gaussian
head references are regenerated from the author's checkpoint, without training.
Positive pairing survives, but the joint image/head excess is unresolved.
This is neither proof of model failure nor a cross-model trustworthiness rank.
The raw-image path and full-cohort verification are separate from CPU rescoring.

## Practitioner checklist

1. Declare recipient and concept eligibility, attribute selection and macro weights.
2. Fix the coordinate frame, acceptance window, tie/zero-map rule and application threshold.
3. Specify donor eligibility, exclusions and shared schedules before comparing heads.
4. Match backbone and explainer; describe reference-weight construction and bank size.
5. Report S, G, G_rand and delta_G, chance/prior controls and paired intervals.
6. Report retained resamples, selection/follow-up status and unresolved results; do not infer semantic or causal correctness from spatial correspondence.

The [worked audit card](audit_card.md) applies these rules to the frozen native
Loc-CBM result, with the four primitive scores and the supported claim.

## Threats to validity

| Type | What remains unresolved |
|---|---|
| Construct | A part segmenter may pass without explaining the classifier. No decision-utility or causal criterion is tested; co-present attributes may share spatial evidence. |
| Internal | The random reference retains backbone structure. Species-constrained donors do not match pose/background. Bird-box warping can bias correspondence in either direction. |
| External | Two datasets and 40 inherited CUB attributes limit generalization. A disjoint cohort in an explored dataset is not independent confirmation. The width sweep uses one native adapter. |
| Statistical | Three/five-head banks provide finite support. Intervals omit training uncertainty; exploratory pointwise intervals are not multiplicity-adjusted. Crossing zero establishes neither absence nor equivalence. |
