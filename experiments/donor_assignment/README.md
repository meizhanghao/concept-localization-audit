# Donor assignment sensitivity

This analysis asks how much the crossed excess `ΔG` can vary when the
recipient queries, maps, windows, and trained and random heads are fixed.
It reuses nine cases on FunnyBirds and CUB. Within each part or attribute
stratum, a donor assignment is a permutation with no self-matches. Each
score uses one policy across the trained and reference heads. We keep
expected annotation-point agreement for each stratum at its uniform-donor
value.

From the repository root:

```bash
python -m pip install -r requirements-reproduce.txt
python -m pip install scipy==1.15.3
python -B -m experiments.donor_assignment.donor_assignment
python -B -m experiments.donor_assignment.donor_calibrated
python -B -m experiments.donor_assignment.donor_group_influence
python -B -m experiments.donor_assignment.donor_policy_details
python -B -m experiments.donor_assignment.donor_query_calibration
```

The first command that runs an audit checks the 57 base input hashes,
recomputes the nine published score rows, and records the unconstrained
assignment analysis. The calibrated command computes the attainable
range and includes the policy that minimizes window Jaccard distance
using only window geometry and annotation points. The commands write
protocols, JSON results, and feasible assignment mixtures to
`results/donor_assignment/`. Use `--out-dir` to select another output
directory. CPU computation uses the existing peak, window, and bootstrap
caches.

| Case | Uniform excess | Calibrated range | Mixture share at zero |
|---|---:|---:|---:|
| FunnyBirds CAM | .110 | [−.169, .410] | 39.4% |
| Disjoint CUB, native Loc-CBM | .021 | [−.640, .505] | 3.2% |

The mixture share is the smallest weight on an adverse admissible policy
that can bring uniform excess to zero. The range and mixture share
condition on the cohort, eligibility strata, windows and head bank.
The geometry-based policy is shared by the displays on each cohort. Its
paired pointwise bootstrap intervals condition on that policy and support
lower excess in five cases. All five retain positive excess intervals.
The archived schedules and the uniform policy have slightly different means.
Fixing mean annotation agreement within each stratum permits compensating
changes between queries. Per-query calibration holds each recipient's
expected annotation hit at the baseline value.
Zero is unreachable for a positive lower endpoint and is reached at weight
one for a zero lower endpoint, assuming a positive uniform score.

For disjoint CUB Loc-CBM, each of the 40 attribute-level lower endpoints is
negative. The two largest attributes contribute 6.6% of the total
uniform-to-minimum drop. Leaving out any one attribute gives a zero-crossing
weight between 2.4% and 3.7%. The influence command independently recomputes
the attribute optima. The paper's numerical supplement gives the full tables
and witnesses.

The policy details command recovers the 200 archived schedule scores for
each case and checks their means and extrema. A score is the crossed excess
over all queries with one assignment shared by the trained and reference heads.
The archived assignments use the original policy before annotation calibration.
The query calibration command independently reproduces the paper's
common-support sensitivity table from verified inputs. All four classes use
6,522 Loc-CBM queries with a same-species donor, the same head bank, and the
uniform same-species baseline `d0 = .0407`.

| Donors | Calibration | Attainable `ΔG` | Mixture share at zero |
|---|---|---:|---:|
| All species | Attribute mean | [−.6039, .5253] | 6.3% |
| All species | Per query | [−.5596, .4939] | 6.8% |
| Same species | Attribute mean | [−.0348, .1153] | 53.9% |
| Same species | Per query | [.0080, .0731] | Unreachable |

The command needs no preceding audit outputs. It writes
`results/donor_assignment/query_calibration.json`. The archived output
[query_calibration_results.json](query_calibration_results.json) includes
input hashes and solver checks. The per-query solver checks feasibility and
nesting, and verifies the same-species endpoints with exact enumeration in
2,358 species-by-attribute blocks. These bounds describe the declared policy
classes on this fixed cohort.

Render the paper figure and the detailed supplement from these outputs:

```bash
python -m pip install matplotlib==3.10.9
python -B -m experiments.donor_assignment.plot_donor_sensitivity --paper --supplement
```

Figure 4 compares the observed schedule ranges with the calibrated attainable
ranges. Supplemental Fig. S1 (`donor_schedule_distribution.pdf`) plots the two
empirical cumulative distributions. `donor_schedule_scores.csv` exports the
scores with their schedule indices. `donor_diagnostics_supplement.pdf` shows the
six-query counterexample, the attribute omission results and the species
ranges. Figure 4's CUB cap spans the single-attribute omissions.

The solver certifies each endpoint with a feasible mixture of at most
two permutations per stratum and a supporting line for their annotation
and score totals. Independent exhaustive small-stratum linear programs
check the solver. A six-query example in the output gives identical
uniform scores and annotation-point gain for two hit matrices with
different calibrated ranges.
