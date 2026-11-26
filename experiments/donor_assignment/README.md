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
changes between queries. A per-query equality would restrict the class.
Zero is unreachable for a positive lower endpoint and is reached at weight
one for a zero lower endpoint, assuming a positive uniform score.

For disjoint CUB Loc-CBM, each of the 40 attribute-level lower endpoints is
negative. The two largest attributes contribute 6.6% of the total
uniform-to-minimum drop. Leaving out any one attribute gives a zero-crossing
weight between 2.4% and 3.7%. The influence command independently recomputes
the attribute optima. The paper's numerical supplement gives the full tables
and witnesses.

Render the English paper figure from the calibrated results:

```bash
python -m pip install matplotlib==3.10.9
python -B -m experiments.donor_assignment.plot_donor_sensitivity --paper
```

The solver certifies each endpoint with a feasible mixture of at most
two permutations per stratum and a supporting line for their annotation
and score totals. Independent exhaustive small-stratum linear programs
check the solver. A six-query example in the output gives identical
uniform scores and annotation-point gain for two hit matrices with
different calibrated ranges.
