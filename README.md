# Auditing Concept Localization

CPU/NumPy audit code for **Auditing Concept Localization: Crossed Controls and
Concept Eligibility**, a manuscript prepared for BlockSys 2027.

Authors: Zhanghao Mei, Zuqiang Meng (corresponding author), Xuwei Tang;
Guangxi University.

## Availability

The portable rescoring source is visible in `code/`. The complete numerical
input bundle has passed local clean-directory verification, but its public
upload is still pending because the current network repeatedly truncates
large uploads. **This repository does not yet provide the complete inputs
required to reproduce the numerical results.** A Release link will be added
only after the full bundle has been uploaded and verified.

The prepared bundle contains 36 score cases, 12 paired identity contrasts,
primary/both-present concept-label contrasts and the CUB fixed-recipient donor
bridge. It uses only NumPy, cached derived records and frozen resampling draws;
it contains no raw images or model weights. It is not a full training release.

Prepared ZIP SHA-256: `7fe51ea1c0760d25888d8eb22e3c10121664c14945ef413f7c0dcd07ae0cbbd8`.

In the complete bundle, run `python -I -B run_all.py` after extraction with
Python 3.10+ and NumPy 1.24+. Cloning the source-only repository is insufficient
until the derived input bundle is available.

The 133-pair/7,081-recipient bridge is on CUB native Loc-CBM, not FunnyBirds.
It isolates donor eligibility on a fixed recipient set, not a decomposition of
the original 0.120 contrast or a causal effect of concept presence. A positive
spatial correspondence score does not certify semantic or causal faithfulness.
