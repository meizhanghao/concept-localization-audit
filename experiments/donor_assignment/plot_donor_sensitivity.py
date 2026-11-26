"""Render the donor sensitivity figure from the verified numerical output."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
PDF_METADATA = {"CreationDate": None}
CASES = (
    ("FB_mask_context", "FB_mask_cam", "FunnyBirds CAM", "#2166ac"),
    ("new_CUB_context", "new_CUB_native_LocCBM", "独立 CUB · Loc-CBM", "#b54831"),
)


def signed(value):
    return f"{value:.3f}".replace("-", "−")


def case_values(results, cohort, case):
    values = results["cohorts"][cohort]["cases"][case]
    uniform = values["uniform"]
    low, high = values["exact_range"]
    crossing = values["fraction_to_zero"]["fraction"]
    assert low <= uniform <= high
    if crossing is not None:
        assert 0 <= crossing <= 1
        np.testing.assert_allclose(uniform + crossing * (low - uniform), 0, atol=1e-12)
    else:
        assert low > 0
    return uniform, low, high, crossing


def add_schedule_strip(axis, details, cohort, case, y, amplitude, color, size):
    values = np.asarray(details["schedules"][cohort]["delta_G"][case])
    if len(values) != 200:
        raise ValueError(f"Expected the 200 archived schedules for {case}.")
    jitter = np.random.default_rng(42).uniform(-amplitude, amplitude, len(values))
    axis.scatter(values, y + jitter, color=color, s=size, alpha=.34, linewidth=0, zorder=2)
    return values


def render_chinese(results, details, output):
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 11,
        "text.color": "#203044",
        "axes.labelcolor": "#203044",
        "axes.edgecolor": "#8793a0",
        "xtick.color": "#43566a",
        "ytick.color": "#43566a",
        "savefig.facecolor": "white",
    })
    figure, (bounds, mixture) = plt.subplots(
        1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1, 1.08]}
    )
    figure.subplots_adjust(left=.085, right=.975, bottom=.21, top=.78, wspace=.30)
    figure.text(.085, .93, "合格供体的重新配对能把超额改变多少？",
                fontsize=18, weight="bold")
    figure.text(.085, .865, "图像、热图、概念头和窗口固定。每组的标注点正对照保持原增益。",
                color="#43566a", fontsize=10.5)

    bounds.set_title("A   固定队列上的可达范围", loc="left", pad=13, fontsize=12, weight="bold")
    bounds.axvspan(-.76, 0, color="#f2f4f7", zorder=0)
    bounds.axvline(0, color="#6f7d8a", linestyle="--", linewidth=1)
    bounds.set(xlim=(-.76, .65), ylim=(-.4, 1.43), yticks=[],
               xlabel=r"交叉超额 $\Delta G$")
    bounds.set_xticks([-.6, -.4, -.2, 0, .2, .4, .6])

    mixture.set_title("B   最不利策略的混合比例", loc="left", pad=13, fontsize=12, weight="bold")
    mixture.axhline(0, color="#6f7d8a", linestyle="--", linewidth=1)
    mixture.set(xlim=(0, .50), ylim=(-.345, .145),
                xlabel="混入最不利合格策略的比例",
                ylabel=r"可达最低 $\Delta G$")
    mixture.set_xticks([0, .1, .2, .3, .4, .5])
    mixture.set_yticks([-.3, -.2, -.1, 0, .1])
    mixture.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    mixture.grid(axis="y", color="#e7ebf0", linewidth=.8)

    shares = np.linspace(0, .5, 151)
    for row, (cohort, case, label, color) in enumerate(CASES):
        uniform, low, high, crossing = case_values(results, cohort, case)
        height = 1 - row
        bounds.text(-.72, height + .21, label, va="center", color=color, weight="bold")
        bounds.hlines(height, low, high, color=color, linewidth=3, alpha=.70)
        bounds.vlines([low, high], height - .045, height + .045, color=color, linewidth=1.5)
        add_schedule_strip(bounds, details, cohort, case, height + .21, .09, color, 11)
        bounds.scatter(uniform, height, marker="D", s=55, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        for position in (low, high):
            bounds.text(position, height - .15, signed(position), ha="center",
                        va="top", fontsize=9, color=color)
        bounds.text(uniform, height - .15, signed(uniform), ha="center",
                    va="top", fontsize=9, color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.1})

        mixture.plot(shares, uniform + shares * (low - uniform), color=color, linewidth=2)
        if crossing is None:
            mixture.text(.98, .95 - .11 * row, "零不可达", transform=mixture.transAxes,
                         ha="right", va="top", color=color)
        elif crossing <= shares[-1]:
            mixture.scatter(crossing, 0, s=53, color=color, edgecolor="white", linewidth=1, zorder=3)
            mixture.annotate(
                f"{crossing:.1%}", xy=(crossing, 0),
                xytext=(10 if row == 1 else 0, 12), textcoords="offset points",
                ha="left" if row == 1 else "center", color=color, weight="bold",
            )

    for axis in (bounds, mixture):
        axis.spines[["top", "right"]].set_visible(False)
    bounds.spines["left"].set_visible(False)
    figure.text(.085, .055, "小点为 200 次原配对；菱形为均匀配对；横线为约束范围。",
                color="#43566a", fontsize=9)
    figure.text(.57, .055, "斜线由可达下界计算。横轴显示 0–50%。",
                color="#43566a", fontsize=9)
    figure.savefig(output / "donor_sensitivity.png", dpi=220)
    plt.close(figure)


def render_paper(results, details, influence, output):
    """Render the compact English vector figure at the LNCS column width."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.edgecolor": "#4b5563",
        "axes.linewidth": .55,
        "pdf.fonttype": 42,
    })
    figure, (bounds, mixture) = plt.subplots(1, 2, figsize=(4.78, 1.50))
    figure.subplots_adjust(left=.036, right=.962, bottom=.245, top=.825, wspace=.34)
    bounds.set_title("a   Schedules and range", loc="left", fontsize=8.5, weight="bold", pad=4)
    mixture.set_title("b   Worst-case mixture", loc="left", fontsize=8.5, weight="bold", pad=4)
    bounds.set(xlim=(-.73, .61), ylim=(-.52, 1.53), yticks=[],
               xlabel=r"Crossed excess $\Delta G$")
    bounds.set_xticks([-.6, -.3, 0, .3, .6])
    mixture.set(xlim=(0, .50), ylim=(-.335, .138),
                xlabel=r"Adverse policy weight $\eta$", ylabel=r"Minimum $\Delta G$")
    mixture.set_xticks([0, .1, .2, .3, .4, .5])
    mixture.set_yticks([-.3, -.15, 0, .1], labels=["−.30", "−.15", "0", ".10"])
    mixture.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    bounds.axvline(0, color="#7b8793", linestyle="--", linewidth=.7)
    mixture.axhline(0, color="#7b8793", linestyle="--", linewidth=.7)
    shares = np.linspace(0, .5, 101)
    for row, ((cohort, case, _, color), label) in enumerate(
        zip(CASES, ("FunnyBirds CAM", "CUB Loc-CBM"))
    ):
        uniform, low, high, crossing = case_values(results, cohort, case)
        height = 1 - row
        bounds.text(-.695, height + .17, label, color=color, va="bottom")
        bounds.hlines(height, low, high, color=color, linewidth=1.5)
        bounds.vlines([low, high], height - .075, height + .075, color=color, linewidth=.8)
        add_schedule_strip(bounds, details, cohort, case, height + .28, .12, color, 4)
        bounds.scatter(uniform, height, marker="D", s=19, color=color, edgecolor="white", linewidth=.55, zorder=3)
        mixture.plot(shares, uniform + shares * (low - uniform), color=color,
                     linewidth=1.35, linestyle="-" if row == 0 else "--")
        if crossing is None:
            mixture.text(.99, .98 - .16 * row, "Zero unreachable", color=color,
                         ha="right", va="top", transform=mixture.transAxes)
        elif crossing <= shares[-1]:
            mixture.scatter(crossing, 0, s=16, color=color, edgecolor="white", linewidth=.55, zorder=3)
            mixture.annotate(
                f"{crossing:.1%}", (crossing, 0),
                xytext=(5, -11) if row == 1 else (0, 5), textcoords="offset points",
                ha="left" if row == 1 else "center",
                va="center" if row == 1 else "bottom", color=color,
            )
    cut_low, cut_high = influence["leave_attributes_out"]["1"]["mixture_share_range"]
    mixture.hlines(0, cut_low, cut_high, color="#b54831", linewidth=1.0, zorder=4)
    mixture.vlines([cut_low, cut_high], -.011, .011, color="#b54831", linewidth=.8, zorder=4)
    for axis in (bounds, mixture):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.2, pad=2)
    bounds.spines["left"].set_visible(False)
    figure.savefig(output / "donor_sensitivity_paper.pdf", metadata=PDF_METADATA)
    plt.close(figure)


def clean_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, pad=3)


def render_supplement(results, details, influence, output):
    """Render the archived scores, analytical example and two influence checks."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    ink, muted, gold = "#182939", "#546375", "#dc911e"
    figure = plt.figure(figsize=(8.3, 10.25), layout=None)
    grid = figure.add_gridspec(
        4, 6, left=.115, right=.95, top=.865, bottom=.074,
        height_ratios=(.83, 1.51, .91, 1.27), hspace=.82, wspace=1.33,
    )
    figure.text(.115, .96, "Where donor sensitivity comes from", color=ink, fontsize=20, weight="bold")
    figure.text(.115, .93, "Peak-in-window audit · fixed cached maps, labels and head banks", color=muted, fontsize=10)
    figure.text(.115, .9, "a   Archived schedules before annotation calibration", color=ink, fontsize=11, weight="bold")

    for index, (cohort, case, _, color) in enumerate(CASES):
        axis = figure.add_subplot(grid[0, index * 3:(index+1)*3])
        uniform, _, _, _ = case_values(results, cohort, case)
        scores = add_schedule_strip(axis, details, cohort, case, .48, .24, color, 15)
        box = axis.boxplot([scores], positions=[-.18], widths=.22, vert=False, showfliers=False,
                           whis=(0, 100), patch_artist=True, manage_ticks=False,
                           boxprops={"facecolor": "#edf1f5", "edgecolor": color},
                           medianprops={"color": color, "linewidth": 1.5},
                           whiskerprops={"color": color}, capprops={"color": color})
        box["boxes"][0].set_zorder(2)
        axis.axvline(uniform, color=color, linestyle=":", linewidth=.8)
        axis.scatter(uniform, -.18, marker="D", color=color, edgecolor="white", linewidth=.5, s=34, zorder=4)
        label = ("FunnyBirds CAM", "Disjoint CUB Loc-CBM")[index]
        axis.set_title(f"{label} · $d_0$ = {uniform:.3f}", loc="left", color=color, fontsize=10, pad=2)
        axis.set(ylim=(-.4, .88), yticks=[])
        clean_axes(axis)
        axis.spines["left"].set_visible(False)

    example = details["counterexample"]
    row = np.arange(6)
    cmap = ListedColormap(["#ffffff", "#3e7899"])
    matrices = (example["trained_hits"]["stable"], example["trained_hits"]["sensitive"], example["anchor"])
    titles = ("Fixed range [2/3, 2/3]", "Variable range [0, 1]", "Shared annotation hits")
    matrix_axes = []
    for index, (matrix, label) in enumerate(zip(matrices, titles)):
        axis = figure.add_subplot(grid[1, index*2:(index+1)*2])
        matrix_axes.append(axis)
        display = np.array(matrix, dtype=float)
        display[row, row] = np.nan
        axis.imshow(display, vmin=0, vmax=1, cmap=cmap.with_extremes(bad="#d9dee3"),
                    origin="upper", interpolation="nearest")
        axis.set_xticks(row, row+1)
        axis.set_yticks(row, row+1)
        axis.set_xticks(np.arange(-.5, 6, 1), minor=True)
        axis.set_yticks(np.arange(-.5, 6, 1), minor=True)
        axis.grid(which="minor", color="#bfcbd4", linewidth=.55)
        axis.tick_params(which="both", length=0, pad=2)
        axis.set_title(label, fontsize=9.2, pad=5)
        axis.set_xlabel("Donor query", labelpad=3)
        if index == 0:
            axis.set_ylabel("Target query")
        if index >= 1:
            lower, upper = example["sensitive_lower_then_upper_donors"]
            axis.scatter(lower, row, marker="o", s=36, facecolors="none", edgecolors=gold, linewidth=1.15)
            axis.scatter(upper, row, marker="x", s=24, c="#ad3b27", linewidth=1.0)
    matrix_top = max(axis.get_position().y1 for axis in matrix_axes)
    figure.text(.115, matrix_top+.048, "b   Identical summary scores can hide different assignment ranges",
                color=ink, fontsize=11, weight="bold")
    figure.text(.115, matrix_top+.028, r"Both have $(S,G,G_{\rm rand},\Delta G)=(1,2/3,0,2/3)$ and annotation gain $5/6$.",
                color=muted, fontsize=9)
    matrix_bottom = min(axis.get_position().y0 for axis in matrix_axes)
    figure.text(.115, matrix_bottom-.055, "Blue = hit, white = miss, gray = own query (ineligible as donor).",
                color=muted, fontsize=8.2)
    figure.text(.115, matrix_bottom-.071,
                "Circle and cross assignments both deliver one annotation hit. They give 6 and 0 trained hits, respectively.",
                color=muted, fontsize=8.2)

    contributions = sorted(influence["rows"], key=lambda row: row["share_of_total_drop"], reverse=True)
    left = figure.add_subplot(grid[2, :3])
    right = figure.add_subplot(grid[2, 3:])
    left.set_title("c   CUB influence across 40 attributes", loc="left", color=ink, fontsize=11, weight="bold", pad=7)
    rank = np.arange(1, len(contributions)+1)
    color = CASES[1][3]
    left.bar(rank, [r["share_of_total_drop"] for r in contributions], color=color, width=.79, alpha=.83)
    left.axhline(1 / len(rank), linewidth=.75, linestyle="--", color=muted)
    left.set(ylim=(0, .036), xlim=(0, 41.5), ylabel="Share of total drop", xlabel="Attribute rank by share")
    left.set_xticks([1, 10, 20, 30, 40])
    left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    one_out = {r["excluded_group_ids"][0]: r["minimum_mixture_share"]
               for r in influence["leave_attributes_out"]["1"]["rows"]}
    right.axhline(influence["minimum_mixture_share"], color=muted, linewidth=.8, linestyle="--")
    right.scatter(rank, [one_out[r["group_id"]] for r in contributions], s=16, c=color, linewidth=0)
    right.set_title("Omit one attribute at a time", loc="left", color=muted, fontsize=9.3, pad=7)
    right.set(xlim=(0, 41.5), ylabel=r"Zero-crossing $\eta_*$", xlabel="Rank of omitted attribute")
    right.set_xticks([1, 10, 20, 30, 40])
    right.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    clean_axes(left)
    clean_axes(right)
    figure.text(.115, left.get_position().y0-.052, "Dashed guides show an equal 2.5% share and the full cohort's threshold.",
                color=muted, fontsize=8.2)

    species = details["species"]["same_support"]
    axis = figure.add_subplot(grid[3, :])
    axis.set_title("d   Same-species restriction on identical CUB queries", loc="left", color=ink,
                   fontsize=11, weight="bold", pad=15)
    baseline = species["uniform_excess"]
    axis.axvline(0, color=muted, linewidth=.8, linestyle="--")
    axis.scatter(baseline, 1.16, marker="D", c=gold, edgecolor="white", linewidth=.55, s=38, zorder=4)
    axis.annotate(f"Shared $d_0 = {baseline:.3f}$", (baseline, 1.16), xytext=(8, 0),
                  textcoords="offset points", va="center", color=muted, fontsize=8.7)
    for y, (rule, label) in zip((.74, .22), (("all_species", "Any species"), ("same_species", "Same species"))):
        entry = species["classes"][rule]
        low, high = entry["range"]
        axis.hlines(y, low, high, color=color, linewidth=2.2)
        axis.vlines([low, high], y-.073, y+.073, color=color, linewidth=1)
        axis.text(-.626, y+.095, label, color=ink, fontsize=9, va="bottom")
        axis.annotate(f"{entry['positive_baseline_zero_mixing_share']:.1%} to zero", (high, y),
                      xytext=(6, 0), textcoords="offset points", color=color, va="center")
    axis.set(xlim=(-.65, .72), ylim=(-.02, 1.35), yticks=[], xlabel=r"Attainable crossed excess $\Delta G$")
    axis.set_xticks([-.6, -.3, 0, .3, .6])
    clean_axes(axis)
    axis.spines["left"].set_visible(False)
    figure.text(.115, .025, f"Both ranges use {species['queries']:,} queries and the same annotation-point target per attribute.",
                color=muted, fontsize=8.5)
    figure.savefig(output / "donor_diagnostics_supplement.pdf", metadata=PDF_METADATA)
    plt.close(figure)


def main():
    default_results = HERE if (HERE / "calibrated_results.json").is_file() else Path("results/donor_assignment")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--paper", action="store_true", help="Render the manuscript's English PDF.")
    parser.add_argument("--supplement", action="store_true", help="Render the detailed supplementary PDF.")
    args = parser.parse_args()
    results = json.loads((args.results_dir / "calibrated_results.json").read_text(encoding="utf-8"))
    details = json.loads((args.results_dir / "donor_policy_details.json").read_text(encoding="utf-8"))
    influence = json.loads((args.results_dir / "attribute_influence.json").read_text(encoding="utf-8"))
    if args.paper:
        render_paper(results, details, influence, args.results_dir)
    if args.supplement:
        render_supplement(results, details, influence, args.results_dir)
    if not (args.paper or args.supplement):
        render_chinese(results, details, args.results_dir)


if __name__ == "__main__":
    main()
