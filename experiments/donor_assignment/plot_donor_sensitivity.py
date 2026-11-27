"""Render the donor sensitivity figures from the verified numerical output."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
PDF_METADATA = {"CreationDate": None}
PAPER_FONT = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
}
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


def schedule_scores(details, cohort, case):
    archived = details["schedules"][cohort]
    values = np.asarray(archived["delta_G"][case], dtype=float)
    if values.shape != (archived["schedule_count"],) or not values.size or not np.isfinite(values).all():
        raise ValueError(f"Invalid archived schedule scores for {case}.")
    return values


def add_observed_range(axis, details, cohort, case, height, color):
    scores = schedule_scores(details, cohort, case)
    endpoints = [scores.min(), scores.max()]
    axis.hlines(height, *endpoints, color=color, linewidth=2.0)
    axis.vlines(endpoints, height - .055, height + .055, color=color, linewidth=.8)


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
        add_observed_range(bounds, details, cohort, case, height + .21, color)
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
    figure.text(.085, .055, "上方短线为 200 次原配对的极差；菱形与下方长线为均匀值与约束范围。",
                color="#43566a", fontsize=9)
    figure.text(.57, .055, "斜线由可达下界计算。横轴显示 0–50%。",
                color="#43566a", fontsize=9)
    figure.savefig(output / "donor_sensitivity.png", dpi=220)
    plt.close(figure)


def render_paper(results, details, influence, output):
    """Render the compact English vector figure at the LNCS column width."""
    plt.rcParams.update({
        **PAPER_FONT,
        "font.size": 8,
        "axes.edgecolor": "#4b5563",
        "axes.linewidth": .55,
    })
    figure, (bounds, mixture) = plt.subplots(1, 2, figsize=(4.78, 1.50))
    figure.subplots_adjust(left=.036, right=.962, bottom=.245, top=.825, wspace=.34)
    bounds.set_title("(a) Donor score ranges", loc="left", fontsize=8.5, weight="bold", pad=4)
    mixture.set_title("(b) Worst-case mixture", loc="left", fontsize=8.5, weight="bold", pad=4)
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
        add_observed_range(bounds, details, cohort, case, height + .27, color)
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


def render_distributions(results, details, output):
    """Plot the empirical CDF for each case, using every archived score."""
    plt.rcParams.update({**PAPER_FONT, "font.size": 7.3})
    figure, axes = plt.subplots(1, 2, sharey=True, figsize=(4.8, 2.02))
    figure.subplots_adjust(left=.128, right=.965, bottom=.23, top=.87, wspace=.30)
    muted = "#546375"

    csv_path = output / "donor_schedule_scores.csv"
    csv_rows = []
    for axis, (cohort, case, _, color), title in zip(
        axes, CASES, ("(a) FunnyBirds CAM", "(b) Frozen CUB Loc-CBM")
    ):
        scores = schedule_scores(details, cohort, case)
        csv_rows.extend((case, index, score) for index, score in enumerate(scores))
        ordered = np.sort(scores)
        cumulative = np.arange(1, len(scores) + 1) / len(scores)
        q25, q75 = np.quantile(scores, [.25, .75])
        uniform, _, _, _ = case_values(results, cohort, case)
        pad = max(.075 * np.ptp(scores), .0005)
        xmin, xmax = ordered[0] - pad, ordered[-1] + pad
        axis.axvspan(q25, q75, color=color, alpha=.10, linewidth=0)
        axis.step(np.r_[xmin, ordered, xmax], np.r_[0, cumulative, 1], where="post", color=color, linewidth=1.2)
        axis.scatter(ordered, cumulative, color=color, s=6, alpha=.64, linewidth=0, zorder=3)
        axis.axvline(uniform, color=muted, linewidth=.9, linestyle="--")
        axis.set_title(title, loc="left", fontsize=8.2, weight="bold", color=color, pad=5)
        axis.set(xlim=(xmin, xmax), ylim=(-.025, 1.025), xlabel=r"Crossed excess $\Delta G$")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_yticks([0, .25, .50, .75, 1])
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(axis="y", color="#e4e9ed", linewidth=.65)
        clean_axes(axis)
    axes[0].set_ylabel("Cumulative share of schedules")
    figure.savefig(output / "donor_schedule_distribution.pdf", metadata=PDF_METADATA)
    plt.close(figure)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case", "schedule_index", "delta_G"))
        writer.writerows(csv_rows)


def render_supplement(details, influence, output):
    """Render the analytical example and two influence checks."""
    plt.rcParams.update({**PAPER_FONT, "font.size": 7.3})
    ink, muted, gold = "#182939", "#546375", "#dc911e"
    figure = plt.figure(figsize=(4.8, 4.35))
    figure.text(.025, .976, "(a) Same four scores, different policy ranges",
                color=ink, fontsize=8.5, weight="bold")
    example = details["counterexample"]
    row = np.arange(6)
    cmap = ListedColormap(["#ffffff", "#3e7899"])
    matrices = (example["trained_hits"]["stable"], example["trained_hits"]["sensitive"], example["anchor"])
    titles = ("Fixed range [2/3, 2/3]", "Variable range [0, 1]", "Shared annotation hits")
    for index, (matrix, label) in enumerate(zip(matrices, titles)):
        axis = figure.add_axes((.071 + index * .313, .704, .235, .231))
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
        axis.set_title(label, fontsize=7.6, pad=4)
        axis.set_xlabel("Donor query", labelpad=2)
        if index == 0:
            axis.set_ylabel("Target query")
        if index >= 1:
            lower, upper = example["sensitive_lower_then_upper_donors"]
            axis.scatter(lower, row, marker="o", s=36, facecolors="none", edgecolors=gold, linewidth=1.15)
            axis.scatter(upper, row, marker="x", s=24, c="#ad3b27", linewidth=1.0)
    figure.text(.5, .621, r"Baseline $(S,G,G_{\rm rand},\Delta G)=(1,2/3,0,2/3)$ in both cases",
                color=muted, ha="center")

    contributions = sorted(influence["rows"], key=lambda row: row["share_of_total_drop"], reverse=True)
    figure.text(.025, .565, "(b) CUB influence across 40 attributes",
                color=ink, fontsize=8.5, weight="bold")
    left = figure.add_axes((.127, .320, .312, .203))
    right = figure.add_axes((.617, .320, .312, .203))
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
    right.set(xlim=(0, 41.5), ylabel=r"Zero-crossing $\eta_*$", xlabel="Omitted attribute rank")
    right.set_xticks([1, 10, 20, 30, 40])
    right.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    clean_axes(left)
    clean_axes(right)
    species = details["species"]["same_support"]
    figure.text(.025, .227, "(c) Same-species restriction on identical CUB queries",
                color=ink, fontsize=8.5, weight="bold")
    axis = figure.add_axes((.096, .090, .817, .107))
    baseline = species["uniform_excess"]
    axis.axvline(0, color=muted, linewidth=.8, linestyle="--")
    axis.scatter(baseline, 1.16, marker="D", c=gold, edgecolor="white", linewidth=.55, s=38, zorder=4)
    axis.annotate(f"Shared $d_0 = {baseline:.3f}$", (baseline, 1.16), xytext=(8, 0),
                  textcoords="offset points", va="center", color=muted, fontsize=7.3)
    for y, (rule, label) in zip((.74, .22), (("all_species", "Any species"), ("same_species", "Same species"))):
        entry = species["classes"][rule]
        low, high = entry["range"]
        axis.hlines(y, low, high, color=color, linewidth=2.2)
        axis.vlines([low, high], y-.073, y+.073, color=color, linewidth=1)
        axis.text(-.626, y+.095, label, color=ink, fontsize=7.3, va="bottom")
        axis.annotate(f"{entry['positive_baseline_zero_mixing_share']:.1%} to zero", (high, y),
                      xytext=(6, 0), textcoords="offset points", color=color, va="center")
    axis.set(xlim=(-.65, .72), ylim=(-.02, 1.35), yticks=[], xlabel=r"Attainable crossed excess $\Delta G$")
    axis.set_xticks([-.6, -.3, 0, .3, .6])
    clean_axes(axis)
    axis.spines["left"].set_visible(False)
    figure.savefig(output / "donor_diagnostics_supplement.pdf", metadata=PDF_METADATA)
    plt.close(figure)


def main():
    default_results = HERE if (HERE / "calibrated_results.json").is_file() else Path("results/donor_assignment")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--out-dir", type=Path, help="Write plots here, defaulting to the results directory.")
    parser.add_argument("--paper", action="store_true", help="Render the manuscript's English PDF.")
    parser.add_argument("--supplement", action="store_true", help="Render the supplemental PDFs and export plotted scores.")
    args = parser.parse_args()
    output = args.out_dir or args.results_dir
    output.mkdir(parents=True, exist_ok=True)
    results = json.loads((args.results_dir / "calibrated_results.json").read_text(encoding="utf-8"))
    details = json.loads((args.results_dir / "donor_policy_details.json").read_text(encoding="utf-8"))
    influence = json.loads((args.results_dir / "attribute_influence.json").read_text(encoding="utf-8"))
    if args.paper:
        render_paper(results, details, influence, output)
    if args.supplement:
        render_distributions(results, details, output)
        render_supplement(details, influence, output)
    if not (args.paper or args.supplement):
        render_chinese(results, details, output)


if __name__ == "__main__":
    main()
