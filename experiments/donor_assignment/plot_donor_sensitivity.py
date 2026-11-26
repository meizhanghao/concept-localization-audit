"""Render the donor sensitivity figure from the verified numerical output."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
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


def render_chinese(results, output):
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
        bounds.text(-.72, height + .24, label, va="center", color=color, weight="bold")
        bounds.hlines(height, low, high, color=color, linewidth=3, alpha=.70)
        bounds.vlines([low, high], height - .045, height + .045, color=color, linewidth=1.5)
        bounds.scatter(uniform, height, s=64, color=color, edgecolor="white", linewidth=1.2, zorder=3)
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
    figure.text(.085, .055, "圆点为均匀配对得分。横线为满足约束的可达范围。",
                color="#43566a", fontsize=9)
    figure.text(.57, .055, "斜线由可达下界计算。横轴显示 0–50%。",
                color="#43566a", fontsize=9)
    figure.savefig(output / "donor_sensitivity.png", dpi=220)
    plt.close(figure)


def render_paper(results, output):
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
    bounds.set_title("a   Attainable excess", loc="left", fontsize=8.5, weight="bold", pad=4)
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
        bounds.text(-.695, height + .19, label, color=color, va="bottom")
        bounds.hlines(height, low, high, color=color, linewidth=1.5)
        bounds.vlines([low, high], height - .085, height + .085, color=color, linewidth=.8)
        bounds.scatter(uniform, height, s=20, color=color, edgecolor="white", linewidth=.55, zorder=3)
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
    for axis in (bounds, mixture):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.2, pad=2)
    bounds.spines["left"].set_visible(False)
    figure.savefig(output / "donor_sensitivity_paper.pdf", metadata={"CreationDate": None})
    plt.close(figure)


def main():
    default_results = HERE if (HERE / "calibrated_results.json").is_file() else Path("results/donor_assignment")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--paper", action="store_true", help="Render the manuscript's English PDF.")
    args = parser.parse_args()
    results = json.loads((args.results_dir / "calibrated_results.json").read_text(encoding="utf-8"))
    if args.paper:
        render_paper(results, args.results_dir)
    else:
        render_chinese(results, args.results_dir)


if __name__ == "__main__":
    main()
