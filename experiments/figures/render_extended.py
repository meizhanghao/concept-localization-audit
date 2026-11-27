"""Render the audit overview, scoring geometry, co-present case and donor ranges."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from .render import BASE, PAPER_FONT, load_images, sha256


CYAN = "#00d4e8"
GOLD = "#f5ce65"
BLUE = "#2166ac"
RUST = "#b54831"
INK = "#27323f"
GUIDE = "#7b8793"
RULE = "#dce2e8"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def cub_window(points):
    """Expand the cached keypoints on the 224-pixel official scoring lattice."""
    valid = points[((points >= 0) & (points <= 224)).all(axis=-1)]
    y, x = np.mgrid[:224, :224]
    coords = np.stack((x, y), axis=-1)
    return (np.abs(coords[..., None, :] - valid) <= 45).all(axis=-1).any(axis=-1)


def check_hits(maps, peaks, hits, window):
    side = maps.shape[-1]
    flat = maps.reshape(len(maps), -1).argmax(axis=1)
    np.testing.assert_array_equal(peaks, np.stack((flat % side, flat // side), axis=-1))
    np.testing.assert_array_equal(hits, window[peaks[:, 1], peaks[:, 0]].astype(int))


def photo(ax, rgb, title):
    ax.imshow(rgb)
    ax.set_axis_off()
    ax.set_title(title, pad=3)


def outline(ax, mask, color=CYAN):
    ax.contour(mask, levels=[.5], colors=[color], linewidths=.75)


def mark(ax, xy, symbol, color, size=5):
    from matplotlib.patheffects import withStroke

    point, = ax.plot(*xy, marker=symbol, linestyle="none", markersize=size,
                     color=color, markeredgewidth=1.05, markerfacecolor="none", zorder=5)
    point.set_path_effects([withStroke(linewidth=2.0, foreground="black")])


def heatmap(ax, values):
    scaled = (values - values.min()) / max(float(np.ptp(values)), 1e-12)
    ax.imshow(scaled, cmap="magma", vmin=0, vmax=1, alpha=.65 * scaled)


def panel(fig, x, y, name):
    fig.text(x, y, name, fontsize=7.4, weight="bold", va="top", color=INK)


def save(fig, out_dir, preview_dir, name):
    import matplotlib.pyplot as plt

    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None})
    if preview_dir is not None:
        fig.savefig(preview_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def draw_framework(rgb, case, cells, framework, out_dir, preview_dir):
    """Illustrate the audit using a CUB query and its cohort's recorded results."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig = plt.figure(figsize=(4.8, 3.65))
    panel(fig, .018, .985, "(a) Fix one scoring window")
    window = case["cub__window"]
    tiles = (
        (.042, rgb[0], "Recipient window"),
        (.237, rgb[1], "Donor image"),
        (.467, rgb[0], "Donor map on recipient"),
    )
    axes = [fig.add_axes((x, .752, .154, .203)) for x, _, _ in tiles]
    for ax, (_, image, title) in zip(axes, tiles):
        photo(ax, image, "")
        ax.text(.5, -.058, title, ha="center", va="top", transform=ax.transAxes)
    outline(axes[0], window)
    heatmap(axes[2], case["cub__maps"][1])
    outline(axes[2], window)
    peak = case["cub__peaks_xy"][1]
    hit = bool(window[peak[1], peak[0]])
    mark(axes[2], peak, "x" if hit else "o", "white")
    fig.add_artist(FancyArrowPatch((.400, .851), (.455, .851), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=7, color=GUIDE, linewidth=.8))
    fig.text(.711, .883, "Fixed concept", weight="bold", fontsize=8)
    fig.text(.711, .839, "Solid breast pattern", fontsize=7.6)
    fig.text(.711, .786, f"Donor peak {'hits' if hit else 'misses'} ({int(hit)})", fontsize=7.6)

    panel(fig, .018, .677, "(b) Compare image and head controls")
    matrix = fig.add_axes((.035, .349, .93, .257))
    matrix.set(xlim=(0, 1), ylim=(0, 1))
    matrix.set_axis_off()
    centers = (.36, .60, .85)
    for xc, label in zip(centers, ("Own image", "Donor image", "Own − donor")):
        matrix.text(xc, .94, label, ha="center", va="center", fontsize=7.4, weight="bold")
    gains = {}
    for j, (head, label, color, fill) in enumerate(
        (("trained", "Trained", BLUE, "#edf4f9"), ("reference", "Randomized", RUST, "#fbf1ed"))
    ):
        y = .62 - j * .37
        matrix.axhspan(y - .108, y + .108, color=fill, linewidth=0)
        matrix.text(.025, y, label, ha="left", va="center", fontsize=7.6,
                    weight="bold", color=color)
        own, donor = cells[f"{head}_own"], cells[f"{head}_donor"]
        gains[head] = own - donor
        for xc, value in zip(centers, (own, donor, gains[head])):
            matrix.text(xc, y, f"{value:.5f}", ha="center", va="center", color=color,
                        fontsize=8.3)
    delta = gains["trained"] - gains["reference"]
    fig.text(.835, .334, rf"$\Delta G={delta:.5f}$", ha="center", va="center",
             fontsize=8.8, weight="bold")

    panel(fig, .018, .262, "(c) Vary the donor policy")
    ax = fig.add_axes((.128, .126, .82, .071))
    ax.axvline(0, color=GUIDE, linestyle=(0, (3, 2)), linewidth=.75)
    d0 = framework["uniform_excess"]
    lo, hi = framework["attainable_range"]
    plot_range(ax, {"range": [lo, hi]}, .53, d0, INK)
    for val in (lo, hi):
        ax.annotate(f"{val:+.3f}".replace("-", "−"), (val, .53), (0, 6), ha="center",
                    textcoords="offset points", fontsize=7.3)
    ax.set(xlim=(-.74, .64), ylim=(0, 1.35), yticks=[])
    ax.set_xticks([-.6, -.3, 0, .3, .6])
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    fig.text(.54, .038, rf"Uniform pairing (diamond): $d_0={d0:.4f}$",
             ha="center", color=INK, fontsize=7.3)
    save(fig, out_dir, preview_dir, "fig_framework")


def draw_geometry(images, original, extra, out_dir, preview_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(4.8, 3.30))
    fig.subplots_adjust(left=.012, right=.988, top=.85, bottom=.14, wspace=.07, hspace=.43)
    for row, key in enumerate(("cub", "fb")):
        rgb = images[row][0]
        annotation = extra["cub_annotation_xy"] if key == "cub" else extra["fb_part_mask"]
        window = original[f"{key}__window"]
        for ax, title in zip(axes[row], ("Part annotation", "Scoring window", "Scored peaks")):
            photo(ax, rgb, title)
        if key == "cub":
            for point in annotation[(annotation >= 0).all(axis=-1)]:
                mark(axes[row, 0], point, "+", GOLD, size=6)
            panel(fig, .012, .973, "(a) CUB · solid breast pattern ($h=45$)")
        else:
            outline(axes[row, 0], annotation, color=GOLD)
            panel(fig, .012, .536, "(b) FunnyBirds · beak contrast ($r=24$)")
        outline(axes[row, 1], window)
        outline(axes[row, 2], window)
        # These are the same trained/own and random/own peaks used in the four audit cells.
        for peak, color in zip(original[f"{key}__peaks_xy"][[0, 2]], ("#c1e7ff", "#ffc3a8")):
            hit = window[peak[1], peak[0]]
            mark(axes[row, 2], peak, "x" if hit else "o", color)
    styles = [
        Line2D([], [], color=GOLD, linewidth=1.2, label="Annotation"),
        Line2D([], [], color=CYAN, linewidth=1.2, label="Acceptance boundary"),
        Line2D([], [], marker="x", color=INK, linestyle="none", label="Hit (1)"),
        Line2D([], [], marker="o", markerfacecolor="none", color=INK,
               linestyle="none", label="Miss (0)"),
    ]
    fig.legend(handles=styles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .01))
    save(fig, out_dir, preview_dir, "geometry_scoring_examples")


def draw_concept(rgb, info, extra, bridge, out_dir, preview_dir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    fig = plt.figure(figsize=(4.8, 2.18))
    left = fig.add_axes((.005, .255, .257, .564))
    right = fig.add_axes((.268, .255, .257, .564))
    chart = fig.add_axes((.655, .252, .306, .536))
    panel(fig, .012, .971, "(a) One CUB photo, same bill window")
    panel(fig, .598, .971, "(b) Fixed recipients")
    names = {"all-purpose": "All-purpose", "about_the_same_as_head": "About head length"}
    window = cub_window(extra["copresent_annotation_xy"])
    for ax, values, point, concept in zip(
        (left, right), extra["copresent_maps"], info["peaks_xy"], info["concepts"]
    ):
        attribute, value = concept.split("::")
        photo(ax, rgb, attribute.removeprefix("has_").replace("_", " ").capitalize())
        heatmap(ax, values)
        outline(ax, window)
        mark(ax, point, "x", "white")
        ax.text(.5, -.06, names.get(value, value.replace("_", " ")), ha="center", va="top",
                transform=ax.transAxes)
    fig.text(.264, .038, "Both annotated present · both peaks hit", ha="center", color=INK)

    values = [bridge["asymmetric_I"], bridge["symmetric_I"]]
    chart.axhline(0, color=GUIDE, linestyle=(0, (3, 2)), linewidth=.75)
    chart.plot([0, 1], values, color=GUIDE, lw=.85, zorder=2)
    for k, (value, color) in enumerate(zip(values, (RUST, BLUE))):
        chart.scatter(k, value, color=color, s=23, zorder=3)
        chart.annotate(f"{value:+.4f}".replace("-", "−"), (k, value), (0, 5),
                       textcoords="offset points", ha="center", color=color, fontsize=7.0)
    chart.set(xlim=(-.34, 1.34), ylim=(-.032, .013), ylabel="Label contrast $I$")
    chart.set_xticks([0, 1], ["Target\nrequired", "Both\nrequired"])
    chart.set_yticks([-.03, -.02, -.01, 0, .01])
    chart.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    clean_axis(chart)
    effect, (lo, hi) = bridge["donor_rule_difference"], bridge["paired_image_ci95"]
    fig.text(.79, .023, f"Paired shift +{effect:.4f}  [{lo:.4f}, {hi:.4f}]",
             ha="center", color=INK, fontsize=7.0)
    save(fig, out_dir, preview_dir, "concept_eligibility_examples")


def clean_axis(ax):
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("bottom", "left"):
        ax.spines[edge].set(color=GUIDE, linewidth=.6)
    ax.tick_params(length=2.4, width=.6, pad=2)


def plot_range(ax, result, y, d0, color):
    low, high = result["range"]
    if not low <= d0 <= high:
        raise ValueError("The shared baseline must be feasible for every policy class")
    ax.hlines(y, low, high, color=color, linewidth=2)
    ax.plot([low, high], [y, y], linestyle="none", marker="|", color=color, markersize=7)
    ax.plot(d0, y, linestyle="none", marker="D", markersize=4,
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=.75)


def draw_ranges(calibration, out_dir, preview_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(4.8, 2.20))
    overview = fig.add_axes((.235, .258, .343, .540))
    detail = fig.add_axes((.663, .258, .310, .540))
    panel(fig, .012, .97, "(a) All four donor policy classes")
    panel(fig, .641, .97, "(b) Same species, zoomed")
    groups = [
        ("all_species_group_mean", "All species\nAttribute mean", RUST),
        ("all_species_per_query", "All species\nPer query", BLUE),
        ("same_species_group_mean", "Same species\nAttribute mean", RUST),
        ("same_species_per_query", "Same species\nPer query", BLUE),
    ]
    d0 = calibration["uniform_excess"]
    for ax in (overview, detail):
        ax.axvline(0, color=GUIDE, linewidth=.8, linestyle=(0, (3, 2)), zorder=0)
        clean_axis(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(r"Attainable crossed excess $\Delta G$", labelpad=3)
    for y, (key, _, color) in zip((3, 2, 1, 0), groups):
        plot_range(overview, calibration["classes"][key], y, d0, color)
    overview.set(xlim=(-.67, .59), ylim=(-.45, 3.45))
    overview.set_xticks([-.6, -.3, 0, .3, .5])
    overview.set_yticks([3, 2, 1, 0], [label for _, label, _ in groups])
    for y, (key, label, color) in zip((1, 0), groups[2:]):
        record = calibration["classes"][key]
        plot_range(detail, record, y, d0, color)
        detail.text(.048, y + .24, label.splitlines()[1], ha="center", color=color)
        low, high = record["range"]
        for end in (low, high):
            detail.text(end, y - .26, f"{end:+.4f}".replace("-", "−"),
                        color=color, ha="center", fontsize=7.0)
    detail.set(xlim=(-.062, .142), ylim=(-.43, 1.50))
    detail.set_xticks([-.04, 0, .04, .08, .12])
    detail.set_yticks([])
    handles = [
        Line2D([], [], ls="none", marker="D", markeredgecolor=INK, markerfacecolor="white",
               markersize=4, label=f"Shared baseline $d_0$ = {d0:.4f}"),
        Line2D([], [], color=GUIDE, lw=.8, linestyle=(0, (3, 2)), label="Zero excess"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .005))
    save(fig, out_dir, preview_dir, "donor_calibration_ranges")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--funnybirds-zip", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=BASE.parents[1],
                        help="Public audit checkout containing experiments/ and data/")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.preview_dir or args.out_dir
    os.environ["MPLCONFIGDIR"] = str((cache_dir / ".mplconfig").resolve())
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        **PAPER_FONT, "font.size": 7.3, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
    })
    manifest = read_json(BASE / "manifest.json")
    for filename, expected in manifest["files"].items():
        if sha256(BASE / filename) != expected:
            raise ValueError(f"Input checksum mismatch: {filename}")
    provenance = read_json(BASE / "data/extended_provenance.json")
    numeric_paths = {
        "donor_calibration": "experiments/donor_assignment/query_calibration_results.json",
        "donor_bridge": "data/donor_bridge/expected.json",
    }
    numeric = {}
    for name, relative in numeric_paths.items():
        source = args.audit_root / relative
        if sha256(source) != provenance["public_results_sha256"][name]:
            raise ValueError(f"Public result checksum mismatch: {relative}")
        numeric[name] = read_json(source)
    original = read_npz(BASE / "data/maps.npz")
    extra = read_npz(BASE / "data/extended_examples.npz")
    case = provenance["copresent_case"]
    source_cases = read_json(BASE / "data/provenance.json")["cases"]
    images = load_images(args.cub_root, args.funnybirds_zip, [*source_cases, case])
    np.testing.assert_array_equal(original["cub__window"], cub_window(extra["cub_annotation_xy"]))
    np.testing.assert_array_equal(original["fb__window"], maximum_filter(
        extra["fb_part_mask"], size=49, mode="constant"
    ))
    for key, geometry_case in zip(("cub", "fb"), source_cases):
        check_hits(original[f"{key}__maps"], original[f"{key}__peaks_xy"],
                   geometry_case["hits"], original[f"{key}__window"])
    check_hits(extra["copresent_maps"], np.array(case["peaks_xy"]), case["hits"],
               cub_window(extra["copresent_annotation_xy"]))
    framework_source = args.audit_root / "docs/cub_robustness_results.json"
    framework = provenance["framework"]
    if sha256(framework_source) != framework["four_cell_source_sha256"]:
        raise ValueError("Frozen cohort cell values do not match the figure manifest")
    cells = read_json(framework_source)["primary_cells"]
    draw_framework(images[0], original, cells, framework, args.out_dir, args.preview_dir)
    draw_geometry(images, original, extra, args.out_dir, args.preview_dir)
    draw_concept(images[2][0], case, extra, numeric["donor_bridge"], args.out_dir, args.preview_dir)
    draw_ranges(numeric["donor_calibration"], args.out_dir, args.preview_dir)
    print(json.dumps({"source_images_verified": sum(map(len, images)), "peaks_verified": 10,
                      "windows_verified": 3, "numeric_inputs_verified": list(numeric)}))


if __name__ == "__main__":
    main()
