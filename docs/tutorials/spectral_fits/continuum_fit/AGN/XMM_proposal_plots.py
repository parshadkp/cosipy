import argparse
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


plt.rcParams.update(
    {
        "font.size": 19,
        "axes.titlesize": 24,
        "axes.labelsize": 23,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 17,
    }
)

default_plot_dir = Path(
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/"
    "COSI/Radio_Quiet_AGN/XRay/Simulations/Ecut1000_XMM25_NuSTAR30/PlotData"
)
parser = argparse.ArgumentParser(description="Plot the 1000-keV NGC 4151 XSPEC run.")
parser.add_argument(
    "--simulation-dir", type=Path, default=default_plot_dir.parent,
    help="Run directory containing PlotData and Fits (default: 1000-keV 25/30-ks run).",
)
parser.add_argument("--cosi-best", nargs=2, type=float, metavar=("GAMMA", "ECUT"))
parser.add_argument("--joint-best", nargs=2, type=float, metavar=("GAMMA", "ECUT"))
parser.add_argument("--truth-gamma", type=float, default=1.75)
parser.add_argument("--truth-ecut", type=float, default=1000.0)
parser.add_argument("--no-manual-upper-limit", action="store_true",
                    help="Plot all positive COSI points; do not reuse the bright-run hand-set limit.")
parser.add_argument("--cosi-last-count", type=int,
                    help="Replace the last COSI point with a 95%% Poisson upper limit using its observed count.")
parser.add_argument("--cosi-penultimate-display-limit", action="store_true",
                    help="Show the second-to-last COSI point as an illustrative upper-limit arrow between the preceding point and final limit.")
parser.add_argument("--auto-limits", action="store_true",
                    help="Choose plot bounds from the selected run's SED and steppar grids.")
parser.add_argument("--reference-sed-limits", action="store_true",
                    help="Retain the original SED axis limits while autoscaling the contour panel.")
parser.add_argument("--focus-contours", action="store_true",
                    help="Frame the COSI and joint confidence regions when a broad X-ray scan is open.")
parser.add_argument("--no-edge-note", action="store_true",
                    help="Keep the reference figure layout without an in-panel open-contour note.")
parser.add_argument("--sed-file", default="joint_sed.qdp",
                    help="QDP file for the three instrument SED blocks.")
parser.add_argument("--cosi-grid-prefix", default="cosi",
                    help="Prefix of the COSI steppar vectors (for example, cosi_fine).")
parser.add_argument("--state-label", default="",
                    help="Optional state description added to panel titles.")
parser.add_argument("--model-limit-note", action="store_true",
                    help="Mark scans above the linked xillver 1000-keV Ecut range as provisional.")
parser.add_argument("--output-stem", default="NGC4151_COSI_Ecut_improvement",
                    help="Output PDF/PNG basename within PlotData.")
args = parser.parse_args()
plot_dir = args.simulation_dir / "PlotData"
if not plot_dir.is_dir():
    parser.error(f"PlotData directory does not exist: {plot_dir}")
if Path(args.output_stem).name != args.output_stem:
    parser.error("--output-stem must be a filename stem, not a path")
if args.cosi_last_count is not None and args.cosi_last_count <= 0:
    parser.error("--cosi-last-count must be positive")
if args.cosi_last_count is not None and args.no_manual_upper_limit:
    parser.error("Choose either --cosi-last-count or --no-manual-upper-limit")
if args.cosi_penultimate_display_limit and args.no_manual_upper_limit:
    parser.error("The penultimate display limit requires the last COSI upper limit")


def best_fit(pair, name, old_default):
    if pair is not None:
        return pair
    summary = args.simulation_dir / "Fits" / f"fit_{name}.json"
    if summary.is_file():
        values = json.loads(summary.read_text())
        return values["gamma"], values["ecut"]
    return old_default


cosi_best = best_fit(args.cosi_best, "cosi", (1.73997, 967.528))
joint_best = best_fit(args.joint_best, "joint", (1.75366, 993.75))

fit_colors = {
    "XMM only": "#56B4E9",
    "XMM + NuSTAR": "#0072B2",
    "COSI only": "#E69F00",
    "Joint": "#111111",
}

instrument_styles = {
    "XMM": {"color": "#009E73", "marker": "o"},
    "NuSTAR": {"color": "#CC79A7", "marker": "s"},
    "COSI": {"color": "#E69F00", "marker": "D"},
}

contour_levels = [2.30, 4.605, 9.210]
contour_styles = ["-", "--", ":"]


def load_vector(filename):
    return np.atleast_1d(np.loadtxt(plot_dir / filename))


def load_grid(prefix):
    gamma = load_vector(f"{prefix}_2d_gamma.txt")
    ecut = load_vector(f"{prefix}_2d_ecut.txt")
    delta_c = load_vector(f"{prefix}_2d_delc.txt")

    if not (gamma.size == ecut.size == delta_c.size):
        raise ValueError(f"The {prefix} contour vectors have different lengths.")

    gamma_axis = np.unique(gamma)
    ecut_axis = np.unique(ecut)
    grid = np.full((len(ecut_axis), len(gamma_axis)), np.nan)

    gamma_index = np.searchsorted(gamma_axis, gamma)
    ecut_index = np.searchsorted(ecut_axis, ecut)
    grid[ecut_index, gamma_index] = delta_c
    raw_min = np.nanmin(grid)
    if raw_min > 1.0:
        raise ValueError(
            f"{prefix} scan missed the fitted minimum (smallest raw delta C={raw_min:.2f}); "
            "refine the steppar grid before plotting."
        )
    grid -= raw_min

    return gamma_axis, ecut_axis, grid


def warn_if_contour_hits_grid_edge(label, grid):
    edge = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    edge = edge[np.isfinite(edge)]
    open_intervals = []
    for confidence, level in zip(("68%", "90%", "99%"), contour_levels):
        if edge.size and np.nanmin(edge) <= level:
            open_intervals.append(confidence)
            warnings.warn(
                f"{label} {confidence} contour reaches a steppar grid edge; "
                "widen that XSPEC scan before interpreting it as a closed interval.",
                RuntimeWarning,
            )
    return open_intervals


def load_qdp_blocks(filename):
    """Read numeric QDP blocks separated by one or more NO rows."""
    blocks = []
    current = []

    with (plot_dir / filename).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(("!", "@")):
                continue
            if line.upper().startswith(("READ", "LABEL", "LOG", "TIME", "SKIP")):
                continue
            if line.upper().startswith("NO"):
                if current:
                    blocks.append(np.asarray(current, dtype=float))
                    current = []
                continue

            try:
                current.append([float(value) for value in line.split()])
            except ValueError:
                continue

    if current:
        blocks.append(np.asarray(current, dtype=float))

    if not blocks:
        raise ValueError(f"No numeric data found in {filename}.")

    return blocks


def load_model_curve(filename):
    """Load XSPEC `plot eemodel` output: E, dE, and total E^2 model."""
    block = load_qdp_blocks(filename)[0]
    if block.shape[1] < 3:
        raise ValueError(f"{filename} does not contain an eemodel curve.")
    energy = block[:, 0]
    sed = block[:, 2]
    keep = np.isfinite(energy) & np.isfinite(sed) & (energy > 0) & (sed > 0)
    return energy[keep], sed[keep]


def load_joint_sed_points(filename="joint_sed.qdp"):
    """Load XMM, NuSTAR, and COSI blocks from XSPEC `plot eeufspec`."""
    blocks = load_qdp_blocks(filename)
    if len(blocks) != 3:
        raise ValueError(
            f"Expected three instrument blocks in {filename}; found {len(blocks)}."
        )

    result = {}
    for instrument, block in zip(("XMM", "NuSTAR", "COSI"), blocks):
        if block.shape[1] < 4:
            raise ValueError(f"The {instrument} block in {filename} lacks SED errors.")
        energy, energy_error, sed, sed_error = block[:, :4].T
        keep = (
            np.isfinite(energy)
            & np.isfinite(energy_error)
            & np.isfinite(sed)
            & np.isfinite(sed_error)
            & (energy > 0)
            & (sed > 0)
            & (sed_error > 0)
        )
        result[instrument] = tuple(
            values[keep] for values in (energy, energy_error, sed, sed_error)
        )
    return result


contour_data = {
    "XMM + NuSTAR": load_grid("xn"),
    "COSI only": load_grid(args.cosi_grid_prefix),
    "Joint": load_grid("joint"),
}

joint_model_curve = load_model_curve("model_xmm_nustar_cosi.qdp")

sed_points = load_joint_sed_points(args.sed_file)

fig, (ax_sed, ax_contour) = plt.subplots(
    1,
    2,
    figsize=(16.0, 6.7),
    gridspec_kw={"width_ratios": [1.15, 1]},
    constrained_layout=True,
)

# Panel (a): unfolded SED points and the joint broad-band fitted model.
for instrument, (energy, energy_error, sed, sed_error) in sed_points.items():
    style = instrument_styles[instrument]
    if instrument == "COSI" and not args.no_manual_upper_limit:
        omitted = 2 if args.cosi_penultimate_display_limit else 1
        energy, energy_error, sed, sed_error = (
            values[:-omitted] for values in (energy, energy_error, sed, sed_error)
        )
    ax_sed.errorbar(
        energy,
        sed,
        xerr=energy_error,
        yerr=sed_error,
        linestyle="none",
        marker=style["marker"],
        markersize=4.2 if instrument != "COSI" else 5.8,
        markerfacecolor=style["color"],
        markeredgecolor=style["color"],
        ecolor=style["color"],
        elinewidth=0.85,
        capsize=0,
        alpha=0.82 if instrument != "COSI" else 0.95,
        rasterized=True,
        zorder=4 if instrument != "COSI" else 5,
    )

if not args.no_manual_upper_limit:
    cosi_energy, cosi_energy_error, _, _ = sed_points["COSI"]
    if args.cosi_penultimate_display_limit and len(cosi_energy) < 3:
        raise ValueError("At least three COSI SED points are required for the display limit")
    if args.cosi_last_count is None:
        upper_limit_value = 1.0e-2
    else:
        from scipy.stats import chi2

        observed_sed = sed_points["COSI"][2][-1]
        upper_counts = 0.5 * chi2.ppf(0.95, 2 * (args.cosi_last_count + 1))
        upper_limit_value = observed_sed * upper_counts / args.cosi_last_count
    ax_sed.errorbar(
        cosi_energy[-1:],
        [upper_limit_value],
        xerr=cosi_energy_error[-1:],
        yerr=[0.5 * upper_limit_value],
        uplims=True,
        fmt="none",
        color=instrument_styles["COSI"]["color"],
        ecolor=instrument_styles["COSI"]["color"],
        elinewidth=1.8,
        capsize=4,
        label="COSI upper limit",
        zorder=7,
    )
    if args.cosi_penultimate_display_limit:
        preceding_sed = sed_points["COSI"][2][-3]
        if preceding_sed <= upper_limit_value:
            raise ValueError("The seventh COSI point must exceed the ninth upper limit")
        # Geometric midpoint is visually halfway on the logarithmic SED axis.
        # This is a conservative display limit, not a Poisson confidence bound.
        penultimate_display_limit = np.sqrt(preceding_sed * upper_limit_value)
        warnings.warn(
            "The penultimate COSI arrow is an illustrative display limit, "
            "not a statistical confidence bound.", RuntimeWarning,
        )
        ax_sed.errorbar(
            cosi_energy[-2:-1], [penultimate_display_limit],
            xerr=cosi_energy_error[-2:-1],
            yerr=[0.5 * penultimate_display_limit],
            uplims=True, fmt="none",
            color=instrument_styles["COSI"]["color"],
            ecolor=instrument_styles["COSI"]["color"],
            elinewidth=1.8, capsize=4, zorder=7,
        )

joint_energy, joint_sed = joint_model_curve
ax_sed.plot(
    joint_energy,
    joint_sed,
    color=fit_colors["Joint"],
    linewidth=2.6,
    zorder=6,
    label="Joint best fit",
)

ax_sed.set_xscale("log")
ax_sed.set_yscale("log")
if args.auto_limits and not args.reference_sed_limits:
    visible_model = joint_sed[(joint_energy >= 2.0) & (joint_energy <= 1.0e4)]
    visible_points = [values[2] for values in sed_points.values()]
    positive_sed = np.concatenate([visible_model, *visible_points])
    positive_sed = positive_sed[np.isfinite(positive_sed) & (positive_sed > 0)]
    if not positive_sed.size:
        raise ValueError("No positive SED values available for axis scaling")
    ax_sed.set_xlim(2.0, 1.0e4)
    ax_sed.set_ylim(positive_sed.min() / 2.0, positive_sed.max() * 2.0)
else:
    ax_sed.set_xlim(2.0, 1.0e4)
    ax_sed.set_ylim(3.0e-3, 5.0e-1)
ax_sed.set_xlabel("Energy (keV)")
ax_sed.set_ylabel(r"$E^2\,dN/dE$ (keV cm$^{-2}$ s$^{-1}$)")
state_title = f"{args.state_label} " if args.state_label else ""
ax_sed.set_title(f"(a) {state_title}simulated SED and joint best fit")
ax_sed.tick_params(
    which="major", direction="in", top=True, right=True, length=8, width=1.4
)
ax_sed.tick_params(
    which="minor", direction="in", top=True, right=True, length=4.5, width=1.1
)

ax_sed.legend(
    handles=[
        Line2D(
            [0], [0], color=instrument_styles[instrument]["color"],
            marker=instrument_styles[instrument]["marker"], linestyle="none",
            markersize=8, label=instrument,
        )
        for instrument in ("XMM", "NuSTAR", "COSI")
    ] + [
        Line2D(
            [0], [0], color=fit_colors["Joint"], linewidth=2.6,
            label="Joint best fit",
        )
    ],
    loc="lower left",
    frameon=False,
)

# Panel (b): Gamma-Ecut confidence contours.
open_contours = {}
for label, (gamma, ecut, delta_c) in contour_data.items():
    open_contours[label] = warn_if_contour_hits_grid_edge(label, delta_c)
    ax_contour.contour(
        gamma, ecut, delta_c, levels=contour_levels,
        colors=[fit_colors[label]] * 3,
        linestyles=contour_styles,
        linewidths=[2.4, 1.9, 1.4],
    )

ax_contour.scatter(
    args.truth_gamma, args.truth_ecut, marker="*", s=170, facecolor="#D62728",
    edgecolor="black", zorder=10,
)
ax_contour.scatter(
    *cosi_best, marker="o", s=90,
    facecolor=fit_colors["COSI only"], edgecolor="black", zorder=11,
)
ax_contour.scatter(
    *joint_best, marker="x", s=75, linewidth=2.2,
    color="black", zorder=11,
)

ax_contour.set_yscale("log")
if args.auto_limits:
    framing_data = (
        [contour_data["COSI only"], contour_data["Joint"]]
        if args.focus_contours else list(contour_data.values())
    )
    contour_points = []
    for gamma_axis, ecut_axis, delta_c in framing_data:
        gamma_mesh, ecut_mesh = np.meshgrid(gamma_axis, ecut_axis)
        inside = np.isfinite(delta_c) & (delta_c <= contour_levels[-1])
        if np.any(inside):
            contour_points.append((gamma_mesh[inside], ecut_mesh[inside]))
    if not contour_points:
        raise ValueError("No points lie inside the plotted 99% contour levels")
    all_gamma = np.concatenate([item[0] for item in contour_points])
    all_ecut = np.concatenate([item[1] for item in contour_points])
    xlow = min(all_gamma.min(), args.truth_gamma, cosi_best[0], joint_best[0])
    xhigh = max(all_gamma.max(), args.truth_gamma, cosi_best[0], joint_best[0])
    ylow = min(all_ecut.min(), args.truth_ecut, cosi_best[1], joint_best[1])
    yhigh = max(all_ecut.max(), args.truth_ecut, cosi_best[1], joint_best[1])
    ax_contour.set_xlim(xlow - 0.03 * (xhigh - xlow), xhigh + 0.03 * (xhigh - xlow))
    ax_contour.set_ylim(ylow / 1.12, yhigh * 1.12)
else:
    ax_contour.set_xlim(1.65, 1.85)
    ax_contour.set_ylim(780, 1300)
ax_contour.set_xlabel(r"Photon index, $\Gamma$")
ax_contour.set_ylabel(r"$E_{\rm cut}$ (keV)")
ax_contour.set_title(r"(b) $\Gamma$-$E_{\rm cut}$ confidence contours")
ax_contour.tick_params(
    which="major", direction="in", top=True, length=8, width=1.4
)
ax_contour.tick_params(
    which="minor", direction="in", top=True, length=4.5, width=1.1
)
ax_contour.tick_params(axis="x", which="major", pad=8)
ax_contour.tick_params(axis="y", which="major", pad=8)

secondary_axis = ax_contour.secondary_yaxis(
    "right", functions=(lambda ecut: ecut / 2.5, lambda kte: kte * 2.5)
)
secondary_axis.set_ylabel(r"$kT_e (keV)$")

contour_instrument_handles = [
    Line2D([0], [0], color=fit_colors[label], lw=2.3, label=label)
    for label in contour_data
]
confidence_handles = [
    Line2D([0], [0], color="0.35", linestyle=style, lw=1.8, label=confidence)
    for style, confidence in zip(contour_styles, ["68%", "90%", "99%"])
]

contour_legend = ax_contour.legend(
    handles=contour_instrument_handles, loc="upper right", frameon=False
)
ax_contour.add_artist(contour_legend)
confidence_legend = ax_contour.legend(
    handles=confidence_handles, loc="lower right", frameon=False,
)
ax_contour.add_artist(confidence_legend)
if args.auto_limits and any(open_contours.values()) and not args.no_edge_note:
    open_labels = [name for name, levels in open_contours.items() if levels]
    ax_contour.text(
        0.02, 0.02, f"{', '.join(open_labels)}: open at scan edge",
        transform=ax_contour.transAxes, ha="left", va="bottom", fontsize=12,
        color="#A12424", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
if args.state_label:
    fig.suptitle("Background-free simulation; absorption and reflection fixed", fontsize=15)
if args.model_limit_note:
    fig.text(
        0.5, -0.025,
        "Exploratory: linked xillver Ecut range ends at 1000 keV; higher-cutoff contours are not validated.",
        ha="center", fontsize=12, color="#A12424",
    )
# ax_contour.legend(
#     handles=[
#         Line2D([0], [0], marker="*", linestyle="none", markersize=12,
#                markerfacecolor="#D62728", markeredgecolor="black",
#                label="Injected: 1000 keV"),
#         Line2D([0], [0], marker="o", linestyle="none", markersize=8,
#                markerfacecolor=fit_colors["COSI only"], markeredgecolor="black",
#                label="COSI best: 968 keV"),
#         Line2D([0], [0], marker="x", linestyle="none", markersize=8,
#                color=fit_colors["Joint"], label="Joint best: 994 keV"),
#     ],
#     loc="lower left", frameon=True, facecolor="white",
#     framealpha=0.9, edgecolor="none",
# )

fig.savefig(plot_dir / f"{args.output_stem}.pdf", bbox_inches="tight")
fig.savefig(
    plot_dir / f"{args.output_stem}.png",
    dpi=300,
    bbox_inches="tight",
)

if plt.get_backend().lower() != "agg":
    plt.show()
