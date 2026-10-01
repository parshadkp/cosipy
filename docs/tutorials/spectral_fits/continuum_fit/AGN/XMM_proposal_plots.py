from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


plot_dir = Path(
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/"
    "COSI/Radio_Quiet_AGN/XRay/Simulations/PlotData"
)

fit_colors = {
    "XMM only": "#56B4E9",
    "XMM + NuSTAR": "#0072B2",
    "COSI only": "#D55E00",
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
    grid -= np.nanmin(grid)

    return gamma_axis, ecut_axis, grid


def warn_if_contour_hits_grid_edge(label, grid):
    edge = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    edge = edge[np.isfinite(edge)]
    for confidence, level in zip(("68%", "90%", "99%"), contour_levels):
        if edge.size and np.nanmin(edge) <= level:
            warnings.warn(
                f"{label} {confidence} contour reaches a steppar grid edge; "
                "widen that XSPEC scan before interpreting it as a closed interval.",
                RuntimeWarning,
            )


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
    "COSI only": load_grid("cosi"),
    "Joint": load_grid("joint"),
}

joint_model_curve = load_model_curve("model_xmm_nustar_cosi.qdp")

sed_points = load_joint_sed_points()

fig, (ax_sed, ax_contour) = plt.subplots(
    1,
    2,
    figsize=(14.0, 5.8),
    gridspec_kw={"width_ratios": [1.15, 1]},
    constrained_layout=True,
)

# Panel (a): unfolded SED points and the joint broad-band fitted model.
for instrument, (energy, energy_error, sed, sed_error) in sed_points.items():
    style = instrument_styles[instrument]
    ax_sed.errorbar(
        energy,
        sed,
        xerr=energy_error,
        yerr=sed_error,
        linestyle="none",
        marker=style["marker"],
        markersize=3.2 if instrument != "COSI" else 5.0,
        markerfacecolor=style["color"],
        markeredgecolor=style["color"],
        ecolor=style["color"],
        elinewidth=0.65,
        capsize=0,
        alpha=0.50 if instrument != "COSI" else 0.95,
        rasterized=True,
        zorder=2 if instrument != "COSI" else 4,
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
ax_sed.set_xlim(2.0, 1.0e4)
ax_sed.set_ylim(1.0e-4, 4.0e-1)
ax_sed.set_xlabel("Energy (keV)")
ax_sed.set_ylabel(r"$E^2\,dN/dE$ (keV cm$^{-2}$ s$^{-1}$)")
ax_sed.set_title(r"(a) Simulated SED and joint best fit")
ax_sed.tick_params(which="both", direction="in", top=True, right=True)

instrument_handles = [
    Line2D(
        [0], [0], linestyle="none", marker=style["marker"],
        color=style["color"], markersize=6, label=instrument,
    )
    for instrument, style in instrument_styles.items()
]
sed_handles = instrument_handles + [
    Line2D(
        [0], [0], color=fit_colors["Joint"], linewidth=2.6,
        label="Joint best fit",
    )
]
ax_sed.legend(
    handles=sed_handles, loc="lower left", frameon=False, fontsize=9,
)

# Panel (b): Gamma-Ecut confidence contours.
for label, (gamma, ecut, delta_c) in contour_data.items():
    warn_if_contour_hits_grid_edge(label, delta_c)
    ax_contour.contour(
        gamma, ecut, delta_c, levels=contour_levels,
        colors=[fit_colors[label]] * 3,
        linestyles=contour_styles,
        linewidths=[2.4, 1.9, 1.4],
    )

ax_contour.scatter(
    1.75, 1000, marker="*", s=170, facecolor="gold",
    edgecolor="black", zorder=10,
)
ax_contour.scatter(
    1.75366, 993.75, marker="x", s=75, linewidth=2.2,
    color="black", zorder=11,
)

ax_contour.set_yscale("log")
ax_contour.set_xlim(1.68, 1.80)
ax_contour.set_ylim(600, 3000)
ax_contour.set_xlabel(r"Photon index, $\Gamma$")
ax_contour.set_ylabel(r"Cutoff energy, $E_{\rm cut}$ (keV)")
ax_contour.set_title(r"(b) $\Gamma$--$E_{\rm cut}$ confidence contours")
ax_contour.tick_params(which="both", direction="in", top=True)

secondary_axis = ax_contour.secondary_yaxis(
    "right", functions=(lambda ecut: ecut / 2.5, lambda kte: kte * 2.5)
)
secondary_axis.set_ylabel(r"$kT_e=E_{\rm cut}/2.5$ (keV)")

contour_instrument_handles = [
    Line2D([0], [0], color=fit_colors[label], lw=2.3, label=label)
    for label in contour_data
]
confidence_handles = [
    Line2D([0], [0], color="0.35", linestyle=style, lw=1.8, label=confidence)
    for style, confidence in zip(contour_styles, ["68%", "90%", "99%"])
]

contour_legend = ax_contour.legend(
    handles=contour_instrument_handles, loc="upper left", frameon=False, fontsize=9,
)
ax_contour.add_artist(contour_legend)
ax_contour.legend(
    handles=confidence_handles, loc="lower right", frameon=False, fontsize=9,
)

fig.savefig(plot_dir / "NGC4151_COSI_Ecut_improvement.pdf", bbox_inches="tight")
fig.savefig(
    plot_dir / "NGC4151_COSI_Ecut_improvement.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
