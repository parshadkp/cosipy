"""Reusable NGC 1068 fluctuated-spectrum fitting and SED plotting helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astromodels import Line, Model, PointSource
from threeML import DataList, JointLikelihood

from agn_cosi_fit_utils import (
    COSIPlugin,
    cosi_source_detection_ts,
    load_agn_manifest_histograms,
    make_cosi_background_parameter,
    model_improvement_ts,
    open_spacecraft_history,
    scale_spacecraft_livetime,
)
from agn_sed_ensemble import (
    ensemble_summary_to_sed_dataframe,
    fit_representative_sed,
    run_or_load_manifest_sed_ensemble,
)
from cosipy.event_selection import GoodTimeInterval
from astropy.coordinates import SkyCoord


SENSITIVITY_DIR = Path(
    "/Users/parshadkp/Software/cosipy/docs/tutorials/spectral_fits/"
    "Sensitivity_calculator"
)
if str(SENSITIVITY_DIR) not in sys.path:
    sys.path.insert(0, str(SENSITIVITY_DIR))

from agn_fixed_case_sensitivity import _fit_specification, _make_shape


SED_KEV_TO_ERG = u.keV.to(u.erg)
KEV_TO_MEV = u.keV.to(u.MeV)


def load_manifest(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sensitivity manifest not found: {path}. Run its NGC 1068 "
            "sensitivity notebook first."
        )
    return json.loads(path.read_text())


def resolved_case_from_manifest(manifest):
    for key in ("resolved_case_at_threshold", "resolved_case"):
        if key in manifest:
            return manifest[key]
    raise KeyError("Manifest has no resolved NGC 1068 spectral case.")


def make_injected_shape(manifest):
    case = resolved_case_from_manifest(manifest)
    shape = _make_shape(case["primary"], fit=False)
    if case.get("secondary") is not None:
        shape = shape + _make_shape(case["secondary"], fit=False)
    return shape


def make_fit_model(manifest, *, include_secondary=True):
    case = resolved_case_from_manifest(manifest)
    fit_pivot_keV = float(case.get("fit_pivot_keV", 200.0))
    primary_spec = _fit_specification(case["primary"], fit_pivot_keV)
    primary = _make_shape(primary_spec, fit=True)
    source_name = str(case.get("source_name", "NGC1068"))
    secondary_spec = None
    link_function = None

    if include_secondary and case.get("secondary") is not None:
        secondary_spec = _fit_specification(case["secondary"], fit_pivot_keV)
        secondary = _make_shape(secondary_spec, fit=True)
        source = PointSource(
            source_name,
            l=float(case["longitude_deg"]),
            b=float(case["latitude_deg"]),
            spectral_shape=primary + secondary,
        )
        model = Model(source)
        link_function = Line(
            a=0.0,
            b=max(float(secondary_spec["K"] / primary_spec["K"]), 1e-12),
        )
        link_function.a.fix = True
        link_function.b.min_value = 0.0
        model.link(
            model[source_name].spectrum.main.composite.K_2,
            model[source_name].spectrum.main.composite.K_1,
            link_function,
        )
    else:
        source = PointSource(
            source_name,
            l=float(case["longitude_deg"]),
            b=float(case["latitude_deg"]),
            spectral_shape=primary,
        )
        model = Model(source)

    return model, source_name, link_function


def _fit_likelihood(model, plugin):
    likelihood = JointLikelihood(model, DataList(plugin), verbose=False)
    try:
        likelihood.fit(quiet=True)
        likelihood._agn_fit_status = "covariance"
    except Exception as covariance_error:
        results = getattr(likelihood, "_analysis_results", None)
        if results is None:
            likelihood.fit(quiet=True, compute_covariance=False)
            results = likelihood.results
        results._agn_uncertainty_warning = (
            "Covariance sampling failed after minimization: "
            f"{type(covariance_error).__name__}: {covariance_error}"
        )
        likelihood._agn_fit_status = "best-fit-only fallback"
    return likelihood


def fit_manifest_spectrum(
    manifest_path,
    *,
    dataset_name,
    background_pseudocount=None,
):
    """Fit one 3- or 24-month manifest with the manifest's own spectral case."""

    manifest = load_manifest(manifest_path)
    exposure_months = int(manifest["exposure_months"])
    case = resolved_case_from_manifest(manifest)
    _, source_expectation, data_histogram, background_expectation = (
        load_agn_manifest_histograms(
            manifest_path,
            manifest["background_file_3m"],
            exposure_months,
        )
    )

    source_coord = SkyCoord(
        l=float(case["longitude_deg"]),
        b=float(case["latitude_deg"]),
        frame="galactic",
        unit="deg",
    )
    orientation_3m = open_spacecraft_history(manifest["orientation_file_3m"])
    source_gti = GoodTimeInterval.from_pointing_cut(
        source_coord,
        orientation_3m,
        float(case.get("fov_cut_deg", 60.0)) * u.deg,
        earth_occ=False,
    )
    fit_orientation = scale_spacecraft_livetime(
        orientation_3m.apply_gti(source_gti),
        exposure_months / 3.0,
    )

    model, source_name, link_function = make_fit_model(
        manifest, include_secondary=True
    )
    plugin = COSIPlugin(
        dataset_name,
        dr=manifest["response_file"],
        data=data_histogram.project("Em", "Phi", "PsiChi"),
        bkg=background_expectation.project("Em", "Phi", "PsiChi"),
        sc_orientation=fit_orientation,
        nuisance_param=make_cosi_background_parameter(dataset_name),
        background_pseudocount=background_pseudocount,
        earth_occ=True,
    )
    plugin.set_model(model)
    likelihood = _fit_likelihood(model, plugin)
    source_ts, null_result = cosi_source_detection_ts(likelihood, plugin)

    primary_only_likelihood = None
    added_component_delta_ts = None
    if case.get("secondary") is not None:
        primary_model, _, _ = make_fit_model(manifest, include_secondary=False)
        primary_plugin_name = f"{dataset_name}_primary_only"
        primary_plugin = COSIPlugin(
            primary_plugin_name,
            dr=manifest["response_file"],
            data=data_histogram.project("Em", "Phi", "PsiChi"),
            bkg=background_expectation.project("Em", "Phi", "PsiChi"),
            sc_orientation=fit_orientation,
            nuisance_param=make_cosi_background_parameter(primary_plugin_name),
            background_pseudocount=background_pseudocount,
            earth_occ=True,
        )
        primary_plugin.set_model(primary_model)
        primary_only_likelihood = _fit_likelihood(primary_model, primary_plugin)
        added_component_delta_ts = model_improvement_ts(
            primary_only_likelihood, likelihood
        )

    return {
        "manifest_path": Path(manifest_path),
        "manifest": manifest,
        "case": case,
        "exposure_months": exposure_months,
        "source_name": source_name,
        "source_expectation": source_expectation,
        "data_histogram": data_histogram,
        "background_expectation": background_expectation,
        "fit_orientation": fit_orientation,
        "likelihood": likelihood,
        "plugin": plugin,
        "primary_only_likelihood": primary_only_likelihood,
        "link_function": link_function,
        "source_ts": source_ts,
        "null_result": null_result,
        "added_component_delta_ts": added_component_delta_ts,
        "injected_shape": make_injected_shape(manifest),
    }


def make_flux_propagator(fit_bundle):
    """Build a covariance-aware total spectral-flux propagator."""

    results = fit_bundle["likelihood"].results
    source = results.optimized_model[fit_bundle["source_name"]]
    shape = source.spectrum.main.shape

    if shape.__class__.__name__ == "Cutoff_powerlaw":
        def evaluate(energy, K, xc, index):
            return shape.evaluate_at(energy, K=K, xc=xc, index=index)

        return results.propagate(
            evaluate,
            K=results.get_variates(shape.K.path),
            xc=results.get_variates(shape.xc.path),
            index=results.get_variates(shape.index.path),
        )

    primary, secondary = shape.functions
    link = secondary.K.auxiliary_variable[1]
    secondary_is_cutoff = secondary.__class__.__name__ == "Cutoff_powerlaw"

    if secondary_is_cutoff:
        def evaluate(energy, K_1, xc_1, index_1, b, xc_2, index_2):
            thermal = primary.evaluate_at(
                energy, K=K_1, xc=xc_1, index=index_1
            )
            tail = secondary.evaluate_at(
                energy, K=b * K_1, xc=xc_2, index=index_2
            )
            return thermal + tail

        return results.propagate(
            evaluate,
            K_1=results.get_variates(primary.K.path),
            xc_1=results.get_variates(primary.xc.path),
            index_1=results.get_variates(primary.index.path),
            b=results.get_variates(link.b.path),
            xc_2=results.get_variates(secondary.xc.path),
            index_2=results.get_variates(secondary.index.path),
        )

    def evaluate(energy, K_1, xc_1, index_1, b, index_2):
        thermal = primary.evaluate_at(
            energy, K=K_1, xc=xc_1, index=index_1
        )
        tail = secondary.evaluate_at(energy, K=b * K_1, index=index_2)
        return thermal + tail

    return results.propagate(
        evaluate,
        K_1=results.get_variates(primary.K.path),
        xc_1=results.get_variates(primary.xc.path),
        index_1=results.get_variates(primary.index.path),
        b=results.get_variates(link.b.path),
        index_2=results.get_variates(secondary.index.path),
    )


def evaluate_flux_curves(fit_bundle, energy_keV=None, confidence_level=0.68):
    if energy_keV is None:
        energy_keV = np.geomspace(100.0, 10000.0, 160)
    energy_keV = np.asarray(energy_keV, dtype=float)
    injected = np.asarray(
        [fit_bundle["injected_shape"].evaluate_at(e) for e in energy_keV],
        dtype=float,
    )
    shape = fit_bundle["likelihood"].results.optimized_model[
        fit_bundle["source_name"]
    ].spectrum.main.shape
    best_fit = np.asarray([shape.evaluate_at(e) for e in energy_keV], dtype=float)
    low = best_fit.copy()
    high = best_fit.copy()
    try:
        propagator = make_flux_propagator(fit_bundle)
        for index, energy in enumerate(energy_keV):
            distribution = propagator(float(energy))
            best_fit[index] = float(distribution.median)
            low[index], high[index] = distribution.equal_tail_interval(
                cl=float(confidence_level)
            )
    except Exception as error:
        fit_bundle["flux_uncertainty_warning"] = (
            f"Flux covariance propagation unavailable: {type(error).__name__}: {error}"
        )
    return {
        "energy_keV": energy_keV,
        "injected": injected,
        "best_fit": best_fit,
        "low": low,
        "high": high,
    }


def build_sed_products(
    fit_bundle,
    *,
    n_sed_bins=10,
    background_pseudocount=None,
    recompute_ensemble=False,
    expected_seed_count=300,
):
    representative = fit_representative_sed(
        source_expectation=fit_bundle["source_expectation"],
        background_expectation=fit_bundle["background_expectation"],
        data_histogram=fit_bundle["data_histogram"],
        injected_shape=fit_bundle["injected_shape"],
        n_sed_bins=n_sed_bins,
        energy_min_keV=100.0,
        energy_max_keV=10000.0,
        background_pseudocount=background_pseudocount,
        detection_ts=4.0,
    )
    _, ensemble_summary, _, summary_file = run_or_load_manifest_sed_ensemble(
        manifest=fit_bundle["manifest"],
        source_expectation=fit_bundle["source_expectation"],
        background_expectation=fit_bundle["background_expectation"],
        injected_shape=fit_bundle["injected_shape"],
        n_sed_bins=n_sed_bins,
        energy_min_keV=100.0,
        energy_max_keV=10000.0,
        background_pseudocount=background_pseudocount,
        recompute=recompute_ensemble,
        detection_ts=4.0,
        expected_seed_count=expected_seed_count,
    )
    ensemble = ensemble_summary_to_sed_dataframe(ensemble_summary)
    return {
        "representative": representative,
        "ensemble": ensemble,
        "ensemble_summary": ensemble_summary,
        "ensemble_summary_file": summary_file,
    }


def _style_axis(axis, font_size):
    axis.tick_params(
        axis="both", which="major", size=12, width=1.5,
        direction="in", top=True, right=True, pad=8, labelsize=font_size,
    )
    axis.tick_params(
        axis="both", which="minor", size=6, width=1.5,
        direction="in", top=True, right=True,
    )
    axis.spines["right"].set_visible(True)
    axis.spines["top"].set_visible(True)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.085, 12.0)
    axis.set_xlabel("Energy (MeV)", fontsize=font_size)
    axis.set_ylabel(
        r"Energy Flux (erg cm$^{-2}$ s$^{-1}$)", fontsize=font_size
    )


def draw_sed_panel(
    axis,
    fit_bundle,
    curves,
    sed_dataframe,
    *,
    title,
    color="#D55E00",
    font_size=25,
):
    _style_axis(axis, font_size)
    energy = curves["energy_keV"]
    axis.plot(
        energy * KEV_TO_MEV,
        SED_KEV_TO_ERG * energy**2 * curves["injected"],
        color=color, linestyle=":", linewidth=3,
    )
    axis.plot(
        energy * KEV_TO_MEV,
        SED_KEV_TO_ERG * energy**2 * curves["best_fit"],
        color=color, linewidth=2,
    )
    axis.fill_between(
        energy * KEV_TO_MEV,
        SED_KEV_TO_ERG * energy**2 * curves["low"],
        SED_KEV_TO_ERG * energy**2 * curves["high"],
        color=color, alpha=0.18,
    )

    roles = sed_dataframe["plot_role"].astype(str)
    detections = sed_dataframe.loc[roles == "detection"]
    upper_limits = sed_dataframe.loc[roles == "upper_limit"]

    def xerr(frame):
        return KEV_TO_MEV * np.vstack(
            [
                frame["e_ref_keV"] - frame["e_min_keV"],
                frame["e_max_keV"] - frame["e_ref_keV"],
            ]
        )

    if not detections.empty:
        axis.errorbar(
            detections["e_ref_keV"] * KEV_TO_MEV,
            detections["sed_erg_cm2_s"],
            xerr=xerr(detections),
            yerr=np.vstack(
                [
                    np.maximum(
                        detections["sed_erg_cm2_s"]
                        - detections["sed_lo_erg_cm2_s"],
                        0.0,
                    ),
                    np.maximum(
                        detections["sed_hi_erg_cm2_s"]
                        - detections["sed_erg_cm2_s"],
                        0.0,
                    ),
                ]
            ),
            fmt="o", color=color, ecolor=color,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.6, elinewidth=1.5,
            capsize=4, markersize=8, zorder=5,
        )
    if not upper_limits.empty:
        if "sed_ul95_erg_cm2_s" in upper_limits:
            upper_y = upper_limits["sed_ul95_erg_cm2_s"].to_numpy(dtype=float)
        elif "sed_ul95_median_erg_cm2_s" in upper_limits:
            upper_y = upper_limits["sed_ul95_median_erg_cm2_s"].to_numpy(dtype=float)
        else:
            upper_y = upper_limits["sed_hi_erg_cm2_s"].to_numpy(dtype=float)
        axis.errorbar(
            upper_limits["e_ref_keV"] * KEV_TO_MEV,
            upper_y,
            xerr=xerr(upper_limits),
            yerr=np.maximum(0.5 * upper_y, np.finfo(float).tiny),
            uplims=True, fmt="v", color=color, ecolor=color,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.8, elinewidth=2.4,
            capsize=6, markersize=11, barsabove=True, zorder=6,
        )

    axis.text(
        0.98, 0.97, title,
        transform=axis.transAxes, ha="right", va="top",
        fontsize=font_size, fontweight="550",
    )
    handles = [
        plt.Line2D([], [], color=color, linestyle=":", linewidth=3,
                   label="Injected"),
        plt.Line2D([], [], color=color, linewidth=2,
                   label="Best fit & 68% band"),
        plt.Line2D([], [], color="0.25", marker="o",
                   markerfacecolor="white", markeredgewidth=1.6,
                   linestyle="None", markersize=8, label="COSI SED"),
        plt.Line2D([], [], color="0.25", marker="v",
                   markerfacecolor="white", markeredgewidth=1.8,
                   linestyle="None", markersize=10, label="95% upper limit"),
    ]
    axis.legend(handles=handles, fontsize=font_size, loc="lower left", frameon=False)
    return axis


def plot_sed(
    fit_bundle,
    curves,
    sed_dataframe,
    *,
    title,
    save_path=None,
    color="#D55E00",
    font_size=25,
):
    plt.rcParams.update({"font.size": font_size, "font.family": "Times New Roman"})
    fig, axis = plt.subplots(figsize=(12, 9), constrained_layout=True)
    draw_sed_panel(
        axis, fit_bundle, curves, sed_dataframe,
        title=title, color=color, font_size=font_size,
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig, axis


def plot_sed_pair(
    panels,
    *,
    save_path=None,
    font_size=22,
):
    """Plot two named fit/SED products side by side with shared styling."""

    plt.rcParams.update({"font.size": font_size, "font.family": "Times New Roman"})
    fig, axes = plt.subplots(
        1, 2, figsize=(22, 8.5), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for axis, panel in zip(axes, panels):
        draw_sed_panel(
            axis,
            panel["fit_bundle"],
            panel["curves"],
            panel["sed_dataframe"],
            title=panel["title"],
            color=panel.get("color", "#D55E00"),
            font_size=font_size,
        )
    axes[1].set_ylabel("")
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig, axes
