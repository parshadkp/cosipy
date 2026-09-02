"""Fixed-spectrum AGN Poisson-ensemble sensitivity helpers.

Each notebook in this directory defines one injected spectral case and calls
``run_fixed_case_ensemble``.  Pure cutoff-power-law cases use source detection
TS relative to a background-only reference.  Composite cases use the model
improvement Delta TS relative to the primary cutoff power law alone.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astromodels import Cutoff_powerlaw, Line, Model, Parameter, PointSource, Powerlaw
from histpy import Histogram
from threeML import DataList, JointLikelihood

from cosipy import SpacecraftHistory
from cosipy.data_io import EmCDSBinnedData
from cosipy.event_selection import GoodTimeInterval
from cosipy.response import BinnedInstrumentResponse, PointSourceResponse


AGN_UTILS_DIR = Path(
    "/Users/parshadkp/Software/cosipy/docs/tutorials/spectral_fits/"
    "continuum_fit/AGN/Fluctuate_True"
)
if str(AGN_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(AGN_UTILS_DIR))

from agn_cosi_fit_utils import COSIPlugin, _open_response, scale_spacecraft_livetime


DEFAULT_RESPONSE_PATH = Path(
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/"
    "COSI/Radio_Quiet_AGN/DC4_Files/"
    "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284."
    "filtered.nonsparse.binnedimaging.imagingresponse.h5"
)
DEFAULT_ORIENTATION_PATH = Path(
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/"
    "COSI/Radio_Quiet_AGN/DC4_Files/"
    "DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits"
)
DEFAULT_BACKGROUND_PATH = Path(
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/"
    "COSI/Radio_Quiet_AGN/DC4_Files/Background/"
    "Total_DC4_BG_3months_binned_data_filtered_with_SAAcut_withSAAbck_"
    "NGC4151_60deg_fov_cut.hdf5"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/"
    "COSI/Radio_Quiet_AGN/GammaRay/Paper_Models/Fluctuate_True/"
    "Sensitivity_Ensemble"
)


def _histogram_values(histogram):
    values = histogram.contents
    if hasattr(values, "compute"):
        values = values.compute()
    if hasattr(values, "todense"):
        values = values.todense()
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float)


def histogram_sum(histogram):
    return float(_histogram_values(histogram).sum())


def poisson_realization(total_expectation, seed):
    mean = _histogram_values(total_expectation)
    if np.any(~np.isfinite(mean)) or np.any(mean < 0):
        raise ValueError("Poisson means must be finite and non-negative.")
    realization = total_expectation.copy()
    realization[:] = np.random.default_rng(int(seed)).poisson(mean)
    return realization


def _set_shape_parameter(parameter, value, unit=None):
    parameter.value = float(value)
    if unit is not None:
        parameter.unit = unit


def _make_cpl(specification, *, fit=False):
    shape = Cutoff_powerlaw()
    _set_shape_parameter(shape.K, specification["K"], u.keV**-1 * u.cm**-2 * u.s**-1)
    _set_shape_parameter(shape.piv, specification["pivot_keV"], u.keV)
    _set_shape_parameter(shape.xc, specification["cutoff_keV"], u.keV)
    shape.index.value = float(specification["index"])

    if fit:
        k_min = float(specification.get("K_min", 1e-10))
        k_max = float(specification.get("K_max", 1.0))
        shape.K.value = float(np.clip(specification["K"], k_min, k_max))
        shape.K.min_value = k_min
        shape.K.max_value = k_max
        shape.xc.min_value = float(specification.get("cutoff_min_keV", 100.0))
        shape.xc.max_value = float(specification.get("cutoff_max_keV", 10000.0))
        shape.index.min_value = float(specification.get("index_min", -5.0))
        shape.index.max_value = float(specification.get("index_max", 1.0))
        shape.index.delta = float(specification.get("index_delta", 0.25))
        shape.index.fix = not bool(specification.get("fit_index_free", False))
        shape.xc.fix = not bool(specification.get("fit_cutoff_free", True))
    return shape


def _make_pl(specification, *, fit=False):
    shape = Powerlaw()
    _set_shape_parameter(shape.K, specification["K"], u.keV**-1 * u.cm**-2 * u.s**-1)
    _set_shape_parameter(shape.piv, specification["pivot_keV"], u.keV)
    shape.index.value = float(specification["index"])
    if fit:
        k_min = float(specification.get("K_min", 1e-10))
        k_max = float(specification.get("K_max", 1.0))
        shape.K.value = float(np.clip(specification["K"], k_min, k_max))
        shape.K.min_value = k_min
        shape.K.max_value = k_max
        shape.index.min_value = float(specification.get("index_min", -5.0))
        shape.index.max_value = float(specification.get("index_max", 1.0))
        shape.index.delta = float(specification.get("index_delta", 0.25))
        shape.index.fix = not bool(specification.get("fit_index_free", True))
    return shape


def _make_shape(specification, *, fit=False):
    if specification["kind"] == "cpl":
        return _make_cpl(specification, fit=fit)
    if specification["kind"] == "pl":
        return _make_pl(specification, fit=fit)
    raise ValueError(f"Unsupported spectral shape: {specification['kind']}")


def _re_pivot_specification(specification, new_pivot_keV):
    updated = dict(specification)
    old_pivot_keV = float(specification["pivot_keV"])
    updated["K"] = float(
        specification["K"]
        * (float(new_pivot_keV) / old_pivot_keV) ** float(specification["index"])
    )
    updated["pivot_keV"] = float(new_pivot_keV)
    return updated


def _resolve_case(case):
    resolved = dict(case)
    primary = dict(case["primary"])
    primary_shape = _make_shape(primary, fit=False)
    resolved["primary"] = primary

    secondary = case.get("secondary")
    if secondary is not None:
        secondary = dict(secondary)
        normalization = secondary.pop("normalization", None)
        if normalization is not None:
            if normalization["mode"] != "flux_ratio":
                raise ValueError("Only flux_ratio secondary normalization is supported.")
            reference_energy_keV = float(normalization["energy_keV"])
            secondary["K"] = float(
                normalization["ratio"]
                * primary_shape.evaluate_at(reference_energy_keV)
            )
            secondary["injected_flux_ratio"] = float(normalization["ratio"])
            secondary["normalization_energy_keV"] = reference_energy_keV
        elif "K" not in secondary:
            raise ValueError(
                "A secondary component requires either normalization or K."
            )
        resolved["secondary"] = secondary
    return resolved


def _make_injected_model(case):
    primary = _make_shape(case["primary"], fit=False)
    secondary = case.get("secondary")
    spectral_shape = primary
    if secondary is not None:
        spectral_shape = primary + _make_shape(secondary, fit=False)
    source = PointSource(
        str(case.get("source_name", "NGC4151")),
        l=float(case["longitude_deg"]),
        b=float(case["latitude_deg"]),
        spectral_shape=spectral_shape,
    )
    return Model(source)


def _fit_specification(specification, fit_pivot_keV):
    fit_spec = _re_pivot_specification(specification, fit_pivot_keV)
    fit_spec["K_min"] = float(specification.get("K_min", 1e-10))
    fit_spec["K_max"] = float(specification.get("K_max", 1.0))
    return fit_spec


def _make_background_parameter(name, initial_value=1.0):
    return Parameter(
        name,
        float(initial_value),
        min_value=0.0,
        max_value=5.0,
        delta=0.05,
        desc="Background amplitude for an AGN fixed-case sensitivity fit",
    )


def _source_name(case):
    """Return the filename/model source label, preserving NGC 4151 defaults."""

    return str(case.get("source_name", "NGC4151"))


def _likelihood_nll(likelihood):
    statistic = likelihood.results.get_statistic_frame()["-log(likelihood)"]
    if "total" in statistic.index:
        return float(statistic.loc["total"])
    return float(statistic.sum())


def _bounded_start(parameter, value):
    """Set a finite fit start after clipping it to the parameter bounds."""

    value = float(value)
    if not np.isfinite(value):
        return
    if parameter.min_value is not None:
        value = max(value, float(parameter.min_value))
    if parameter.max_value is not None:
        value = min(value, float(parameter.max_value))
    parameter.value = value


def _install_cached_point_source_response(plugin, model, prepared):
    """Prime a new model folding with the case's precomputed sky response.

    The spectral expectation must still be recomputed as fit parameters change,
    but the expensive orientation/response convolution depends only on the fixed
    source position and can be shared by every fit and optimizer start.
    """

    cached_response = prepared.get("fit_point_source_response")
    if cached_response is None:
        return

    model_folding = plugin._response
    model_folding._cache_source_responses()
    for source_name, source_response in model_folding._source_responses.items():
        source = model.sources[source_name]
        source_response._psr = cached_response
        source_response._last_convolved_source_skycoord = source.position.sky_coord
        source_response._expectation = None
        source_response._last_convolved_source_dict = None

    # _cache_source_responses() records the current model dictionary before an
    # expectation has been computed. Invalidate only that dictionary cache so
    # the first likelihood call folds the new spectrum instead of returning the
    # initially empty expectation histogram. The installed sky response stays.
    model_folding._cached_model_dict = None
    model_folding._expectation.clear()


def _fit_model(
    case,
    data_histogram,
    prepared,
    *,
    include_secondary,
    background_only=False,
    initial_values=None,
    start_label="injected",
):
    initial_values = dict(initial_values or {})
    data_projection = data_histogram.project("Em", "Phi", "PsiChi")
    background_projection = prepared["background_expectation"].project("Em", "Phi", "PsiChi")

    if background_only:
        null_shape = Powerlaw()
        null_shape.K.value = 1e-30
        null_shape.K.fix = True
        null_shape.index.value = 1.0
        null_shape.index.fix = True
        source = PointSource(
            "source_background_only",
            l=float(case["longitude_deg"]),
            b=float(case["latitude_deg"]),
            spectral_shape=null_shape,
        )
        model = Model(source)
        link_function = None
        primary_shape = null_shape
        secondary_shape = None
        label = "background_only"
    else:
        fit_pivot_keV = float(case.get("fit_pivot_keV", 200.0))
        primary_spec = _fit_specification(case["primary"], fit_pivot_keV)
        primary_shape = _make_shape(primary_spec, fit=True)
        secondary_shape = None
        link_function = None
        label = "full" if include_secondary else "primary_only"

        if include_secondary and case.get("secondary") is not None:
            secondary_spec = _fit_specification(case["secondary"], fit_pivot_keV)
            secondary_shape = _make_shape(secondary_spec, fit=True)
            source = PointSource(
                "source_full",
                l=float(case["longitude_deg"]),
                b=float(case["latitude_deg"]),
                spectral_shape=primary_shape + secondary_shape,
            )
            model = Model(source)
            injected_link_ratio = float(secondary_spec["K"] / primary_spec["K"])
            link_function = Line(a=0.0, b=max(injected_link_ratio, 1e-10))
            link_function.a.fix = True
            link_function.b.min_value = 0.0
            model.link(
                model.source_full.spectrum.main.composite.K_2,
                model.source_full.spectrum.main.composite.K_1,
                link_function,
            )
        else:
            source = PointSource(
                "source_primary_only",
                l=float(case["longitude_deg"]),
                b=float(case["latitude_deg"]),
                spectral_shape=primary_shape,
            )
            model = Model(source)

        _bounded_start(primary_shape.K, initial_values.get("primary_K", primary_shape.K.value))
        if hasattr(primary_shape, "xc"):
            _bounded_start(
                primary_shape.xc,
                initial_values.get("primary_cutoff_keV", primary_shape.xc.value),
            )
        if secondary_shape is not None:
            _bounded_start(
                secondary_shape.index,
                initial_values.get("secondary_index", secondary_shape.index.value),
            )
        if link_function is not None:
            _bounded_start(
                link_function.b,
                initial_values.get("secondary_link_ratio", link_function.b.value),
            )

    background_parameter = _make_background_parameter(f"background_{label}")

    plugin = COSIPlugin(
        f"cosi_{label}",
        dr=prepared.get("response", prepared["response_path"]),
        data=data_projection,
        bkg=background_projection,
        sc_orientation=prepared["fit_orientation"],
        nuisance_param=background_parameter,
        earth_occ=True,
    )
    plugin.set_model(model)
    _install_cached_point_source_response(plugin, model, prepared)
    fitted_background_parameter = list(plugin.nuisance_parameters.values())[0]
    if "background_rate_hz" in initial_values:
        _bounded_start(
            fitted_background_parameter,
            initial_values["background_rate_hz"],
        )
    likelihood = JointLikelihood(model, DataList(plugin), verbose=False)
    likelihood.fit(compute_covariance=False, quiet=True)

    return {
        "nll": _likelihood_nll(likelihood),
        "start_label": str(start_label),
        "background_rate_hz": float(fitted_background_parameter.value),
        "primary_K": float(primary_shape.K.value),
        "primary_cutoff_keV": (
            float(primary_shape.xc.value) if hasattr(primary_shape, "xc") else np.nan
        ),
        "secondary_link_ratio": (
            float(link_function.b.value) if link_function is not None else np.nan
        ),
        "secondary_index": (
            float(secondary_shape.index.value) if secondary_shape is not None else np.nan
        ),
        "secondary_cutoff_keV": (
            float(secondary_shape.xc.value)
            if secondary_shape is not None and hasattr(secondary_shape, "xc")
            else np.nan
        ),
    }


def fit_case_statistic(case, data_histogram, prepared):
    if case["comparison"] == "source_detection":
        alternative = _fit_model(
            case,
            data_histogram,
            prepared,
            include_secondary=case.get("secondary") is not None,
        )
        reference = _fit_model(
            case,
            data_histogram,
            prepared,
            include_secondary=False,
            background_only=True,
        )
    elif case["comparison"] == "added_component":
        reference = _fit_model(
            case,
            data_histogram,
            prepared,
            include_secondary=False,
        )
        fit_strategy = str(case.get("fit_strategy", "single_start"))
        if fit_strategy == "single_start":
            alternatives = [
                _fit_model(
                    case,
                    data_histogram,
                    prepared,
                    include_secondary=True,
                )
            ]
        elif fit_strategy == "nested_single_start":
            fit_pivot_keV = float(case.get("fit_pivot_keV", 200.0))
            primary_spec = _fit_specification(case["primary"], fit_pivot_keV)
            secondary_spec = _fit_specification(case["secondary"], fit_pivot_keV)
            injected_link_ratio = float(secondary_spec["K"] / primary_spec["K"])
            alternatives = [
                _fit_model(
                    case,
                    data_histogram,
                    prepared,
                    include_secondary=True,
                    initial_values={
                        "primary_K": reference["primary_K"],
                        "primary_cutoff_keV": reference["primary_cutoff_keV"],
                        "background_rate_hz": reference["background_rate_hz"],
                        "secondary_link_ratio": max(injected_link_ratio, 1e-12),
                        "secondary_index": float(secondary_spec["index"]),
                    },
                    start_label="nested_cpl_plus_injected_tail",
                )
            ]
        elif fit_strategy == "nested_multistart":
            alternatives = []
            errors = []

            # This is the exact starting point used by the unfluctuated
            # comparison notebook: injected CPL, tail ratio, and tail index.
            try:
                alternatives.append(
                    _fit_model(
                        case,
                        data_histogram,
                        prepared,
                        include_secondary=True,
                        start_label="injected_comp_start",
                    )
                )
            except Exception as error:
                errors.append(f"injected_comp_start: {error!r}")

            fit_pivot_keV = float(case.get("fit_pivot_keV", 200.0))
            primary_spec = _fit_specification(case["primary"], fit_pivot_keV)
            secondary_spec = _fit_specification(case["secondary"], fit_pivot_keV)
            injected_link_ratio = float(secondary_spec["K"] / primary_spec["K"])
            default_tail_starts = (
                {"label": "nested_true_tail", "ratio_factor": 1.0, "index": secondary_spec["index"]},
                {"label": "nested_half_steep", "ratio_factor": 0.5, "index": -3.2},
                {"label": "nested_double_shallow", "ratio_factor": 2.0, "index": -2.4},
                {"label": "nested_near_null", "ratio_factor": 1e-6, "index": secondary_spec["index"]},
            )
            for start in case.get("multistart_tail_starts", default_tail_starts):
                initial_values = {
                    "primary_K": reference["primary_K"],
                    "primary_cutoff_keV": reference["primary_cutoff_keV"],
                    "background_rate_hz": reference["background_rate_hz"],
                    "secondary_link_ratio": max(
                        injected_link_ratio * float(start["ratio_factor"]),
                        1e-12,
                    ),
                    "secondary_index": float(start["index"]),
                }
                try:
                    alternatives.append(
                        _fit_model(
                            case,
                            data_histogram,
                            prepared,
                            include_secondary=True,
                            initial_values=initial_values,
                            start_label=start["label"],
                        )
                    )
                except Exception as error:
                    errors.append(f"{start['label']}: {error!r}")

            if not alternatives:
                raise RuntimeError(
                    "All CPL+PL multi-start fits failed: " + "; ".join(errors)
                )
        else:
            raise ValueError(f"Unsupported fit_strategy: {fit_strategy}")

        alternative = min(alternatives, key=lambda result: result["nll"])
    else:
        raise ValueError(f"Unsupported comparison: {case['comparison']}")

    raw_statistic = 2.0 * (reference["nll"] - alternative["nll"])
    return {
        "delta_ts": max(float(raw_statistic), 0.0),
        "raw_delta_ts": float(raw_statistic),
        "alternative_nll": alternative["nll"],
        "reference_nll": reference["nll"],
        "fit_strategy": str(case.get("fit_strategy", "single_start")),
        "fit_start_label": alternative["start_label"],
        "n_successful_starts": int(len(alternatives)) if case["comparison"] == "added_component" else 1,
        "fit_background_rate_hz": alternative["background_rate_hz"],
        "reference_background_rate_hz": reference["background_rate_hz"],
        "fit_primary_K": alternative["primary_K"],
        "fit_primary_cutoff_keV": alternative["primary_cutoff_keV"],
        "fit_secondary_link_ratio": alternative["secondary_link_ratio"],
        "fit_secondary_index": alternative["secondary_index"],
        "fit_secondary_cutoff_keV": alternative["secondary_cutoff_keV"],
    }


def prepare_case(
    case,
    *,
    exposure_months=24,
    response=None,
    point_response_cache=None,
):
    case = _resolve_case(case)
    response_path = Path(case.get("response_path", DEFAULT_RESPONSE_PATH))
    orientation_path = Path(case.get("orientation_path", DEFAULT_ORIENTATION_PATH))
    background_path = Path(case.get("background_path", DEFAULT_BACKGROUND_PATH))
    for required_path in (response_path, orientation_path, background_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    base_exposure_months = 3
    exposure_multiplier = float(exposure_months / base_exposure_months)
    source_coord = SkyCoord(
        l=float(case["longitude_deg"]),
        b=float(case["latitude_deg"]),
        frame="galactic",
        unit="deg",
    )
    orientation_3m = SpacecraftHistory.open(orientation_path)
    fov_cut = float(case.get("fov_cut_deg", 60.0)) * u.deg
    source_gti = GoodTimeInterval.from_pointing_cut(
        source_coord,
        orientation_3m,
        fov_cut,
        earth_occ=False,
    )
    orientation_fov_3m = orientation_3m.apply_gti(source_gti)
    fit_orientation = scale_spacecraft_livetime(
        orientation_fov_3m,
        exposure_multiplier,
    )

    background_expectation = (
        Histogram.open(background_path).project("Em", "Phi", "PsiChi")
        * exposure_multiplier
    )

    # Open the detector response once for this study. The same handle is used
    # both for source injection and by every likelihood plugin constructed by
    # the ensemble fits.
    if response is None:
        response = _open_response(response_path)
    if point_response_cache is None:
        point_response_cache = {}

    source_model = _make_injected_model(case)
    source = next(iter(source_model.point_sources.values()))

    # Build the modern likelihood point-source response before the injection
    # helper rotates its own response axes. Only the spectral convolution
    # changes during fitting, so this object can be safely shared by all seeds
    # and all multi-start fits at this position.
    fit_point_source_response = point_response_cache.get("fit")
    if fit_point_source_response is None:
        data_interface = EmCDSBinnedData(background_expectation)
        instrument_response = BinnedInstrumentResponse(response, data_interface)
        fit_scatt_map = fit_orientation.get_scatt_map(
            nside=2 * data_interface.axes["PsiChi"].nside,
            target_coord=source_coord,
            earth_occ=True,
        )
        fit_point_source_response = PointSourceResponse.from_scatt_map(
            source_coord,
            data_interface,
            instrument_response,
            fit_scatt_map,
            response.axes["Ei"],
            response.axes["Pol"] if "Pol" in response.axes.labels else None,
        )
        point_response_cache["fit"] = fit_point_source_response

    # Preserve the SourceInjector calculation used previously, but use the
    # already-open response rather than reopening the 417 MB HDF5 file.
    injection_point_source_response = point_response_cache.get("injection")
    if injection_point_source_response is None:
        injection_scatt_map = orientation_fov_3m.get_scatt_map(
            response.nside * 2,
            target_coord=source_coord,
            earth_occ=True,
        )
        injection_point_source_response = response.get_point_source_response(
            coord=source_coord,
            scatt_map=injection_scatt_map,
        )
        point_response_cache["injection"] = injection_point_source_response
    source_expectation_3m = injection_point_source_response.get_expectation(
        source.spectrum.main.shape,
    )
    em_axis = source_expectation_3m.axes["Em"].copy()
    em_axis.axis_scale = "log"
    source_expectation_3m.axes.set("Em", em_axis, copy=False)

    source_expectation = source_expectation_3m * exposure_multiplier
    source_expectation.axes["Em"].axis_scale = background_expectation.axes["Em"].axis_scale
    source_expectation = source_expectation.to(
        unit=background_expectation.unit,
        update=False,
    )
    total_expectation = source_expectation + background_expectation

    return {
        "case": case,
        "response": response,
        "response_path": response_path,
        "orientation_path": orientation_path,
        "background_path": background_path,
        "exposure_months": int(exposure_months),
        "exposure_multiplier": exposure_multiplier,
        "orientation_fov_3m": orientation_fov_3m,
        "fit_orientation": fit_orientation,
        "fit_point_source_response": fit_point_source_response,
        "background_expectation": background_expectation,
        "source_expectation": source_expectation,
        "total_expectation": total_expectation,
    }


def _trial_key(seed):
    return int(seed)


def _write_histogram_overwrite(histogram, output_file):
    if output_file.exists():
        output_file.unlink()
    histogram.write(output_file)


def _select_representative_by_medians(
    trials,
    *,
    median_delta_ts,
    secondary_column,
    secondary_label,
):
    """Select by Delta TS first and a recovered-parameter median second."""

    pool = trials.copy()
    if pool.empty:
        raise RuntimeError("No trials were available for representative selection.")

    secondary_values = (
        pd.to_numeric(pool[secondary_column], errors="coerce")
        if secondary_column in pool
        else pd.Series(np.nan, index=pool.index, dtype=float)
    )
    finite_secondary_values = secondary_values[np.isfinite(secondary_values)]
    median_secondary_value = (
        float(finite_secondary_values.median())
        if not finite_secondary_values.empty
        else np.nan
    )

    pool["delta_ts_distance_from_median"] = (
        pool["delta_ts"] - float(median_delta_ts)
    ).abs()
    minimum_delta_ts_distance = float(
        pool["delta_ts_distance_from_median"].min()
    )
    delta_ts_tolerance = max(
        np.finfo(float).eps,
        abs(float(median_delta_ts)) * 1e-12,
    )
    delta_ts_candidates = pool.loc[
        np.isclose(
            pool["delta_ts_distance_from_median"],
            minimum_delta_ts_distance,
            rtol=1e-10,
            atol=delta_ts_tolerance,
        )
    ].copy()

    secondary_distance_column = (
        f"{secondary_label}_distance_from_median"
    )
    if np.isfinite(median_secondary_value):
        delta_ts_candidates[secondary_distance_column] = (
            pd.to_numeric(
                delta_ts_candidates[secondary_column],
                errors="coerce",
            )
            - median_secondary_value
        ).abs()
    else:
        delta_ts_candidates[secondary_distance_column] = np.inf
    delta_ts_candidates[secondary_distance_column] = delta_ts_candidates[
        secondary_distance_column
    ].fillna(np.inf)

    representative_trial = (
        delta_ts_candidates.sort_values(
            [secondary_distance_column, "seed"],
            kind="mergesort",
        )
        .iloc[0]
    )
    representative_secondary_value = float(
        representative_trial.get(secondary_column, np.nan)
    )

    return {
        "secondary_column": secondary_column,
        "secondary_label": secondary_label,
        "secondary_distance_column": secondary_distance_column,
        "median_secondary_value": median_secondary_value,
        "delta_ts_candidates": delta_ts_candidates,
        "representative_trial": representative_trial,
        "representative_secondary_value": representative_secondary_value,
        "selection_method": (
            "closest to ensemble median Delta TS; equal-distance candidates "
            f"are ranked by closeness to the ensemble median {secondary_label}, "
            "then by lowest seed"
        ),
    }


def run_fixed_case_ensemble(
    case,
    *,
    n_seeds=100,
    seed_start=4100,
    exposure_months=24,
    target_delta_ts=9.0,
    resume_existing_trials=True,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    """Run one fixed spectral case and save its representative median trial."""

    prepared = prepare_case(case, exposure_months=exposure_months)
    case = prepared["case"]
    seeds = np.arange(int(seed_start), int(seed_start) + int(n_seeds), dtype=int)
    output_dir = Path(output_root) / case["case_tag"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = case.get("output_prefix", case["case_tag"])
    trials_csv = output_dir / f"{output_prefix}_trials_{exposure_months}months.csv"
    trial_rows = []
    if resume_existing_trials and trials_csv.exists():
        checkpoint = pd.read_csv(trials_csv)
        checkpoint = checkpoint.loc[
            checkpoint["exposure_months"] == int(exposure_months)
        ].copy()
        checkpoint["fit_ok"] = (
            checkpoint["fit_ok"].astype(str).str.lower().isin(("true", "1"))
        )
        trial_rows = checkpoint.to_dict(orient="records")
        print(f"Loaded {len(trial_rows)} checkpointed trials from {trials_csv}")

    completed = {
        _trial_key(row["seed"]): row
        for row in trial_rows
        if bool(row.get("fit_ok", False))
        and np.isfinite(row.get("delta_ts", np.nan))
    }

    for trial_number, seed in enumerate(seeds, start=1):
        key = _trial_key(seed)
        if key in completed:
            continue

        simulated_data = poisson_realization(prepared["total_expectation"], seed)
        row = {
            "case_tag": case["case_tag"],
            "comparison": case["comparison"],
            "seed": int(seed),
            "exposure_months": int(exposure_months),
            "fit_ok": True,
            "fit_error": "",
        }
        try:
            row.update(fit_case_statistic(case, simulated_data, prepared))
            if row["raw_delta_ts"] < -1e-3:
                row["fit_ok"] = False
                row["fit_error"] = (
                    "The alternative fit was worse than its nested reference fit."
                )
        except Exception as error:
            row.update(
                {
                    "fit_ok": False,
                    "fit_error": repr(error),
                    "delta_ts": np.nan,
                    "raw_delta_ts": np.nan,
                    "alternative_nll": np.nan,
                    "reference_nll": np.nan,
                    "fit_primary_K": np.nan,
                    "fit_primary_cutoff_keV": np.nan,
                    "fit_secondary_link_ratio": np.nan,
                    "fit_secondary_index": np.nan,
                    "fit_secondary_cutoff_keV": np.nan,
                }
            )

        trial_rows.append(row)
        if row["fit_ok"] and np.isfinite(row["delta_ts"]):
            completed[key] = row

        pd.DataFrame(trial_rows).sort_values("seed").drop_duplicates(
            "seed", keep="last"
        ).to_csv(trials_csv, index=False)

        if trial_number % 10 == 0 or trial_number == len(seeds):
            print(f"Completed {trial_number}/{len(seeds)} seeds")

    trial_results = (
        pd.DataFrame(trial_rows)
        .sort_values("seed")
        .drop_duplicates("seed", keep="last")
        .reset_index(drop=True)
    )
    requested_results = trial_results.loc[trial_results["seed"].isin(seeds)].copy()
    valid_trials = requested_results.loc[
        requested_results["fit_ok"].astype(bool)
        & np.isfinite(requested_results["delta_ts"])
    ].copy()
    if valid_trials.empty:
        raise RuntimeError("No successful likelihood fits were available.")

    median_delta_ts = float(valid_trials["delta_ts"].median())
    if case["comparison"] == "source_detection":
        secondary_column = "fit_primary_cutoff_keV"
        secondary_label = "recovered_cutoff_keV"
    else:
        secondary_column = "fit_secondary_link_ratio"
        secondary_label = "recovered_norm_nt"
    representative_selection = _select_representative_by_medians(
        valid_trials,
        median_delta_ts=median_delta_ts,
        secondary_column=secondary_column,
        secondary_label=secondary_label,
    )
    delta_ts_candidates = representative_selection["delta_ts_candidates"]
    representative_trial = representative_selection["representative_trial"]
    median_secondary_value = representative_selection["median_secondary_value"]
    representative_secondary_value = representative_selection[
        "representative_secondary_value"
    ]
    representative_seed = int(representative_trial["seed"])
    detection_fraction = float(
        (valid_trials["delta_ts"] >= float(target_delta_ts)).mean()
    )
    q16 = float(valid_trials["delta_ts"].quantile(0.16))
    q84 = float(valid_trials["delta_ts"].quantile(0.84))

    summary = pd.DataFrame(
        [
            {
                "case_tag": case["case_tag"],
                "comparison": case["comparison"],
                "exposure_months": int(exposure_months),
                "n_requested_seeds": int(n_seeds),
                "n_successful_fits": int(len(valid_trials)),
                "median_delta_ts": median_delta_ts,
                "delta_ts_q16": q16,
                "delta_ts_q84": q84,
                "target_delta_ts": float(target_delta_ts),
                "fraction_delta_ts_ge_target": detection_fraction,
                "representative_secondary_criterion": secondary_label,
                f"median_{secondary_label}": median_secondary_value,
                "representative_seed": representative_seed,
                "representative_delta_ts": float(representative_trial["delta_ts"]),
                f"representative_{secondary_label}": (
                    representative_secondary_value
                ),
            }
        ]
    )
    summary_csv = output_dir / f"{output_prefix}_summary_{exposure_months}months.csv"
    summary.to_csv(summary_csv, index=False)

    representative_data = poisson_realization(
        prepared["total_expectation"],
        representative_seed,
    )
    file_detail = case.get("file_detail", "")
    output_suffix = (
        f"{case['file_stem']}_{exposure_months}months{file_detail}_medianSeed_"
        f"{representative_seed}"
    )
    source_label = _source_name(case)
    source_output_file = output_dir / f"{source_label}_{output_suffix}.hdf5"
    data_output_file = output_dir / f"{source_label}_plus_bkg_{output_suffix}.hdf5"
    _write_histogram_overwrite(prepared["source_expectation"], source_output_file)
    _write_histogram_overwrite(representative_data, data_output_file)

    statistic_label = (
        "source detection TS"
        if case["comparison"] == "source_detection"
        else "added-component Delta TS"
    )
    manifest = {
        "case_tag": case["case_tag"],
        "case_description": case["description"],
        "comparison": case["comparison"],
        "statistic_label": statistic_label,
        "exposure_months": int(exposure_months),
        "exposure_multiplier": prepared["exposure_multiplier"],
        "n_requested_seeds": int(n_seeds),
        "seed_start": int(seed_start),
        "target_delta_ts": float(target_delta_ts),
        "median_delta_ts": median_delta_ts,
        "delta_ts_q16": q16,
        "delta_ts_q84": q84,
        "fraction_delta_ts_ge_target": detection_fraction,
        "representative_selection_method": representative_selection[
            "selection_method"
        ],
        "representative_secondary_criterion": secondary_label,
        "representative_secondary_column": secondary_column,
        f"median_{secondary_label}": (
            median_secondary_value
            if np.isfinite(median_secondary_value)
            else None
        ),
        "delta_ts_primary_candidate_count": int(len(delta_ts_candidates)),
        "representative_seed": representative_seed,
        "representative_delta_ts": float(representative_trial["delta_ts"]),
        "representative_delta_ts_distance_from_median": float(
            representative_trial["delta_ts_distance_from_median"]
        ),
        f"representative_{secondary_label}": (
            representative_secondary_value
            if np.isfinite(representative_secondary_value)
            else None
        ),
        f"representative_{secondary_label}_distance_from_median": float(
            representative_trial[
                representative_selection["secondary_distance_column"]
            ]
        ) if np.isfinite(
            representative_trial[
                representative_selection["secondary_distance_column"]
            ]
        ) else None,
        "source_expectation_file": str(source_output_file),
        "median_data_file": str(data_output_file),
        "background_file_3m": str(prepared["background_path"]),
        "background_exposure_multiplier": prepared["exposure_multiplier"],
        "response_file": str(prepared["response_path"]),
        "orientation_file_3m": str(prepared["orientation_path"]),
        "trials_csv": str(trials_csv),
        "summary_csv": str(summary_csv),
        "resolved_case": case,
    }
    manifest_file = output_dir / (
        f"{output_prefix}_median_realization_{exposure_months}months.json"
    )
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Median {statistic_label}: {median_delta_ts:.3f}")
    print(f"Central 68% range: {q16:.3f}-{q84:.3f}")
    print(f"Fraction >= {target_delta_ts:g}: {detection_fraction:.3f}")
    print(f"Median {secondary_label}: {median_secondary_value:.5g}")
    print(f"Delta-TS-primary candidates: {len(delta_ts_candidates)}")
    print(f"Representative seed: {representative_seed}")
    print(f"Representative statistic: {representative_trial['delta_ts']:.3f}")
    print(
        f"Representative {secondary_label}: "
        f"{representative_secondary_value:.5g}"
    )
    print(f"Saved median data: {data_output_file}")
    print(f"Saved manifest: {manifest_file}")

    return {
        "case": case,
        "prepared": prepared,
        "trials": requested_results,
        "valid_trials": valid_trials,
        "summary": summary,
        "delta_ts_candidates": delta_ts_candidates,
        "representative_secondary_criterion": secondary_label,
        "median_secondary_value": median_secondary_value,
        "representative_trial": representative_trial,
        "source_output_file": source_output_file,
        "data_output_file": data_output_file,
        "manifest_file": manifest_file,
        "manifest": manifest,
    }


def _case_with_secondary_norm(case, norm_nt):
    updated = copy.deepcopy(case)
    if updated.get("secondary") is None:
        raise ValueError("A threshold scan requires a secondary spectral component.")
    updated["secondary"]["normalization"]["ratio"] = float(norm_nt)
    return updated


def run_unfluctuated_norm_scan(case, norm_nt_grid, *, exposure_months=24):
    """Evaluate added-component Delta TS on expected, unfluctuated counts.

    Every grid point uses ``total_expectation`` directly rather than drawing a
    Poisson realization.  The fit strategy and parameter bounds are taken from
    ``case``, so this provides an apples-to-apples Asimov comparison before an
    ensemble is run.
    """

    norm_nt_grid = np.asarray(norm_nt_grid, dtype=float)
    if norm_nt_grid.ndim != 1 or len(norm_nt_grid) == 0:
        raise ValueError("norm_nt_grid must contain at least one value.")
    if np.any(~np.isfinite(norm_nt_grid)) or np.any(norm_nt_grid < 0):
        raise ValueError("norm_nt_grid must contain finite non-negative values.")
    if case.get("secondary") is None or case.get("comparison") != "added_component":
        raise ValueError("An unfluctuated norm scan requires an added-component case.")

    shared_response = _open_response(
        Path(case.get("response_path", DEFAULT_RESPONSE_PATH))
    )
    point_response_cache = {}
    rows = []
    for norm_nt in norm_nt_grid:
        case_at_norm = _case_with_secondary_norm(case, norm_nt)
        prepared = prepare_case(
            case_at_norm,
            exposure_months=exposure_months,
            response=shared_response,
            point_response_cache=point_response_cache,
        )
        row = {
            "norm_nt": float(norm_nt),
            "exposure_months": int(exposure_months),
            "fit_ok": True,
            "fit_error": "",
        }
        try:
            row.update(
                fit_case_statistic(
                    prepared["case"],
                    prepared["total_expectation"],
                    prepared,
                )
            )
            if row["raw_delta_ts"] < -1e-3:
                row["fit_ok"] = False
                row["fit_error"] = (
                    "The alternative fit was worse than its nested reference fit."
                )
        except Exception as error:
            row.update(
                {
                    "fit_ok": False,
                    "fit_error": repr(error),
                    "delta_ts": np.nan,
                    "raw_delta_ts": np.nan,
                }
            )
        rows.append(row)
        if row["fit_ok"]:
            print(f"norm_nt={norm_nt:g}: unfluctuated Delta TS={row['delta_ts']:.4f}")
        else:
            print(f"norm_nt={norm_nt:g}: fit failed: {row['fit_error']}")

    scan_results = pd.DataFrame(rows).sort_values("norm_nt").reset_index(drop=True)
    valid_results = scan_results.loc[
        scan_results["fit_ok"].astype(bool)
        & np.isfinite(scan_results["delta_ts"])
    ].copy()
    if valid_results.empty:
        raise RuntimeError("No successful unfluctuated norm-scan fits were available.")

    best_index = valid_results["delta_ts"].idxmax()
    best_row = scan_results.loc[best_index].copy()
    return {
        "case": copy.deepcopy(case),
        "exposure_months": int(exposure_months),
        "scan_results": scan_results,
        "best_row": best_row,
        "best_norm_nt": float(best_row["norm_nt"]),
        "best_delta_ts": float(best_row["delta_ts"]),
    }


def evaluate_injected_components(case, norm_nt, energy_keV):
    """Evaluate the injected primary, secondary, and total photon spectra."""

    energy_keV = np.asarray(energy_keV, dtype=float)
    if energy_keV.ndim != 1 or len(energy_keV) == 0:
        raise ValueError("energy_keV must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(energy_keV)) or np.any(energy_keV <= 0):
        raise ValueError("energy_keV must contain finite positive values.")

    resolved_case = _resolve_case(_case_with_secondary_norm(case, norm_nt))
    primary_shape = _make_shape(resolved_case["primary"], fit=False)
    secondary_shape = _make_shape(resolved_case["secondary"], fit=False)
    primary = np.asarray(
        [primary_shape.evaluate_at(energy) for energy in energy_keV],
        dtype=float,
    )
    secondary = np.asarray(
        [secondary_shape.evaluate_at(energy) for energy in energy_keV],
        dtype=float,
    )
    return {
        "energy_keV": energy_keV,
        "primary": primary,
        "secondary": secondary,
        "total": primary + secondary,
    }


def run_norm_threshold_ensemble(
    case,
    norm_nt_grid,
    *,
    n_seeds=100,
    seed_start=4100,
    exposure_months=24,
    target_delta_ts=9.0,
    resume_existing_trials=True,
    require_target_reached=True,
    checkpoint_every=10,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    """Scan an ensemble for the secondary normalization reaching a Delta-TS target.

    By default the scan raises when the tested grid does not reach the target.
    Set ``require_target_reached=False`` to retain the ensemble summary and save
    the representative realization at the highest tested median Delta TS.
    Newly evaluated trials are saved atomically every ``checkpoint_every`` seeds.
    """

    norm_nt_grid = np.asarray(norm_nt_grid, dtype=float)
    if norm_nt_grid.ndim != 1 or len(norm_nt_grid) == 0:
        raise ValueError("norm_nt_grid must contain at least one value.")
    if np.any(~np.isfinite(norm_nt_grid)) or np.any(norm_nt_grid < 0):
        raise ValueError("norm_nt_grid must contain finite non-negative values.")
    checkpoint_every = int(checkpoint_every)
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be a positive integer.")

    seeds = np.arange(int(seed_start), int(seed_start) + int(n_seeds), dtype=int)
    output_dir = Path(output_root) / case["case_tag"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = case.get("output_prefix", case["case_tag"])
    trials_csv = output_dir / f"{output_prefix}_trials_{exposure_months}months.csv"

    trial_rows = []
    fit_strategy = str(case.get("fit_strategy", "single_start"))
    if resume_existing_trials and trials_csv.exists():
        checkpoint = pd.read_csv(trials_csv)
        checkpoint = checkpoint.loc[
            checkpoint["exposure_months"] == int(exposure_months)
        ].copy()
        checkpoint["fit_ok"] = (
            checkpoint["fit_ok"].astype(str).str.lower().isin(("true", "1"))
        )
        if "fit_strategy" not in checkpoint.columns:
            checkpoint["fit_strategy"] = "legacy_single_start"
        checkpoint["fit_strategy"] = checkpoint["fit_strategy"].fillna(
            "legacy_single_start"
        )
        trial_rows = checkpoint.to_dict(orient="records")
        print(f"Loaded {len(trial_rows)} checkpointed trials from {trials_csv}")

    def trial_key(norm_nt, seed, strategy):
        return (round(float(norm_nt), 12), int(seed), str(strategy))

    completed = {
        trial_key(row["norm_nt"], row["seed"], row.get("fit_strategy")): row
        for row in trial_rows
        if bool(row.get("fit_ok", False))
        and np.isfinite(row.get("delta_ts", np.nan))
    }

    def write_trial_checkpoint(rows):
        normalized_rows = (
            pd.DataFrame(rows)
            .sort_values(["norm_nt", "seed", "fit_strategy"])
            .drop_duplicates(["norm_nt", "seed", "fit_strategy"], keep="last")
        )
        temporary_csv = trials_csv.with_name(f".{trials_csv.name}.tmp")
        normalized_rows.to_csv(temporary_csv, index=False)
        temporary_csv.replace(trials_csv)
        return normalized_rows.to_dict(orient="records")

    shared_response = _open_response(
        Path(case.get("response_path", DEFAULT_RESPONSE_PATH))
    )
    point_response_cache = {}
    prepared_by_norm = {}
    for norm_nt in norm_nt_grid:
        case_at_norm = _case_with_secondary_norm(case, norm_nt)
        prepared = prepare_case(
            case_at_norm,
            exposure_months=exposure_months,
            response=shared_response,
            point_response_cache=point_response_cache,
        )
        prepared_by_norm[round(float(norm_nt), 12)] = prepared
        trials_since_checkpoint = 0

        for trial_number, seed in enumerate(seeds, start=1):
            key = trial_key(norm_nt, seed, fit_strategy)
            if key in completed:
                continue

            simulated_data = poisson_realization(prepared["total_expectation"], seed)
            row = {
                "case_tag": case["case_tag"],
                "comparison": "added_component",
                "norm_nt": float(norm_nt),
                "seed": int(seed),
                "exposure_months": int(exposure_months),
                "fit_strategy": fit_strategy,
                "fit_ok": True,
                "fit_error": "",
            }
            try:
                row.update(
                    fit_case_statistic(prepared["case"], simulated_data, prepared)
                )
                if row["raw_delta_ts"] < -1e-3:
                    row["fit_ok"] = False
                    row["fit_error"] = (
                        "The alternative fit was worse than its nested reference fit."
                    )
            except Exception as error:
                row.update(
                    {
                        "fit_ok": False,
                        "fit_error": repr(error),
                        "delta_ts": np.nan,
                        "raw_delta_ts": np.nan,
                        "alternative_nll": np.nan,
                        "reference_nll": np.nan,
                        "fit_primary_K": np.nan,
                        "fit_primary_cutoff_keV": np.nan,
                        "fit_secondary_link_ratio": np.nan,
                        "fit_secondary_index": np.nan,
                        "fit_secondary_cutoff_keV": np.nan,
                    }
                )

            trial_rows.append(row)
            trials_since_checkpoint += 1
            if row["fit_ok"] and np.isfinite(row["delta_ts"]):
                completed[key] = row

            if trials_since_checkpoint >= checkpoint_every:
                trial_rows = write_trial_checkpoint(trial_rows)
                trials_since_checkpoint = 0
                print(
                    f"norm_nt={norm_nt:g}: checkpoint saved at "
                    f"{trial_number}/{len(seeds)} seeds"
                )

            if trial_number % 10 == 0 or trial_number == len(seeds):
                print(
                    f"norm_nt={norm_nt:g}: completed "
                    f"{trial_number}/{len(seeds)} seeds"
                )

        trial_rows = write_trial_checkpoint(trial_rows)

    trial_results = pd.DataFrame(trial_rows)
    requested = trial_results.loc[
        trial_results["seed"].isin(seeds)
        & trial_results["fit_strategy"].eq(fit_strategy)
        & trial_results["norm_nt"].apply(
            lambda value: np.any(np.isclose(float(value), norm_nt_grid))
        )
    ].copy()
    valid_trials = requested.loc[
        requested["fit_ok"].astype(bool)
        & np.isfinite(requested["delta_ts"])
    ].copy()
    if valid_trials.empty:
        raise RuntimeError("No successful threshold-scan fits were available.")

    threshold_summary = (
        valid_trials.groupby("norm_nt", as_index=False)
        .agg(
            n_successful_fits=("delta_ts", "size"),
            median_delta_ts=("delta_ts", "median"),
            delta_ts_q16=("delta_ts", lambda values: values.quantile(0.16)),
            delta_ts_q84=("delta_ts", lambda values: values.quantile(0.84)),
            median_recovered_norm_nt=("fit_secondary_link_ratio", "median"),
            recovered_norm_nt_q16=(
                "fit_secondary_link_ratio",
                lambda values: values.quantile(0.16),
            ),
            recovered_norm_nt_q84=(
                "fit_secondary_link_ratio",
                lambda values: values.quantile(0.84),
            ),
            fraction_delta_ts_ge_target=(
                "delta_ts",
                lambda values: float((values >= target_delta_ts).mean()),
            ),
        )
        .sort_values("norm_nt")
        .reset_index(drop=True)
    )
    requested_counts = (
        requested.groupby("norm_nt", as_index=False)
        .size()
        .rename(columns={"size": "n_requested_fits"})
    )
    threshold_summary = threshold_summary.merge(
        requested_counts,
        on="norm_nt",
        how="left",
    )
    threshold_summary["n_failed_fits"] = (
        threshold_summary["n_requested_fits"]
        - threshold_summary["n_successful_fits"]
    )
    passing = threshold_summary.loc[
        threshold_summary["median_delta_ts"] >= float(target_delta_ts)
    ]
    target_reached = not passing.empty
    if passing.empty:
        if require_target_reached:
            raise RuntimeError(
                "The tested norm_nt grid does not reach the target median Delta TS. "
                "Extend the grid and rerun."
            )
        selected_summary = threshold_summary.loc[
            threshold_summary["median_delta_ts"].idxmax()
        ]
        selection_mode = "highest_tested_median_delta_ts"
    else:
        selected_summary = passing.iloc[0]
        selection_mode = "lowest_tested_norm_reaching_target"

    selected_norm_nt = float(selected_summary["norm_nt"])
    selected_median_delta_ts = float(selected_summary["median_delta_ts"])
    if not target_reached:
        interpolated_threshold = np.nan
    else:
        lower = threshold_summary.loc[
            (threshold_summary["norm_nt"] < selected_norm_nt)
            & (threshold_summary["median_delta_ts"] < float(target_delta_ts))
        ]
        if lower.empty:
            interpolated_threshold = selected_norm_nt
        else:
            lower_summary = lower.iloc[-1]
            interpolated_threshold = float(
                np.interp(
                    target_delta_ts,
                    [lower_summary["median_delta_ts"], selected_median_delta_ts],
                    [lower_summary["norm_nt"], selected_norm_nt],
                )
            )

    selected_trials = valid_trials.loc[
        np.isclose(valid_trials["norm_nt"], selected_norm_nt)
    ].copy()
    representative_selection = _select_representative_by_medians(
        selected_trials,
        median_delta_ts=selected_median_delta_ts,
        secondary_column="fit_secondary_link_ratio",
        secondary_label="recovered_norm_nt",
    )
    delta_ts_candidates = representative_selection["delta_ts_candidates"]
    representative_trial = representative_selection["representative_trial"]
    median_recovered_norm_nt = representative_selection[
        "median_secondary_value"
    ]
    representative_recovered_norm_nt = representative_selection[
        "representative_secondary_value"
    ]
    representative_seed = int(representative_trial["seed"])

    summary_csv = output_dir / f"{output_prefix}_summary_{exposure_months}months.csv"
    threshold_summary.to_csv(summary_csv, index=False)

    selected_prepared = prepared_by_norm[round(selected_norm_nt, 12)]
    representative_data = poisson_realization(
        selected_prepared["total_expectation"],
        representative_seed,
    )
    norm_tag = f"{selected_norm_nt:.4g}".replace(".", "p")
    selection_tag = "threshold" if target_reached else "bestMedian"
    output_suffix = (
        f"{case['file_stem']}_{exposure_months}months_{selection_tag}_normNT_"
        f"{norm_tag}_medianSeed_{representative_seed}"
    )
    source_label = _source_name(case)
    source_output_file = output_dir / f"{source_label}_{output_suffix}.hdf5"
    data_output_file = output_dir / f"{source_label}_plus_bkg_{output_suffix}.hdf5"
    _write_histogram_overwrite(
        selected_prepared["source_expectation"],
        source_output_file,
    )
    _write_histogram_overwrite(representative_data, data_output_file)

    manifest = {
        "case_tag": case["case_tag"],
        "case_description": case["description"],
        "comparison": "added_component",
        "statistic_label": "added-component Delta TS",
        "exposure_months": int(exposure_months),
        "exposure_multiplier": selected_prepared["exposure_multiplier"],
        "target_delta_ts": float(target_delta_ts),
        "target_reached": bool(target_reached),
        "selection_mode": selection_mode,
        "selected_norm_nt": selected_norm_nt,
        "grid_threshold_norm_nt": selected_norm_nt,
        "interpolated_threshold_norm_nt": (
            float(interpolated_threshold)
            if np.isfinite(interpolated_threshold)
            else None
        ),
        "median_delta_ts_at_grid_threshold": selected_median_delta_ts,
        "detection_fraction_at_grid_threshold": float(
            selected_summary["fraction_delta_ts_ge_target"]
        ),
        "representative_selection_method": representative_selection[
            "selection_method"
        ],
        "representative_secondary_criterion": "recovered_norm_nt",
        "representative_secondary_column": "fit_secondary_link_ratio",
        "median_recovered_norm_nt": (
            median_recovered_norm_nt
            if np.isfinite(median_recovered_norm_nt)
            else None
        ),
        "delta_ts_primary_candidate_count": int(len(delta_ts_candidates)),
        "representative_seed": representative_seed,
        "representative_delta_ts": float(representative_trial["delta_ts"]),
        "representative_delta_ts_distance_from_median": float(
            representative_trial["delta_ts_distance_from_median"]
        ),
        "representative_recovered_norm_nt": (
            representative_recovered_norm_nt
            if np.isfinite(representative_recovered_norm_nt)
            else None
        ),
        "representative_recovered_norm_nt_distance_from_median": float(
            representative_trial["recovered_norm_nt_distance_from_median"]
        ) if np.isfinite(
            representative_trial["recovered_norm_nt_distance_from_median"]
        ) else None,
        "n_seeds_per_normalization": int(n_seeds),
        "seed_start": int(seed_start),
        "norm_nt_grid": [float(value) for value in norm_nt_grid],
        "source_expectation_file": str(source_output_file),
        "median_data_file": str(data_output_file),
        "background_file_3m": str(selected_prepared["background_path"]),
        "background_exposure_multiplier": selected_prepared["exposure_multiplier"],
        "response_file": str(selected_prepared["response_path"]),
        "orientation_file_3m": str(selected_prepared["orientation_path"]),
        "trials_csv": str(trials_csv),
        "summary_csv": str(summary_csv),
        "resolved_case_at_threshold": selected_prepared["case"],
    }
    manifest_file = output_dir / (
        f"{output_prefix}_median_realization_{exposure_months}months.json"
    )
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    if target_reached:
        print(f"Grid threshold norm_nt: {selected_norm_nt:g}")
        print(f"Interpolated threshold norm_nt: {interpolated_threshold:.5g}")
        print(f"Median Delta TS at grid threshold: {selected_median_delta_ts:.3f}")
    else:
        print("Target median Delta TS was not reached by the tested grid.")
        print(f"Highest-median tested norm_nt: {selected_norm_nt:g}")
        print(f"Highest tested median Delta TS: {selected_median_delta_ts:.3f}")
    print(f"Median recovered norm_nt: {median_recovered_norm_nt:.5g}")
    print(f"Delta-TS-primary candidates: {len(delta_ts_candidates)}")
    print(f"Representative seed: {representative_seed}")
    print(
        "Representative recovered norm_nt: "
        f"{representative_recovered_norm_nt:.5g}"
    )
    print(f"Saved median data: {data_output_file}")
    print(f"Saved manifest: {manifest_file}")

    return {
        "case": case,
        "trials": requested,
        "valid_trials": valid_trials,
        "threshold_summary": threshold_summary,
        "selected_trials": selected_trials,
        "delta_ts_candidates": delta_ts_candidates,
        "target_reached": bool(target_reached),
        "selection_mode": selection_mode,
        "selected_norm_nt": selected_norm_nt,
        "interpolated_threshold_norm_nt": interpolated_threshold,
        "selected_median_delta_ts": selected_median_delta_ts,
        "median_recovered_norm_nt": median_recovered_norm_nt,
        "representative_trial": representative_trial,
        "source_output_file": source_output_file,
        "data_output_file": data_output_file,
        "manifest_file": manifest_file,
        "manifest": manifest,
    }


def save_threshold_companion_exposure(
    threshold_result,
    *,
    exposure_months,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    """Save another exposure for a threshold selected by an existing ensemble.

    This does not run likelihood fits or select a new median.  It reuses the
    selected ``norm_nt`` and representative seed from ``threshold_result``,
    rebuilds the same injected spectrum at the requested exposure, and saves
    both the source expectation and source-plus-background realization.
    """

    case = threshold_result["case"]
    selected_norm_nt = float(threshold_result["selected_norm_nt"])
    representative_seed = int(threshold_result["representative_trial"]["seed"])
    selection_manifest = threshold_result["manifest"]
    selection_exposure_months = int(selection_manifest["exposure_months"])

    selected_case = _case_with_secondary_norm(case, selected_norm_nt)
    prepared = prepare_case(selected_case, exposure_months=exposure_months)
    representative_data = poisson_realization(
        prepared["total_expectation"],
        representative_seed,
    )

    output_dir = Path(output_root) / case["case_tag"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = case.get("output_prefix", case["case_tag"])
    norm_tag = f"{selected_norm_nt:.4g}".replace(".", "p")
    output_suffix = (
        f"{case['file_stem']}_{int(exposure_months)}months_threshold_normNT_"
        f"{norm_tag}_medianSeed_{representative_seed}"
    )
    source_label = _source_name(case)
    source_output_file = output_dir / f"{source_label}_{output_suffix}.hdf5"
    data_output_file = output_dir / f"{source_label}_plus_bkg_{output_suffix}.hdf5"
    _write_histogram_overwrite(prepared["source_expectation"], source_output_file)
    _write_histogram_overwrite(representative_data, data_output_file)

    manifest = {
        "case_tag": case["case_tag"],
        "case_description": case["description"],
        "comparison": "added_component",
        "statistic_label": "added-component Delta TS",
        "exposure_months": int(exposure_months),
        "exposure_multiplier": prepared["exposure_multiplier"],
        "is_companion_exposure": True,
        "selection_exposure_months": selection_exposure_months,
        "selection_manifest_file": str(threshold_result["manifest_file"]),
        "n_seeds_at_selection_exposure": selection_manifest.get(
            "n_seeds_per_normalization"
        ),
        "grid_threshold_norm_nt": selected_norm_nt,
        "representative_selection_method": selection_manifest.get(
            "representative_selection_method"
        ),
        "representative_secondary_criterion": selection_manifest.get(
            "representative_secondary_criterion"
        ),
        "median_recovered_norm_nt_at_selection_exposure": (
            selection_manifest.get("median_recovered_norm_nt")
        ),
        "representative_seed": representative_seed,
        "representative_delta_ts_at_selection_exposure": float(
            threshold_result["representative_trial"]["delta_ts"]
        ),
        "representative_recovered_norm_nt_at_selection_exposure": (
            selection_manifest.get("representative_recovered_norm_nt")
        ),
        "source_expectation_file": str(source_output_file),
        "median_data_file": str(data_output_file),
        "background_file_3m": str(prepared["background_path"]),
        "background_exposure_multiplier": prepared["exposure_multiplier"],
        "response_file": str(prepared["response_path"]),
        "orientation_file_3m": str(prepared["orientation_path"]),
        "resolved_case_at_threshold": prepared["case"],
    }
    manifest_file = output_dir / (
        f"{output_prefix}_companion_realization_{int(exposure_months)}months.json"
    )
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Saved {int(exposure_months)}-month companion using the "
        f"{selection_exposure_months}-month median seed {representative_seed}."
    )
    print(f"Saved companion source: {source_output_file}")
    print(f"Saved companion data: {data_output_file}")
    print(f"Saved companion manifest: {manifest_file}")

    return {
        "case": prepared["case"],
        "prepared": prepared,
        "source_output_file": source_output_file,
        "data_output_file": data_output_file,
        "manifest_file": manifest_file,
        "manifest": manifest,
    }


def save_fixed_case_companion_exposure(
    fixed_result,
    *,
    exposure_months,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    """Save another exposure using a fixed case's selected representative seed.

    No likelihood ensemble is run at the companion exposure.  The injected
    spectral parameters and representative seed are taken from ``fixed_result``
    (normally the 24-month run), while the source and background expectations
    are rebuilt for ``exposure_months``.  The ``selectedAt`` filename tag keeps
    this companion distinct from an independently selected median realization
    at the same exposure.
    """

    case = copy.deepcopy(fixed_result["case"])
    representative_seed = int(fixed_result["representative_trial"]["seed"])
    selection_manifest = fixed_result["manifest"]
    selection_exposure_months = int(selection_manifest["exposure_months"])

    prepared = prepare_case(case, exposure_months=exposure_months)
    representative_data = poisson_realization(
        prepared["total_expectation"],
        representative_seed,
    )

    output_dir = Path(output_root) / case["case_tag"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = case.get("output_prefix", case["case_tag"])
    file_detail = case.get("file_detail", "")
    output_suffix = (
        f"{case['file_stem']}_{int(exposure_months)}months{file_detail}_"
        f"selectedAt{selection_exposure_months}months_medianSeed_"
        f"{representative_seed}"
    )
    source_label = _source_name(case)
    source_output_file = output_dir / f"{source_label}_{output_suffix}.hdf5"
    data_output_file = output_dir / f"{source_label}_plus_bkg_{output_suffix}.hdf5"
    _write_histogram_overwrite(prepared["source_expectation"], source_output_file)
    _write_histogram_overwrite(representative_data, data_output_file)

    secondary_label = selection_manifest.get(
        "representative_secondary_criterion",
        "recovered_parameter",
    )
    manifest = {
        "case_tag": case["case_tag"],
        "case_description": case["description"],
        "comparison": case["comparison"],
        "statistic_label": selection_manifest.get("statistic_label"),
        "exposure_months": int(exposure_months),
        "exposure_multiplier": prepared["exposure_multiplier"],
        "is_companion_exposure": True,
        "selection_exposure_months": selection_exposure_months,
        "selection_manifest_file": str(fixed_result["manifest_file"]),
        "n_seeds_at_selection_exposure": selection_manifest.get(
            "n_requested_seeds"
        ),
        "representative_selection_method": selection_manifest.get(
            "representative_selection_method"
        ),
        "representative_secondary_criterion": secondary_label,
        f"median_{secondary_label}_at_selection_exposure": (
            fixed_result.get("median_secondary_value")
        ),
        "representative_seed": representative_seed,
        "representative_delta_ts_at_selection_exposure": float(
            fixed_result["representative_trial"]["delta_ts"]
        ),
        f"representative_{secondary_label}_at_selection_exposure": (
            selection_manifest.get(f"representative_{secondary_label}")
        ),
        "source_expectation_file": str(source_output_file),
        "median_data_file": str(data_output_file),
        "background_file_3m": str(prepared["background_path"]),
        "background_exposure_multiplier": prepared["exposure_multiplier"],
        "response_file": str(prepared["response_path"]),
        "orientation_file_3m": str(prepared["orientation_path"]),
        "resolved_case": prepared["case"],
    }
    manifest_file = output_dir / (
        f"{output_prefix}_companion_from_{selection_exposure_months}months_"
        f"realization_{int(exposure_months)}months.json"
    )
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Saved {int(exposure_months)}-month companion using the "
        f"{selection_exposure_months}-month median seed {representative_seed}."
    )
    print(f"Saved companion source: {source_output_file}")
    print(f"Saved companion data: {data_output_file}")
    print(f"Saved companion manifest: {manifest_file}")

    return {
        "case": prepared["case"],
        "prepared": prepared,
        "source_output_file": source_output_file,
        "data_output_file": data_output_file,
        "manifest_file": manifest_file,
        "manifest": manifest,
    }


def plot_norm_threshold_ensemble(result):
    summary = result["threshold_summary"]
    manifest = result["manifest"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(
        summary["norm_nt"],
        summary["median_delta_ts"],
        marker="o",
        color="#D55E00",
    )
    axes[0].fill_between(
        summary["norm_nt"],
        summary["delta_ts_q16"],
        summary["delta_ts_q84"],
        color="#D55E00",
        alpha=0.25,
        label="16th-84th percentile",
    )
    axes[0].axhline(
        manifest["target_delta_ts"],
        color="black",
        linestyle="--",
        label=f"Target = {manifest['target_delta_ts']:g}",
    )
    axes[0].axvline(
        manifest["grid_threshold_norm_nt"],
        color="0.4",
        linestyle=":",
        label=f"Grid threshold = {manifest['grid_threshold_norm_nt']:g}",
    )
    axes[0].set_xlabel("Injected secondary-component norm_nt")
    axes[0].set_ylabel("Delta TS")
    axes[0].legend()

    axes[1].plot(
        summary["norm_nt"],
        summary["fraction_delta_ts_ge_target"],
        marker="o",
        color="#0072B2",
    )
    axes[1].axhline(0.5, color="black", linestyle="--")
    axes[1].axvline(
        manifest["grid_threshold_norm_nt"],
        color="0.4",
        linestyle=":",
    )
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("Injected secondary-component norm_nt")
    axes[1].set_ylabel(
        f"Fraction with Delta TS >= {manifest['target_delta_ts']:g}"
    )
    return fig, axes


def plot_fixed_case_ensemble(result, *, bins=20):
    valid_trials = result["valid_trials"]
    manifest = result["manifest"]
    representative = result["representative_trial"]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(valid_trials["delta_ts"], bins=bins, color="0.75", edgecolor="0.25")
    ax.axvline(
        manifest["target_delta_ts"],
        color="black",
        linestyle="--",
        label=f"Target = {manifest['target_delta_ts']:g}",
    )
    ax.axvline(
        manifest["median_delta_ts"],
        color="#D55E00",
        label=f"Median = {manifest['median_delta_ts']:.2f}",
    )
    ax.axvline(
        representative["delta_ts"],
        color="#0072B2",
        linestyle=":",
        label=(
            f"Saved seed {int(representative['seed'])}: "
            f"{representative['delta_ts']:.2f}"
        ),
    )
    ax.set_xlabel(manifest["statistic_label"])
    ax.set_ylabel("Number of realizations")
    ax.set_title(manifest["case_description"])
    ax.legend()
    return fig, ax


def load_sed_handoff(manifest_file):
    manifest_file = Path(manifest_file)
    manifest = json.loads(manifest_file.read_text())
    source_expectation = Histogram.open(manifest["source_expectation_file"])
    data = Histogram.open(manifest["median_data_file"])
    background = (
        Histogram.open(manifest["background_file_3m"])
        .project("Em", "Phi", "PsiChi")
        * manifest["background_exposure_multiplier"]
    )
    return manifest, source_expectation, data, background
