"""Joint Poisson fits for response-folded, fixed-shape BinnedSED templates.

The spectral shape is supplied by a BinnedSED model in the notebook. Once its
shape parameters and background are fixed, expected detector counts are linear
in the BinnedSED normalizations. This module profiles those normalizations in
one likelihood over the complete measured-energy data cube.
"""

from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import pandas as pd
import astropy.units as u
from scipy.optimize import brentq, minimize


@dataclass
class JointSEDFitResult:
    k: np.ndarray
    scaled_k: np.ndarray
    relative_nll: float
    ts: np.ndarray | None = None
    lower68: np.ndarray | None = None
    upper68: np.ndarray | None = None
    upper95: np.ndarray | None = None


class JointSEDTemplates:
    def __init__(self, background, probe_templates, probe_k, max_k):
        self.background = np.asarray(background, dtype=float).ravel()
        templates = np.asarray(probe_templates, dtype=float)
        self.probe_k = np.asarray(probe_k, dtype=float)
        self.max_k = np.asarray(max_k, dtype=float)
        if templates.ndim < 2 or templates.shape[0] != len(self.probe_k):
            raise ValueError("One detector template is required for each SED bin.")
        self.templates = templates.reshape(len(self.probe_k), -1)
        if self.templates.shape[1] != self.background.size:
            raise ValueError("Templates and background have different detector cells.")
        if np.any(self.background < 0) or np.any(self.templates < 0):
            raise ValueError("Poisson expectations must be non-negative.")
        if np.any(self.probe_k <= 0) or np.any(self.max_k <= 0):
            raise ValueError("Probe and maximum normalizations must be positive.")
        self.probe_totals = self.templates.sum(axis=1)
        self.active_bins = np.flatnonzero(self.probe_totals > 0)
        if self.active_bins.size == 0:
            raise ValueError("No SED bin has a nonzero detector response.")
        self.k_scale = np.zeros(len(self.probe_k))
        self.k_scale[self.active_bins] = np.minimum(
            self.max_k[self.active_bins],
            self.probe_k[self.active_bins] * 10.0 / self.probe_totals[self.active_bins],
        )
        self.count_templates = self.templates[self.active_bins] * (
            self.k_scale[self.active_bins] / self.probe_k[self.active_bins]
        )[:, None]
        self.count_totals = self.count_templates.sum(axis=1)
        self.upper_scaled = self.max_k[self.active_bins] / self.k_scale[self.active_bins]

    def fit(self, data, initial_k=None, profile_ts=False, intervals=False):
        """Fit all K_i at once; optionally profile each bin's TS and limits."""
        data = np.asarray(data, dtype=float).ravel()
        if data.shape != self.background.shape or np.any(data < 0):
            raise ValueError("Data and background must match and be non-negative.")
        occupied = data > 0
        background = self.background[occupied]
        if np.any(background <= 0):
            raise ValueError("Background must be positive in occupied detector cells.")
        observed = data[occupied]
        templates = self.count_templates[:, occupied]
        totals = self.count_totals
        n_free = len(self.active_bins)
        bounds = list(zip(np.zeros(n_free), self.upper_scaled))
        if initial_k is None:
            x0 = np.ones(n_free)
        else:
            initial_k = np.asarray(initial_k, dtype=float)
            x0 = initial_k[self.active_bins] / self.k_scale[self.active_bins]
            x0 = np.clip(x0, 0, self.upper_scaled)

        def objective(x):
            source = x @ templates
            expected = background + source
            value = float(
                totals @ x - observed @ np.log1p(source / background)
            )
            gradient = totals - templates @ (observed / expected)
            return value, gradient

        optimum = minimize(
            objective, x0, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6},
        )
        if not optimum.success:
            optimum = minimize(
                objective, optimum.x, method="SLSQP", jac=True, bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-9},
            )
        if not optimum.success:
            raise RuntimeError(f"Joint BinnedSED fit failed: {optimum.message}")
        best = np.asarray(optimum.x, dtype=float)
        best_nll = float(objective(best)[0])
        k = np.zeros(len(self.probe_k))
        k[self.active_bins] = best * self.k_scale[self.active_bins]
        result = JointSEDFitResult(k=k, scaled_k=best, relative_nll=best_nll)
        if not profile_ts and not intervals:
            return result

        def profile_nll(local_index, fixed_value):
            if np.isclose(fixed_value, best[local_index], rtol=0, atol=1e-12):
                return best_nll
            other = np.arange(n_free) != local_index
            trial = best.copy()
            trial[local_index] = fixed_value
            if not np.any(other):
                return float(objective(trial)[0])

            def other_objective(values):
                trial[other] = values
                value, gradient = objective(trial)
                return value, gradient[other]

            profiled = minimize(
                other_objective, best[other], method="L-BFGS-B", jac=True,
                bounds=[bounds[i] for i in np.flatnonzero(other)],
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-6},
            )
            if not profiled.success:
                profiled = minimize(
                    other_objective, profiled.x, method="SLSQP", jac=True,
                    bounds=[bounds[i] for i in np.flatnonzero(other)],
                    options={"maxiter": 500, "ftol": 1e-9},
                )
            if not profiled.success or not np.isfinite(profiled.fun):
                raise RuntimeError(f"SED bin {local_index} profile failed: {profiled.message}")
            return float(profiled.fun)

        ts = np.zeros(len(self.probe_k))
        lower68 = np.zeros(len(self.probe_k))
        upper68 = np.full(len(self.probe_k), np.nan)
        upper95 = np.full(len(self.probe_k), np.nan)
        for j, bin_index in enumerate(self.active_bins):
            null_nll = profile_nll(j, 0.0)
            ts[bin_index] = max(0.0, 2.0 * (null_nll - best_nll))
            if not intervals:
                continue

            def delta(value):
                return max(0.0, 2.0 * (profile_nll(j, value) - best_nll))

            if best[j] > 0 and ts[bin_index] > 1.0:
                lower_scaled = brentq(
                    lambda value: delta(value) - 1.0, 0.0, best[j],
                    xtol=1e-8, rtol=1e-7,
                )
                lower68[bin_index] = lower_scaled * self.k_scale[bin_index]

            def upper_root(target):
                lo = best[j]
                hi = min(self.upper_scaled[j], max(lo + 1.0, lo * 2.0))
                while delta(hi) < target and hi < self.upper_scaled[j]:
                    hi = min(self.upper_scaled[j], hi * 2.0)
                if delta(hi) < target:
                    return float("nan")
                return brentq(
                    lambda value: delta(value) - target, lo, hi,
                    xtol=1e-8, rtol=1e-7,
                ) * self.k_scale[bin_index]

            upper68[bin_index] = upper_root(1.0)
            upper95[bin_index] = upper_root(2.705543)
        result.ts = ts
        if intervals:
            result.lower68 = lower68
            result.upper68 = upper68
            result.upper95 = upper95
        return result


def _histogram_values(histogram):
    """Return dense numerical counts from a histpy histogram."""
    contents = histogram.contents
    if hasattr(contents, "compute"):
        contents = contents.compute()
    if hasattr(contents, "todense"):
        contents = contents.todense()
    if hasattr(contents, "value"):
        contents = contents.value
    return np.asarray(contents, dtype=float)


def fit_global_binned_sed(
    *,
    global_results,
    global_plugin,
    source_name,
    dr_path,
    sc_orientation,
    longitude_deg,
    latitude_deg,
    background_pseudocount=1e-12,
    detection_ts=4.0,
    label="representative",
):
    """Fit all response Ei-bin normalizations to one saved realization.

    Every bin uses the complete fitted global spectral shape, including all
    components of a composite model. The global COSI background rate is held
    fixed. One Poisson likelihood covers the full detector data cube, and each
    bin's TS and intervals profile the other bin normalizations.

    ``global_plugin`` supplies the exact data and raw background histograms
    used in the global fit, so no new seed or ensemble is generated here.
    """
    from astromodels import Model, PointSource
    from cosipy.threeml import BinnedSED
    from agn_cosi_fit_utils import (
        COSIPlugin,
        _open_response,
        make_cosi_background_parameter,
    )
    from agn_sed_ensemble import classify_representative_sed

    response = _open_response(dr_path)
    ei_axis = response.axes["Ei"]
    ei_indices = np.arange(ei_axis.nbins, dtype=int)
    edges = ei_axis.edges
    edges_kev = (
        edges.to_value(u.keV)
        if isinstance(edges, u.Quantity)
        else np.asarray(edges, dtype=float)
    )
    pivots_kev = np.sqrt(edges_kev[:-1] * edges_kev[1:])

    fitted_shape = deepcopy(
        global_results.optimized_model[source_name].spectrum.main.shape
    )
    initial_k = np.asarray(
        [float(fitted_shape.evaluate_at(energy)) for energy in pivots_kev],
        dtype=float,
    )
    if np.any(~np.isfinite(initial_k)) or np.any(initial_k <= 0):
        raise ValueError("Global spectral shape must be positive at every SED pivot.")
    sed_shape = BinnedSED.from_response(
        response,
        spectral_shape=fitted_shape,
        ei_bin_indices=ei_indices,
        initial_fluxes=initial_k,
    )
    sed_model = Model(
        PointSource(
            source_name,
            l=float(longitude_deg),
            b=float(latitude_deg),
            spectral_shape=sed_shape,
        )
    )
    if len(sed_model.free_parameters) != sed_shape.n_bins:
        raise RuntimeError("Only the BinnedSED bin normalizations should be free.")

    plugin_name = f"cosi_binned_sed_{label}"
    sed_plugin = COSIPlugin(
        plugin_name,
        dr=dr_path,
        data=global_plugin._data,
        bkg=global_plugin._raw_bkg_hist,
        sc_orientation=sc_orientation,
        nuisance_param=make_cosi_background_parameter(plugin_name),
        background_pseudocount=background_pseudocount,
        earth_occ=True,
    )
    sed_plugin.set_model(sed_model)
    global_background = list(global_plugin.nuisance_parameters.values())
    sed_background = list(sed_plugin.nuisance_parameters.values())
    if len(global_background) != 1 or len(sed_background) != 1:
        raise ValueError("The simultaneous SED requires one COSI background rate.")
    sed_background[0].value = float(global_background[0].value)
    sed_background[0].fix = True

    k_parameters = list(sed_shape.normalizations)
    for parameter in k_parameters:
        parameter.value = 0.0
    null_log_like = float(sed_plugin.get_log_like())
    background_counts = _histogram_values(sed_plugin._bkg.expectation(copy=True))
    source_response = sed_plugin._response._source_responses[source_name]
    probe_k = np.maximum(initial_k, 1e-8)
    probe_templates = []
    for parameter, probe in zip(k_parameters, probe_k):
        parameter.value = float(probe)
        probe_templates.append(
            _histogram_values(source_response.expectation(copy=True))
        )
        parameter.value = 0.0
    templates = JointSEDTemplates(
        background_counts,
        np.asarray(probe_templates),
        probe_k,
        np.asarray([float(parameter.max_value) for parameter in k_parameters]),
    )
    joint_fit = templates.fit(
        _histogram_values(sed_plugin._data),
        initial_k=initial_k,
        profile_ts=True,
        intervals=True,
    )
    for parameter, value in zip(k_parameters, joint_fit.k):
        parameter.value = float(value)
    fitted_log_like = float(sed_plugin.get_log_like())
    if not np.isclose(
        -(fitted_log_like - null_log_like),
        joint_fit.relative_nll,
        rtol=1e-6,
        atol=1e-3,
    ):
        raise RuntimeError("Template likelihood disagrees with direct COSI likelihood.")

    conversion = u.keV.to(u.erg) * pivots_kev**2
    is_upper_limit = (joint_fit.ts < detection_ts) | (joint_fit.lower68 == 0)
    hi_k = np.where(is_upper_limit, joint_fit.upper95, joint_fit.upper68)
    result = pd.DataFrame(
        {
            "bin_index": ei_indices + 1,
            "ei_index": ei_indices,
            "e_min_keV": edges_kev[:-1],
            "e_max_keV": edges_kev[1:],
            "e_ref_keV": pivots_kev,
            "source_K": joint_fit.k,
            "source_K_lo": joint_fit.lower68,
            "source_K_hi": joint_fit.upper68,
            "source_K_ul95": joint_fit.upper95,
            "sed_erg_cm2_s": joint_fit.k * conversion,
            "sed_lo_erg_cm2_s": joint_fit.lower68 * conversion,
            "sed_hi_erg_cm2_s": hi_k * conversion,
            "sed_ul95_erg_cm2_s": joint_fit.upper95 * conversion,
            "ts_value": joint_fit.ts,
            "sigma": np.sqrt(joint_fit.ts),
            "is_upper_limit": is_upper_limit,
            "background_rate_hz": float(sed_background[0].value),
            "background_pseudocount": background_pseudocount,
        }
    )
    result = classify_representative_sed(result, detection_ts=detection_ts)
    result.attrs["spectral_shape"] = repr(fitted_shape)
    result.attrs["source_name"] = source_name
    result.attrs["direct_cosi_nll"] = -fitted_log_like
    return result
