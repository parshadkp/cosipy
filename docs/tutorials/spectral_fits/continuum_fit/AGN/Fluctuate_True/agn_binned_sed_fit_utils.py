"""Joint Poisson fits for response-folded, fixed-shape BinnedSED templates.

The spectral shape is supplied by a BinnedSED model in the notebook. Expected
detector counts are linear in the BinnedSED normalizations and optional
background scale. This module profiles those parameters in one likelihood over
the measured-energy data cube, excluding zero-raw-background detector cells
by default.
"""

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

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
    upper3sigma: np.ndarray | None = None
    bayes95: np.ndarray | None = None
    background_scale: float = 1.0


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

    def asimov_detection_sensitivity(
        self, *, target_ts=9.0, support_mask=None, background_free=False,
        background_scale_bounds=(1e-12, 5.0),
    ):
        """Per-bin source K giving the target background-only Asimov TS.

        Each Ei bin is tested as one source component against background only.
        If background_free, the null hypothesis profiles its one background
        scale while the signal hypothesis has its generating value of one.
        Cells outside ``support_mask`` make no contribution; this avoids
        treating a background pseudocount as measured background support.
        """
        if target_ts <= 0:
            raise ValueError("The target Asimov TS must be positive.")
        if support_mask is None:
            support = self.background > 0
        else:
            support = np.asarray(support_mask, dtype=bool).ravel().copy()
            if support.shape != self.background.shape:
                raise ValueError("The support mask must match the background shape.")
            support &= self.background > 0
        if not np.any(support):
            raise ValueError("No positive-background detector cells are supported.")
        background = self.background[support]
        background_total = float(background.sum())
        lower_scale, upper_scale = map(float, background_scale_bounds)
        if background_free and not (0 < lower_scale <= 1 <= upper_scale):
            raise ValueError("Background scale bounds must contain one.")

        sensitivity_k = np.full(len(self.probe_k), np.nan)
        for j, bin_index in enumerate(self.active_bins):
            template = self.count_templates[j, support]
            if not np.any(template > 0):
                continue

            def asimov_ts(scaled_k):
                signal = scaled_k * template
                fixed_ts = 2.0 * np.sum(
                    (background + signal) * np.log1p(signal / background) - signal
                )
                if not background_free:
                    return max(0.0, float(fixed_ts))
                data_total = background_total + float(signal.sum())
                null_scale = np.clip(
                    data_total / background_total, lower_scale, upper_scale
                )
                null_gain = 2.0 * (
                    data_total * np.log(null_scale)
                    - (null_scale - 1.0) * background_total
                )
                return max(0.0, float(fixed_ts - null_gain))

            cap = float(self.upper_scaled[j])
            hi = min(cap, 1.0)
            while hi < cap and asimov_ts(hi) < target_ts:
                hi = min(cap, 2.0 * hi)
            if asimov_ts(hi) >= target_ts:
                sensitivity_k[bin_index] = brentq(
                    lambda value: asimov_ts(value) - target_ts,
                    0.0, hi, xtol=1e-8, rtol=1e-7,
                ) * self.k_scale[bin_index]
        return sensitivity_k

    def fit(
        self, data, initial_k=None, profile_ts=False, intervals=False,
        background_free=False, background_scale_bounds=(1e-12, 5.0),
        calculate_upper3sigma=False,
        calculate_upper95=False,
    ):
        """Fit all K_i and optionally the background scale in one likelihood."""
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
        n_source = len(self.active_bins)
        bounds = list(zip(np.zeros(n_source), self.upper_scaled))
        if initial_k is None:
            x0 = np.ones(n_source)
        else:
            initial_k = np.asarray(initial_k, dtype=float)
            x0 = initial_k[self.active_bins] / self.k_scale[self.active_bins]
            x0 = np.clip(x0, 0, self.upper_scaled)
        if background_free:
            lower, upper = map(float, background_scale_bounds)
            if not (0 < lower <= 1 <= upper and np.isfinite(upper)):
                raise ValueError("Background scale bounds must contain 1 and be positive.")
            bounds.append((lower, upper))
            x0 = np.append(x0, 1.0)
        n_free = len(bounds)
        background_total = self.background.sum()

        def objective(x):
            source_k = x[:n_source]
            source = source_k @ templates
            if background_free:
                scale = x[-1]
                expected = scale * background + source
                value = float(
                    (scale - 1.0) * background_total
                    + totals @ source_k
                    - observed @ np.log(expected / background)
                )
                gradient_source = totals - templates @ (observed / expected)
                gradient_background = background_total - background @ (observed / expected)
                return value, np.append(gradient_source, gradient_background)
            expected = background + source
            value = float(totals @ source_k - observed @ np.log1p(source / background))
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
        k[self.active_bins] = best[:n_source] * self.k_scale[self.active_bins]
        result = JointSEDFitResult(
            k=k, scaled_k=best[:n_source], relative_nll=best_nll,
            background_scale=float(best[-1]) if background_free else 1.0,
        )
        if not profile_ts and not intervals and not calculate_upper95:
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
        upper3sigma = np.full(len(self.probe_k), np.nan) if calculate_upper3sigma else None
        for j, bin_index in enumerate(self.active_bins):
            null_nll = profile_nll(j, 0.0)
            ts[bin_index] = max(0.0, 2.0 * (null_nll - best_nll))
            if not intervals and not calculate_upper95:
                continue

            def delta(value):
                return max(0.0, 2.0 * (profile_nll(j, value) - best_nll))

            if intervals and best[j] > 0 and ts[bin_index] > 1.0:
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

            if intervals:
                upper68[bin_index] = upper_root(1.0)
            upper95[bin_index] = upper_root(2.705543)
            if calculate_upper3sigma:
                upper3sigma[bin_index] = upper_root(9.0)
        result.ts = ts
        if intervals:
            result.lower68 = lower68
            result.upper68 = upper68
            result.upper3sigma = upper3sigma
        if intervals or calculate_upper95:
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


def _threeml_bayesian_bin_limits(
    sed_model, k_parameters, templates, joint_fit, data, ts_threshold,
):
    """Conditional 95th posterior percentiles using threeML's BayesianAnalysis.

    The likelihood is the same response-folded Poisson template likelihood as
    the simultaneous fit. Other bin amplitudes and the fitted background are
    held at their joint maximum-likelihood values for each one-bin posterior.
    """
    limits = np.full(len(k_parameters), np.nan)
    if not np.any(joint_fit.ts < ts_threshold):
        return limits

    from astromodels import Uniform_prior
    from threeML import BayesianAnalysis, DataList
    from threeML.config import threeML_config
    from threeML.plugin_prototype import PluginPrototype

    class ConditionalBinPlugin(PluginPrototype):
        def __init__(self, parameter, base, template, observed, total):
            super().__init__("cosi_sed_conditional", {})
            self.parameter = parameter
            self.base = base
            self.template = template
            self.observed = observed
            self.total = total

        def set_model(self, likelihood_model_instance):
            self.model = likelihood_model_instance

        def get_log_like(self):
            amplitude = float(self.parameter.value)
            signal = amplitude * self.template
            return float(
                self.observed @ np.log1p(signal / self.base)
                - amplitude * self.total
            )

        def inner_fit(self):
            return self.get_log_like()

        def get_number_of_data_points(self):
            return len(self.observed)

    counts = np.asarray(data, dtype=float).ravel()
    occupied = counts > 0
    observed = counts[occupied]
    fitted_background = templates.background * joint_fit.background_scale
    fitted_source = np.sum(
        templates.templates * (joint_fit.k / templates.probe_k)[:, None], axis=0
    )
    saved = [(parameter.value, parameter.fix, parameter.prior)
             for parameter in k_parameters]
    try:
        for parameter in k_parameters:
            parameter.fix = True
        for bin_index in np.flatnonzero(joint_fit.ts < ts_threshold):
            parameter = k_parameters[bin_index]
            max_k = float(parameter.max_value)
            if not np.isfinite(max_k) or max_k <= 0:
                raise ValueError(f"SED bin {bin_index + 1} needs a finite positive prior bound.")
            template = templates.templates[bin_index] / templates.probe_k[bin_index]
            base = fitted_background + fitted_source - joint_fit.k[bin_index] * template
            if np.any(base[occupied] <= 0):
                raise ValueError(f"SED bin {bin_index + 1} has no positive background support.")
            plugin = ConditionalBinPlugin(
                parameter, base[occupied], template[occupied], observed,
                float(template.sum()),
            )
            parameter.prior = Uniform_prior(lower_bound=0.0, upper_bound=max_k)
            profile_start = float(joint_fit.upper95[bin_index])
            if not np.isfinite(profile_start) or profile_start <= 0:
                profile_start = max_k * 1e-3
            parameter.value = min(
                max(profile_start / 2.0, max_k * 1e-8), max_k * 0.9
            )
            parameter.fix = False
            analysis = BayesianAnalysis(sed_model, DataList(plugin))
            analysis.set_sampler("emcee")
            analysis.sampler.setup(
                n_iterations=2000, n_burn_in=600, n_walkers=24,
                seed=104729 + int(bin_index),
            )
            previous_progress = threeML_config.interface.progress_bars
            try:
                threeML_config.interface.progress_bars = False
                analysis.sample(quiet=True)
            finally:
                threeML_config.interface.progress_bars = previous_progress
            samples = np.asarray(analysis.raw_samples, dtype=float)[:, 0]
            limit = float(np.quantile(samples, 0.95))
            if not np.isfinite(limit) or limit >= 0.9 * max_k:
                raise RuntimeError(
                    f"SED bin {bin_index + 1} Bayesian limit reaches its prior bound."
                )
            limits[bin_index] = limit
            parameter.fix = True
    finally:
        for parameter, (value, fixed, prior) in zip(k_parameters, saved):
            parameter.fix = fixed
            parameter.prior = prior
            parameter.value = value
    return limits


def classify_sed_ts_only(sed_dataframe, detection_ts=4.0):
    """Plot TS > threshold as detections and every other bin as a 95% limit.

    In particular, TS = 0 is retained. An unconstrained bin without a finite
    profile limit raises an error instead of disappearing from the figure.
    """
    result = sed_dataframe.copy()
    ts = result["ts_value"].to_numpy(dtype=float)
    upper95 = result["sed_ul95_erg_cm2_s"].to_numpy(dtype=float)
    if np.any(~np.isfinite(ts)):
        raise ValueError("Every SED bin needs a finite TS for classification.")
    is_upper_limit = ts <= float(detection_ts)
    missing_limit = is_upper_limit & (~np.isfinite(upper95) | (upper95 <= 0))
    if np.any(missing_limit):
        bins = result.loc[missing_limit, "bin_index"].tolist()
        raise ValueError(f"No finite 95% upper limit for SED bins {bins}.")
    result["plot_role"] = np.where(is_upper_limit, "upper_limit", "detection")
    result["is_upper_limit"] = is_upper_limit
    result.loc[is_upper_limit, "sed_hi_erg_cm2_s"] = upper95[is_upper_limit]
    if "source_K_hi" in result.columns:
        conversion = u.keV.to(u.erg) * result["e_ref_keV"].to_numpy(dtype=float) ** 2
        result.loc[~is_upper_limit, "sed_hi_erg_cm2_s"] = (
            result.loc[~is_upper_limit, "source_K_hi"].to_numpy(dtype=float)
            * conversion[~is_upper_limit]
        )
    result["classification_rule"] = (
        f"TS>{float(detection_ts):g}: detection; "
        f"TS<={float(detection_ts):g}: finite 95% upper limit"
    )
    return result


def expand_sed_y_limits(axes, *sed_dataframes, margin=1.3):
    """Keep all classified SED detections and limits visible on log axes."""
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    plotted = []
    for frame in sed_dataframes:
        roles = frame["plot_role"].astype(str)
        plotted.extend(frame.loc[roles == "detection", "sed_erg_cm2_s"].tolist())
        plotted.extend(frame.loc[roles == "upper_limit", "sed_ul95_erg_cm2_s"].tolist())
    values = np.asarray(plotted, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return
    for axis in np.asarray(axes, dtype=object).ravel():
        lower, upper = axis.get_ylim()
        axis.set_ylim(min(lower, values.min() / margin), max(upper, values.max() * margin))


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
    use_default_powerlaw_shape=False,
    free_background=False,
    calculate_upper3sigma=False,
    bayesian_ul_ts_threshold=1.0,
    calculate_asimov_sensitivity=False,
    exclude_zero_raw_background_cells=True,
    return_ensemble_context=False,
):
    """Fit all response Ei-bin normalizations to one saved realization.

    By default, every bin uses the complete fitted global spectral shape,
    including all components of a composite model. Set
    ``use_default_powerlaw_shape=True`` to use BinnedSED's default fixed
    index -2 power law in each bin instead. The global COSI background rate is
    held fixed unless ``free_background=True``; in that case its overall rate
    is profiled together with all bin normalizations. One Poisson likelihood
    covers the full detector data cube, and each bin's TS and intervals
    profile the other free parameters. By default, detector cells with zero
    raw background are excluded from the SED likelihood.

    ``global_plugin`` supplies the exact data and raw background histograms
    used in the global fit, so no new seed or ensemble is generated here.
    Set ``calculate_upper3sigma=True`` for one-sided profile-likelihood upper
    bounds at 2 Delta log L = 9; this adds profiling work only when requested.
    For bins with TS < 1 by default, threeML samples a conditional posterior
    with a flat prior on nonnegative source normalization and all other fitted
    amplitudes (including background) held fixed. Set
    ``bayesian_ul_ts_threshold=None`` to retain profile limits throughout.
    Set ``calculate_asimov_sensitivity=True`` for the per-bin background-only
    expected detection threshold at TS = 9 on the same positive-raw-background
    detector cells used by the default SED fit.
    Set ``exclude_zero_raw_background_cells=False`` to include detector cells
    with zero raw background for a comparison fit.
    Set ``return_ensemble_context=True`` to reuse the folded templates for
    multiple Poisson realizations with the same fixed background and shape.
    """
    from astromodels import Model, PointSource
    from cosipy.threeml import BinnedSED
    from agn_cosi_fit_utils import (
        COSIPlugin,
        _open_response,
        make_cosi_background_parameter,
    )

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
    sed_kwargs = {
        "ei_bin_indices": ei_indices,
        "initial_fluxes": initial_k,
    }
    if not use_default_powerlaw_shape:
        sed_kwargs["spectral_shape"] = fitted_shape
    sed_shape = BinnedSED.from_response(response, **sed_kwargs)
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
    global_background_rate = float(global_background[0].value)
    sed_background[0].value = global_background_rate
    sed_background[0].fix = not free_background

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
    data_counts = _histogram_values(sed_plugin._data)
    if data_counts.shape != background_counts.shape:
        raise ValueError("Fitted background and data shapes differ.")
    if exclude_zero_raw_background_cells or calculate_asimov_sensitivity:
        raw_background = _histogram_values(sed_plugin._raw_bkg_hist)
        if raw_background.shape != background_counts.shape:
            raise ValueError("Raw and fitted background shapes differ.")
        raw_support = raw_background.ravel() > 0
        if not np.any(raw_support):
            raise ValueError("The raw background has no supported detector cells.")
    if exclude_zero_raw_background_cells:
        full_probes = np.asarray(probe_templates).reshape(len(probe_k), -1)
        probe_totals = full_probes.sum(axis=1)
        excluded_source_fraction = np.divide(
            full_probes[:, ~raw_support].sum(axis=1),
            probe_totals,
            out=np.full(len(probe_k), np.nan),
            where=probe_totals > 0,
        )
        fit_background = background_counts.ravel()[raw_support]
        fit_probes = full_probes[:, raw_support]
        fit_data = data_counts.ravel()[raw_support]
    else:
        fit_background = background_counts
        fit_probes = np.asarray(probe_templates)
        fit_data = data_counts
    templates = JointSEDTemplates(
        fit_background,
        fit_probes,
        probe_k,
        np.asarray([float(parameter.max_value) for parameter in k_parameters]),
    )
    background_scale_bounds = (
        max(1e-12, float(sed_background[0].min_value or 0.0) / global_background_rate),
        float(sed_background[0].max_value) / global_background_rate,
    ) if free_background else (1e-12, 5.0)
    if calculate_asimov_sensitivity:
        support = (
            np.ones(templates.background.size, dtype=bool)
            if exclude_zero_raw_background_cells else raw_support
        )
        asimov_sensitivity_k = templates.asimov_detection_sensitivity(
            target_ts=9.0,
            support_mask=support,
            background_free=free_background,
            background_scale_bounds=background_scale_bounds,
        )
        if exclude_zero_raw_background_cells:
            supported_source_fraction = 1.0 - excluded_source_fraction
        else:
            supported_source_fraction = np.full(len(probe_k), np.nan)
            supported_source_fraction[templates.active_bins] = (
                templates.templates[templates.active_bins][:, support].sum(axis=1)
                / templates.probe_totals[templates.active_bins]
            )
    joint_fit = templates.fit(
        fit_data,
        initial_k=initial_k,
        profile_ts=True,
        intervals=True,
        background_free=free_background,
        calculate_upper3sigma=calculate_upper3sigma,
        background_scale_bounds=background_scale_bounds,
    )
    for parameter, value in zip(k_parameters, joint_fit.k):
        parameter.value = float(value)
    if free_background:
        sed_background[0].value = global_background_rate * joint_fit.background_scale
    fitted_log_like = (
        np.nan if exclude_zero_raw_background_cells else float(sed_plugin.get_log_like())
    )
    if not exclude_zero_raw_background_cells:
        if not np.isclose(
            -(fitted_log_like - null_log_like),
            joint_fit.relative_nll,
            rtol=1e-6,
            atol=1e-3,
        ):
            raise RuntimeError("Template likelihood disagrees with direct COSI likelihood.")

    profile_upper95 = joint_fit.upper95.copy()
    if bayesian_ul_ts_threshold is not None:
        joint_fit.bayes95 = _threeml_bayesian_bin_limits(
            sed_model, k_parameters, templates, joint_fit,
            fit_data,
            float(bayesian_ul_ts_threshold),
        )
        use_bayes = joint_fit.ts < float(bayesian_ul_ts_threshold)
        joint_fit.upper95[use_bayes] = joint_fit.bayes95[use_bayes]

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
    if calculate_upper3sigma:
        result["source_K_ul3sigma"] = joint_fit.upper3sigma
        result["sed_ul3sigma_erg_cm2_s"] = joint_fit.upper3sigma * conversion
    if exclude_zero_raw_background_cells:
        result["excluded_source_template_fraction"] = excluded_source_fraction
    if bayesian_ul_ts_threshold is not None:
        result["source_K_ul95_profile"] = profile_upper95
        result["sed_ul95_profile_erg_cm2_s"] = profile_upper95 * conversion
        result["source_K_ul95_bayes"] = joint_fit.bayes95
        result["sed_ul95_bayes_erg_cm2_s"] = joint_fit.bayes95 * conversion
    if calculate_asimov_sensitivity:
        result["source_K_asimov_sensitivity3sigma"] = asimov_sensitivity_k
        result["sed_asimov_sensitivity3sigma_erg_cm2_s"] = (
            asimov_sensitivity_k * conversion
        )
        result["asimov_supported_source_fraction"] = supported_source_fraction
    result = classify_sed_ts_only(result, detection_ts=detection_ts)
    result["upper_limit_method"] = np.select(
        [
            result["plot_role"].eq("detection"),
            (result["ts_value"] < float(bayesian_ul_ts_threshold))
            if bayesian_ul_ts_threshold is not None
            else np.zeros(len(result), dtype=bool),
        ],
        ["detection", "threeML Bayes 95%"],
        default="profile 95%",
    )
    result.attrs["spectral_shape"] = repr(sed_shape.spectral_shape)
    result.attrs["sed_shape_mode"] = (
        "default_powerlaw_index_minus_2"
        if use_default_powerlaw_shape else "global_fit"
    )
    result.attrs["background_mode"] = "free" if free_background else "fixed"
    result.attrs["global_background_rate_hz"] = global_background_rate
    result.attrs["source_name"] = source_name
    result.attrs["direct_cosi_nll"] = (
        np.nan if exclude_zero_raw_background_cells else -fitted_log_like
    )
    if exclude_zero_raw_background_cells:
        result.attrs["data_support_mode"] = "positive_raw_background_only"
        result.attrs["excluded_detector_cells"] = int((~raw_support).sum())
        result.attrs["excluded_observed_counts"] = float(data_counts.ravel()[~raw_support].sum())
    if calculate_asimov_sensitivity:
        result.attrs["sensitivity_method"] = (
            "per_Ei_bin_Asimov_TS9_raw_background_support"
        )
        result.attrs["sensitivity_background_mode"] = (
            "profile_null_rate" if free_background else "fixed_global_rate"
        )
    if return_ensemble_context:
        return result, {
            "templates": templates,
            "sed_model": sed_model,
            "k_parameters": k_parameters,
            "initial_k": initial_k,
            "e_min_keV": edges_kev[:-1],
            "e_max_keV": edges_kev[1:],
            "e_ref_keV": pivots_kev,
            "conversion": conversion,
            "global_background_rate_hz": global_background_rate,
        }
    return result


def run_or_load_default_binned_sed_ensemble(
    *, manifest, source_expectation, background_expectation,
    global_results, global_plugin, source_name, dr_path, sc_orientation,
    longitude_deg, latitude_deg, output_stem,
    background_pseudocount=1e-12, expected_seed_count=300,
    detection_ts=4.0, recompute=False,
):
    """Summarize repeated joint BinnedSED fits with a fixed index -2 bin shape.

    The injected source and raw background generate each Poisson data cube.
    The response-folded default BinnedSED templates and fixed global-fit
    background are built once and reused for every seed.  Each seed gets a
    simultaneous fit of all true-energy bin normalizations, a profile TS in
    every bin, and an individual 95% upper limit. Limits with TS < 1 use a
    conditional threeML posterior; the other limits use the profile likelihood.
    The reported flux, TS, and upper limit are separate across-seed medians.
    """
    from agn_sed_ensemble import load_manifest_seeds

    seeds = load_manifest_seeds(manifest, expected_count=expected_seed_count)
    output_dir = Path(manifest["trials_csv"]).parent
    trials_file = output_dir / f"{output_stem}_trials.csv"
    summary_file = output_dir / f"{output_stem}_summary.csv"
    background_parameters = list(global_plugin.nuisance_parameters.values())
    if len(background_parameters) != 1:
        raise ValueError("The SED ensemble requires one global COSI background rate.")
    background_rate_hz = float(background_parameters[0].value)

    cached_with_bayesian_limits = (
        trials_file.exists() and summary_file.exists() and not recompute
        and "upper_limit_method" in pd.read_csv(trials_file, nrows=0).columns
        and "upper_limit_method" in pd.read_csv(summary_file, nrows=0).columns
    )
    if cached_with_bayesian_limits:
        trials = pd.read_csv(trials_file)
        summary = pd.read_csv(summary_file)
    else:
        representative, context = fit_global_binned_sed(
            global_results=global_results, global_plugin=global_plugin,
            source_name=source_name, dr_path=dr_path,
            sc_orientation=sc_orientation, longitude_deg=longitude_deg,
            latitude_deg=latitude_deg,
            background_pseudocount=background_pseudocount,
            label=f"{output_stem}_template",
            use_default_powerlaw_shape=True, free_background=False,
            bayesian_ul_ts_threshold=None,
            return_ensemble_context=True,
        )
        templates = context["templates"]
        source = _histogram_values(source_expectation)
        raw_background = _histogram_values(background_expectation)
        if source.shape != raw_background.shape or source.size != templates.background.size:
            raise ValueError("Injected source, background, and BinnedSED template shapes differ.")
        if np.any(source < 0) or np.any(raw_background < 0):
            raise ValueError("Poisson source and background expectations must be nonnegative.")
        expectation = source + raw_background
        conversion = context["conversion"]
        initial_k = representative["source_K"].to_numpy(dtype=float)
        rows = []
        for seed in seeds:
            data = np.random.default_rng(int(seed)).poisson(expectation)
            try:
                fit = templates.fit(
                    data, initial_k=initial_k, profile_ts=True,
                    intervals=False, calculate_upper95=True,
                    background_free=False,
                )
            except Exception as error:
                raise RuntimeError(f"Default BinnedSED fit failed for seed {int(seed)}") from error
            bayes95 = _threeml_bayesian_bin_limits(
                context["sed_model"], context["k_parameters"], templates,
                fit, data, 1.0,
            )
            use_bayes = fit.ts < 1.0
            combined_upper95 = fit.upper95.copy()
            combined_upper95[use_bayes] = bayes95[use_bayes]
            for i in range(len(conversion)):
                rows.append({
                    "seed": int(seed), "bin_index": i + 1,
                    "e_min_keV": float(context["e_min_keV"][i]),
                    "e_max_keV": float(context["e_max_keV"][i]),
                    "e_ref_keV": float(context["e_ref_keV"][i]),
                    "source_K": float(fit.k[i]),
                    "source_K_ul95": float(combined_upper95[i]),
                    "source_K_ul95_profile": float(fit.upper95[i]),
                    "sed_erg_cm2_s": float(fit.k[i] * conversion[i]),
                    "sed_ul95_erg_cm2_s": float(combined_upper95[i] * conversion[i]),
                    "sed_ul95_profile_erg_cm2_s": float(fit.upper95[i] * conversion[i]),
                    "ts_value": float(fit.ts[i]),
                    "upper_limit_method": (
                        "threeML Bayes 95%" if use_bayes[i] else "profile 95%"
                    ),
                })
        trials = pd.DataFrame(rows)
        summary_rows = []
        for bin_index, group in trials.groupby("bin_index", sort=True):
            summary_rows.append({
                "bin_index": int(bin_index),
                "e_min_keV": float(group["e_min_keV"].iloc[0]),
                "e_max_keV": float(group["e_max_keV"].iloc[0]),
                "e_ref_keV": float(group["e_ref_keV"].iloc[0]),
                "n_seeds": int(group["seed"].nunique()),
                "ts_value": float(group["ts_value"].median()),
                "sed_erg_cm2_s": float(group["sed_erg_cm2_s"].median()),
                "sed_lo_erg_cm2_s": float(group["sed_erg_cm2_s"].quantile(0.16)),
                "sed_hi_erg_cm2_s": float(group["sed_erg_cm2_s"].quantile(0.84)),
                "sed_ul95_erg_cm2_s": float(group["sed_ul95_erg_cm2_s"].median()),
                "sed_ul95_profile_erg_cm2_s": float(
                    group["sed_ul95_profile_erg_cm2_s"].median()
                ),
                "n_bayesian_ul_seeds": int(
                    (group["upper_limit_method"] == "threeML Bayes 95%").sum()
                ),
                "upper_limit_method": "median of per-seed threeML/profile 95% limits",
                "background_rate_hz": background_rate_hz,
                "background_pseudocount": float(background_pseudocount),
                "sed_shape_mode": "default_powerlaw_index_minus_2",
                "background_mode": "fixed",
            })
        summary = pd.DataFrame(summary_rows)
        trials.to_csv(trials_file, index=False)
        summary.to_csv(summary_file, index=False)

    if not np.array_equal(np.sort(trials["seed"].astype(int).unique()), seeds):
        raise ValueError("Cached BinnedSED trials do not match the manifest's seed set.")
    if (summary["n_seeds"].to_numpy(dtype=int) != len(seeds)).any():
        raise ValueError("Cached BinnedSED summary has an incomplete seed ensemble.")
    if not (summary["sed_shape_mode"] == "default_powerlaw_index_minus_2").all():
        raise ValueError("Cached BinnedSED summary does not use the default power law.")
    if not (summary["background_mode"] == "fixed").all():
        raise ValueError("Cached BinnedSED summary does not hold the background fixed.")
    if not np.allclose(summary["background_rate_hz"], background_rate_hz, rtol=1e-6):
        raise ValueError("Cached BinnedSED background rate differs from the global fit.")
    if not np.allclose(summary["background_pseudocount"], background_pseudocount, rtol=1e-6):
        raise ValueError("Cached BinnedSED pseudocount differs from the requested value.")
    summary = classify_sed_ts_only(summary, detection_ts=detection_ts)
    return summary, trials_file, summary_file
