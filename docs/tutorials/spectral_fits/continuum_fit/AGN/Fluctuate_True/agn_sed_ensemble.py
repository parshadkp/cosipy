"""Fast bin-by-bin SED ensembles for the AGN simulation notebooks.

The source spectral shape and the background normalization are fixed.  In each
measured-energy group, the source expectation is therefore a one-parameter
Poisson template::

    expectation = background + source_scale * source_template

This is equivalent to the notebook SED likelihood when only the linked source
normalization is free, but avoids rebuilding the detector response and a Minuit
fit for every seed and energy group.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _histogram_values(histogram):
    values = histogram.contents
    if hasattr(values, "compute"):
        values = values.compute()
    if hasattr(values, "todense"):
        values = values.todense()
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float)


def load_threshold_seeds(trials_csv, norm_nt, expected_count=None):
    """Load the unique seed IDs evaluated at one Case 2 ``norm_nt`` value."""
    trials_csv = Path(trials_csv)
    trials = pd.read_csv(trials_csv)
    selected = trials.loc[
        np.isclose(trials["norm_nt"].to_numpy(dtype=float), float(norm_nt))
    ].copy()
    seeds = np.sort(selected["seed"].astype(int).unique())

    if expected_count is not None and len(seeds) != int(expected_count):
        raise ValueError(
            f"Expected {expected_count} unique seeds at norm_nt={norm_nt:g} in "
            f"{trials_csv}, but found {len(seeds)}."
        )

    return seeds


def load_manifest_seeds(manifest, expected_count=None):
    """Load the seed set associated with a fixed-case or threshold manifest."""
    trials = pd.read_csv(manifest["trials_csv"])
    exposure_months = int(manifest["exposure_months"])
    if "exposure_months" in trials.columns:
        trials = trials.loc[
            trials["exposure_months"].astype(int) == exposure_months
        ].copy()

    if "norm_nt" in trials.columns and "grid_threshold_norm_nt" in manifest:
        selected_norm = float(manifest["grid_threshold_norm_nt"])
        trials = trials.loc[
            np.isclose(trials["norm_nt"].to_numpy(dtype=float), selected_norm)
        ].copy()

    seeds = np.sort(trials["seed"].astype(int).unique())
    if expected_count is None:
        expected_count = manifest.get(
            "n_seeds_per_normalization",
            manifest.get("n_requested_seeds"),
        )
    if expected_count is not None and len(seeds) != int(expected_count):
        raise ValueError(
            f"Expected {int(expected_count)} seeds for {manifest['case_tag']} "
            f"at {exposure_months} months, but found {len(seeds)}."
        )
    return seeds


def _split_last_group_remainder(native_bins, n_groups):
    """Match the energy grouping used by ``fit_thermal_cosi_sed``."""
    native_bins = np.asarray(native_bins, dtype=int)
    if len(native_bins) < int(n_groups):
        raise ValueError(
            f"Need at least {n_groups} native energy bins, got {len(native_bins)}."
        )

    sizes = [1] * (int(n_groups) - 1)
    sizes.append(len(native_bins) - (int(n_groups) - 1))
    groups = []
    start = 0
    for size in sizes:
        groups.append(native_bins[start : start + size])
        start += size
    return groups


def _poisson_log_likelihood(data, expectation):
    positive_data = data > 0
    if np.any(expectation[positive_data] <= 0):
        return -np.inf
    return float(
        np.sum(data[positive_data] * np.log(expectation[positive_data]))
        - np.sum(expectation)
    )


def _fit_nonnegative_source_scale(
    data,
    source,
    background,
    upper_limit_delta_ts=2.71,
):
    """Return the source scale, detection TS, and one-sided upper limit.

    The null expectation is exactly the fitted background template (zero
    source).  The upper limit is the source scale above the maximum-likelihood
    value at which ``2 * (logL_max - logL) == upper_limit_delta_ts``.
    """
    data = np.asarray(data, dtype=float).ravel()
    source = np.asarray(source, dtype=float).ravel()
    background = np.asarray(background, dtype=float).ravel()

    useful = (source > 0) | (background > 0) | (data > 0)
    data = data[useful]
    source = source[useful]
    background = background[useful]

    if not np.any(source > 0):
        return 0.0, 0.0, np.nan

    def score(scale):
        expectation = background + float(scale) * source
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = np.divide(data, expectation)
            terms = source * (ratio - 1.0)
            total = np.nansum(terms)
        return float(total)

    score_zero = score(0.0)
    if not np.isfinite(score_zero) or score_zero > 0:
        scale_lo = 0.0
        source_sum = float(source.sum())
        scale_hi = max(1.0, 2.0 * float(data.sum()) / source_sum)
        score_hi = score(scale_hi)
        expansions = 0
        while (not np.isfinite(score_hi) or score_hi > 0) and expansions < 80:
            scale_hi *= 2.0
            score_hi = score(scale_hi)
            expansions += 1

        if not np.isfinite(score_hi) or score_hi > 0:
            raise RuntimeError("Could not bracket the source-scale likelihood maximum.")

        for _ in range(100):
            scale_mid = 0.5 * (scale_lo + scale_hi)
            if score(scale_mid) > 0:
                scale_lo = scale_mid
            else:
                scale_hi = scale_mid
        scale_hat = 0.5 * (scale_lo + scale_hi)
    else:
        scale_hat = 0.0

    null_log_like = _poisson_log_likelihood(data, background)
    alt_log_like = _poisson_log_likelihood(
        data,
        background + scale_hat * source,
    )
    raw_ts = 2.0 * (alt_log_like - null_log_like)
    if not np.isfinite(raw_ts):
        ts_value = np.inf if raw_ts > 0 else 0.0
    else:
        ts_value = max(0.0, float(raw_ts))

    upper_limit_delta_ts = float(upper_limit_delta_ts)
    if not np.isfinite(upper_limit_delta_ts) or upper_limit_delta_ts <= 0:
        raise ValueError("upper_limit_delta_ts must be finite and positive.")

    def profile_delta_ts(scale):
        trial_log_like = _poisson_log_likelihood(
            data,
            background + float(scale) * source,
        )
        return float(2.0 * (alt_log_like - trial_log_like))

    source_sum = float(source.sum())
    scale_ul_lo = float(scale_hat)
    scale_ul_hi = max(
        1.0,
        2.0 * max(float(scale_hat), float(data.sum()) / source_sum),
    )
    expansions = 0
    while profile_delta_ts(scale_ul_hi) < upper_limit_delta_ts:
        scale_ul_hi *= 2.0
        expansions += 1
        if expansions >= 100 or not np.isfinite(scale_ul_hi):
            raise RuntimeError("Could not bracket the source-scale upper limit.")

    for _ in range(60):
        scale_ul_mid = 0.5 * (scale_ul_lo + scale_ul_hi)
        if profile_delta_ts(scale_ul_mid) < upper_limit_delta_ts:
            scale_ul_lo = scale_ul_mid
        else:
            scale_ul_hi = scale_ul_mid
    scale_ul95 = 0.5 * (scale_ul_lo + scale_ul_hi)

    return float(scale_hat), ts_value, float(scale_ul95)


def run_sed_seed_ensemble(
    *,
    source_expectation,
    background_expectation,
    seeds,
    n_sed_bins,
    energy_min_keV=100.0,
    energy_max_keV=10000.0,
    background_pseudocount=None,
    case_label="",
):
    """Fit every SED energy group for every supplied Poisson seed.

    The Poisson realization uses the unregularized source plus background
    expectation.  The pseudocount is used only by the fitted background model,
    matching the role it has in ``make_cosi_plugin``.
    """
    source_values = _histogram_values(source_expectation)
    background_values = _histogram_values(background_expectation)
    if source_values.shape != background_values.shape:
        raise ValueError(
            "Source and background expectation shapes differ: "
            f"{source_values.shape} != {background_values.shape}."
        )
    if np.any(source_values < 0) or np.any(background_values < 0):
        raise ValueError("Poisson expectations must be non-negative.")

    pseudocount = (
        float(np.finfo(float).tiny)
        if background_pseudocount is None
        else float(background_pseudocount)
    )
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("background_pseudocount must be finite and positive.")

    em_edges = source_expectation.axes["Em"].edges.to_value("keV")
    native_bins = np.flatnonzero(
        (em_edges[:-1] >= float(energy_min_keV))
        & (em_edges[1:] <= float(energy_max_keV))
    )
    groups = _split_last_group_remainder(native_bins, n_sed_bins)

    background_for_fit = background_values + pseudocount
    total_expectation = source_values + background_values
    zero_background = background_values == 0
    rows = []

    for seed in np.asarray(seeds, dtype=int):
        data_values = np.random.default_rng(int(seed)).poisson(total_expectation)
        for bin_index, group in enumerate(groups, start=1):
            em_slice = slice(int(group[0]), int(group[-1]) + 1)
            data_slice = data_values[em_slice]
            source_slice = source_values[em_slice]
            background_slice = background_values[em_slice]
            fitted_background_slice = background_for_fit[em_slice]
            zero_slice = zero_background[em_slice]
            source_scale, ts_value, source_scale_ul95 = _fit_nonnegative_source_scale(
                data_slice,
                source_slice,
                fitted_background_slice,
            )
            rows.append(
                {
                    "case_label": case_label,
                    "seed": int(seed),
                    "bin_index": int(bin_index),
                    "e_min_keV": float(em_edges[group[0]]),
                    "e_max_keV": float(em_edges[group[-1] + 1]),
                    "e_ref_keV": float(
                        np.sqrt(em_edges[group[0]] * em_edges[group[-1] + 1])
                    ),
                    "source_scale": source_scale,
                    "source_scale_ul95": source_scale_ul95,
                    "ts_value": ts_value,
                    "data_counts": float(data_slice.sum()),
                    "background_counts": float(background_slice.sum()),
                    "excess_counts": float(
                        data_slice.sum() - background_slice.sum()
                    ),
                    "n_zero_background_cells": int(zero_slice.sum()),
                    "data_counts_in_zero_background_cells": float(
                        data_slice[zero_slice].sum()
                    ),
                    "background_pseudocount": pseudocount,
                    "null_source_mode": "exact_zero",
                    "upper_limit_delta_ts": 2.71,
                }
            )

    trials = pd.DataFrame(rows)
    summary_rows = []
    for bin_index, group in trials.groupby("bin_index", sort=True):
        finite_ts = group["ts_value"].replace([np.inf, -np.inf], np.nan)
        summary_rows.append(
            {
                "case_label": case_label,
                "bin_index": int(bin_index),
                "e_min_keV": float(group["e_min_keV"].iloc[0]),
                "e_max_keV": float(group["e_max_keV"].iloc[0]),
                "e_ref_keV": float(group["e_ref_keV"].iloc[0]),
                "n_seeds": int(len(group)),
                "source_scale_q16": float(group["source_scale"].quantile(0.16)),
                "source_scale_median": float(group["source_scale"].median()),
                "source_scale_q84": float(group["source_scale"].quantile(0.84)),
                "source_scale_ul95_q16": float(
                    group["source_scale_ul95"].quantile(0.16)
                ),
                "source_scale_ul95_median": float(
                    group["source_scale_ul95"].median()
                ),
                "source_scale_ul95_q84": float(
                    group["source_scale_ul95"].quantile(0.84)
                ),
                "ts_q16": float(finite_ts.quantile(0.16)),
                "ts_median": float(finite_ts.median()),
                "ts_q84": float(finite_ts.quantile(0.84)),
                "fraction_ts_zero": float(np.isclose(group["ts_value"], 0.0).mean()),
                "fraction_ts_ge_4": float((group["ts_value"] >= 4.0).mean()),
                "fraction_ts_ge_9": float((group["ts_value"] >= 9.0).mean()),
                "fraction_nonfinite_ts": float((~np.isfinite(group["ts_value"])).mean()),
                "excess_counts_q16": float(group["excess_counts"].quantile(0.16)),
                "excess_counts_median": float(group["excess_counts"].median()),
                "excess_counts_q84": float(group["excess_counts"].quantile(0.84)),
                "n_zero_background_cells": int(
                    group["n_zero_background_cells"].iloc[0]
                ),
                "fraction_seeds_with_data_in_zero_background_cells": float(
                    (group["data_counts_in_zero_background_cells"] > 0).mean()
                ),
                "median_data_counts_in_zero_background_cells": float(
                    group["data_counts_in_zero_background_cells"].median()
                ),
                "background_pseudocount": pseudocount,
                "null_source_mode": "exact_zero",
                "upper_limit_delta_ts": 2.71,
            }
        )

    return trials, pd.DataFrame(summary_rows)


def add_scaled_sed_columns(summary, injected_sed_at_reference):
    """Convert source-scale quantiles to SED-flux quantiles."""
    result = summary.copy()
    reference = np.asarray(injected_sed_at_reference, dtype=float)
    if reference.shape != (len(result),):
        raise ValueError(
            "injected_sed_at_reference must contain one value per SED bin."
        )
    result["injected_sed_erg_cm2_s"] = reference
    for suffix in ("q16", "median", "q84"):
        result[f"sed_{suffix}_erg_cm2_s"] = (
            result[f"source_scale_{suffix}"].to_numpy() * reference
        )
        result[f"sed_ul95_{suffix}_erg_cm2_s"] = (
            result[f"source_scale_ul95_{suffix}"].to_numpy() * reference
        )
    return result


def classify_sed_summary(summary, detection_ts=4.0, zero_atol=1e-12):
    """Assign ensemble SED detections, upper limits, and omitted bins.

    Median TS equal to zero is omitted.  Otherwise a bin is a 95% upper
    limit when median TS is below ``detection_ts`` or the 16th-percentile
    recovered source scale is on the zero boundary.  All remaining bins are
    detections.  No bin index or energy range is hard-coded.
    """
    result = summary.copy()
    required = {"ts_median", "source_scale_q16"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"SED summary is missing columns: {sorted(missing)}")

    median_ts = result["ts_median"].to_numpy(dtype=float)
    source_q16 = result["source_scale_q16"].to_numpy(dtype=float)
    if np.any(~np.isfinite(median_ts)) or np.any(~np.isfinite(source_q16)):
        raise ValueError("Non-finite TS or source-scale quantiles cannot be classified.")

    median_ts_is_zero = np.isclose(
        median_ts, 0.0, rtol=0.0, atol=float(zero_atol)
    )
    source_q16_is_zero = np.isclose(
        source_q16, 0.0, rtol=0.0, atol=float(zero_atol)
    )
    result["plot_role"] = np.select(
        [
            median_ts_is_zero,
            (median_ts < float(detection_ts)) | source_q16_is_zero,
        ],
        ["omitted", "upper_limit"],
        default="detection",
    )
    result["classification_rule"] = (
        f"omit median_TS=0; UL median_TS<{float(detection_ts):g} or "
        "source_scale_q16=0; otherwise detection"
    )
    return result


def run_or_load_manifest_sed_ensemble(
    *,
    manifest,
    source_expectation,
    background_expectation,
    injected_shape,
    n_sed_bins=10,
    energy_min_keV=100.0,
    energy_max_keV=10000.0,
    background_pseudocount=None,
    recompute=False,
    detection_ts=4.0,
    expected_seed_count=300,
):
    """Run/cache a manifest's SED ensemble and apply the shared plot policy.

    By default, plotting is refused unless the sensitivity manifest supplies
    exactly 300 unique seeds.  Pass ``expected_seed_count=None`` only for an
    explicitly labeled diagnostic run that is not used for the final SED.
    """
    seeds = load_manifest_seeds(manifest, expected_count=expected_seed_count)
    exposure_months = int(manifest["exposure_months"])
    output_dir = Path(manifest["trials_csv"]).parent
    pseudocount_tag = (
        "tiny"
        if background_pseudocount is None
        else f"{float(background_pseudocount):.0e}".replace("+", "")
    )
    selection_tag = ""
    if "grid_threshold_norm_nt" in manifest:
        norm_tag = f"{float(manifest['grid_threshold_norm_nt']):.4g}".replace(
            ".", "p"
        )
        selection_tag = f"_normNT_{norm_tag}"
    output_stem = (
        f"NGC4151_{manifest['case_tag']}_SED_ensemble_{exposure_months}months_"
        f"{len(seeds)}seeds_{int(n_sed_bins)}bins_"
        f"pseudocount_{pseudocount_tag}{selection_tag}_null_exactZero_ul95"
    )
    trials_file = output_dir / f"{output_stem}_trials.csv"
    summary_file = output_dir / f"{output_stem}_summary.csv"

    if not recompute and trials_file.exists() and summary_file.exists():
        trials = pd.read_csv(trials_file)
        summary = pd.read_csv(summary_file)
    else:
        trials, summary = run_sed_seed_ensemble(
            source_expectation=source_expectation,
            background_expectation=background_expectation,
            seeds=seeds,
            n_sed_bins=n_sed_bins,
            energy_min_keV=energy_min_keV,
            energy_max_keV=energy_max_keV,
            background_pseudocount=background_pseudocount,
            case_label=manifest["case_tag"],
        )
        reference_energy = summary["e_ref_keV"].to_numpy(dtype=float)
        kev_to_erg = 1.602176634e-9
        injected_sed = kev_to_erg * reference_energy**2 * np.asarray(
            [injected_shape.evaluate_at(energy) for energy in reference_energy],
            dtype=float,
        )
        summary = add_scaled_sed_columns(summary, injected_sed)
        trials.to_csv(trials_file, index=False)

    required = {
        "sed_median_erg_cm2_s",
        "sed_ul95_median_erg_cm2_s",
        "null_source_mode",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            f"Cached SED summary predates the exact-null/UL implementation: "
            f"{sorted(missing)}. Rerun with recompute=True."
        )
    summary = classify_sed_summary(summary, detection_ts=detection_ts)
    summary["classification_seed_count"] = int(len(seeds))
    summary.to_csv(summary_file, index=False)
    return trials, summary, trials_file, summary_file


def ensemble_summary_to_sed_dataframe(summary):
    """Adapt an ensemble summary to the columns used by AGN SED plot cells."""
    result = summary.copy()
    roles = result["plot_role"].astype(str).to_numpy()
    result["sed_erg_cm2_s"] = result["sed_median_erg_cm2_s"]
    result["sed_lo_erg_cm2_s"] = result["sed_q16_erg_cm2_s"]
    result["sed_hi_erg_cm2_s"] = np.where(
        roles == "upper_limit",
        result["sed_ul95_median_erg_cm2_s"],
        result["sed_q84_erg_cm2_s"],
    )
    result["ts_value"] = result["ts_median"]
    result["is_upper_limit"] = roles == "upper_limit"
    return result
