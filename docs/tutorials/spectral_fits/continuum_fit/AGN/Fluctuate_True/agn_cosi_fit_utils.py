"""AGN spectral-fit helpers for the modern cosipy 3ML interface.

The notebooks in this directory were originally written against the removed
legacy COSI 3ML plugin.  These wrappers keep the notebook flow intact while
constructing the current binned likelihood stack explicitly.
"""

from collections import OrderedDict
from collections.abc import Mapping
import json
from pathlib import Path
import tempfile

import numpy as np
import astropy.units as u
from astromodels import Parameter
from histpy import Histogram

from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.data_io import EmCDSBinnedData
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import (
    BinnedInstrumentResponse,
    BinnedThreeMLModelFolding,
    BinnedThreeMLPointSourceResponse,
)
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.statistics import PoissonLikelihood


# Opening the same large detector-response file and recomputing the same
# position/orientation convolution dominated the NGC 4151 notebook runtime.
# The cached objects below are independent of the source spectral parameters,
# the measured counts, and the fitted background normalization, so reusing
# them changes neither the model nor the likelihood.  Keep the point-response
# cache deliberately small so running several notebooks in one kernel does not
# retain an unbounded number of response objects.
_OPEN_DETECTOR_RESPONSES = {}
_POINT_SOURCE_RESPONSE_CACHE = OrderedDict()
_MAX_CACHED_POINT_SOURCE_RESPONSES = 4


def hist_sum(hist):
    contents = hist.contents
    if hasattr(contents, "todense"):
        contents = contents.todense()
    if hasattr(contents, "value"):
        contents = contents.value
    return float(np.asarray(contents).sum())


def _hist_values(hist):
    contents = hist.contents
    if hasattr(contents, "compute"):
        contents = contents.compute()
    if hasattr(contents, "todense"):
        contents = contents.todense()
    if hasattr(contents, "value"):
        contents = contents.value
    return np.asarray(contents, dtype=float)


def _poisson_log_like(data, expectation):
    """Poisson log likelihood without the data-only factorial term."""
    data = np.asarray(data, dtype=float)
    expectation = np.asarray(expectation, dtype=float)
    positive_data = data > 0
    if np.any(expectation[positive_data] <= 0):
        return -np.inf
    return float(
        np.sum(data[positive_data] * np.log(expectation[positive_data]))
        - np.sum(expectation)
    )


def exact_background_only_statistic(plugin, profile_background=False):
    """Return ``-log(L)`` for a model containing exactly zero source.

    This bypasses the response-folded source model entirely, rather than using
    a tiny fixed source normalization.  If requested, the single background
    rate is profiled analytically, subject to its 3ML parameter bounds.  The
    plugin's original background rate is restored before returning.
    """
    parameters = list(plugin.nuisance_parameters.values())
    if len(parameters) != 1:
        raise ValueError(
            "Exact background-only profiling currently requires one background "
            f"rate parameter, found {len(parameters)}."
        )

    parameter = parameters[0]
    original_rate = float(parameter.value)
    em_slice = getattr(plugin._like, "_em_slice", None)

    def sliced_values(histogram):
        if em_slice is not None:
            histogram = histogram.slice[{"Em": em_slice}]
        return _hist_values(histogram)

    try:
        plugin._update_bkg_parameters()
        data = sliced_values(plugin._data)
        background = sliced_values(plugin._bkg.expectation(copy=True))

        fitted_rate = original_rate
        if profile_background:
            background_sum = float(background.sum())
            if background_sum <= 0 or original_rate <= 0:
                raise ValueError("Cannot profile a zero background expectation.")
            fitted_rate = original_rate * float(data.sum()) / background_sum
            if parameter.min_value is not None:
                fitted_rate = max(fitted_rate, float(parameter.min_value))
            if parameter.max_value is not None:
                fitted_rate = min(fitted_rate, float(parameter.max_value))
            parameter.value = fitted_rate
            plugin._update_bkg_parameters()
            background = sliced_values(plugin._bkg.expectation(copy=True))

        log_like = _poisson_log_like(data, background)
        return {
            "statistic": float(-log_like),
            "log_like": float(log_like),
            "background_rate_hz": float(fitted_rate),
            "null_source_mode": "exact_zero",
            "profiled_background": bool(profile_background),
        }
    finally:
        parameter.value = original_rate
        plugin._update_bkg_parameters()


def load_agn_manifest_histograms(manifest_path, background_file, exposure_months):
    """Load one ensemble realization and its exposure-matched background.

    The manifest supplies the source expectation and source-plus-background
    realization.  The common three-month FOV-cut background is scaled only to
    construct the fitted background template; the fluctuated data histogram is
    always loaded directly from the manifest.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open() as stream:
        manifest = json.load(stream)

    exposure_months = int(exposure_months)
    if exposure_months not in (3, 24):
        raise ValueError("AGN dual-exposure fits support 3 or 24 months.")
    manifest_exposure = int(manifest.get("exposure_months", exposure_months))
    if manifest_exposure != exposure_months:
        raise ValueError(
            f"Manifest exposure is {manifest_exposure} months, expected "
            f"{exposure_months}: {manifest_path}"
        )

    source = Histogram.open(manifest["source_expectation_file"])
    data = Histogram.open(manifest["median_data_file"])
    background = Histogram.open(background_file) * (exposure_months / 3.0)
    background = background.project("Em", "Phi", "PsiChi")

    for histogram in (source, data):
        histogram.axes["Em"].axis_scale = background.axes["Em"].axis_scale
    source = source.to(unit=background.unit, update=False)
    data = data.to(unit=background.unit, update=False)
    return manifest, source, data, background


def clone_fit_model_without_external_parameters(fit_results):
    """Clone a fitted source model while removing plugin nuisance parameters."""
    from astromodels import clone_model

    model = clone_model(fit_results.optimized_model)
    source_prefixes = tuple(f"{name}." for name in model.sources)
    external_names = [
        parameter.name
        for path, parameter in list(model.parameters.items())
        if not path.startswith(source_prefixes)
    ]
    for parameter_name in external_names:
        model.remove_external_parameter(parameter_name)
    return model


def fit_cosi_exposure_from_results(
    *,
    reference_results,
    data_hist,
    background_hist,
    dr_path,
    sc_orientation,
    dataset_name,
    background_pseudocount=None,
    additional_plugins=(),
    verbose=False,
):
    """Refit a cloned global model to another COSI exposure.

    ``additional_plugins`` can contain the unchanged BAT/NuSTAR plugins used by
    a joint fit.  All nuisance parameters are reattached by the new plugins;
    only the fitted source parameters are carried over as starting values.
    """
    from threeML import DataList, JointLikelihood

    model = clone_fit_model_without_external_parameters(reference_results)
    nuisance_parameter = make_cosi_background_parameter(dataset_name)
    plugin = COSIPlugin(
        dataset_name,
        dr=dr_path,
        data=data_hist.project("Em", "Phi", "PsiChi"),
        bkg=background_hist.project("Em", "Phi", "PsiChi"),
        sc_orientation=sc_orientation,
        nuisance_param=nuisance_parameter,
        background_pseudocount=background_pseudocount,
        earth_occ=True,
    )
    plugins = DataList(plugin, *tuple(additional_plugins))
    likelihood = JointLikelihood(model, plugins, verbose=verbose)
    try:
        likelihood.fit(quiet=True)
        likelihood._agn_fit_status = "covariance"
    except Exception as covariance_error:
        # ThreeML can finish minimization successfully and then fail while
        # formatting covariance samples when every draw falls outside a bounded
        # parameter's allowed range.  In that case _analysis_results already
        # contains the valid maximum-likelihood model and statistic, so retain
        # it without repeating the minimization.  Only retry without covariance
        # if the original failure happened before results existed.
        results = getattr(likelihood, "_analysis_results", None)
        if results is None:
            likelihood.fit(quiet=True, compute_covariance=False)
            results = likelihood.results
        warning = (
            "Covariance-based uncertainty sampling failed "
            f"({type(covariance_error).__name__}: {covariance_error}). "
            "The best-fit values and likelihood are valid, but parameter "
            "uncertainties are unavailable for this exposure fit."
        )
        likelihood._agn_fit_status = "best-fit-only fallback"
        results._agn_uncertainty_warning = warning
    return likelihood, plugin


def joint_likelihood_statistic(joint_likelihood, dataset_name=None):
    """Return a fitted ``-log(likelihood)`` total or one dataset component."""
    statistic = joint_likelihood.results.get_statistic_frame()["-log(likelihood)"]
    if dataset_name is not None:
        return float(statistic.loc[dataset_name])
    if "total" in statistic.index:
        return float(statistic.loc["total"])
    return float(statistic.sum())


def cosi_source_detection_ts(joint_likelihood, cosi_plugin):
    """Compute exact-zero-source COSI TS with a profiled background."""
    null = exact_background_only_statistic(cosi_plugin, profile_background=True)
    alternative = joint_likelihood_statistic(
        joint_likelihood, dataset_name=cosi_plugin.name
    )
    ts_value = max(0.0, 2.0 * (float(null["statistic"]) - alternative))
    return ts_value, null


def model_improvement_ts(reference_likelihood, test_likelihood):
    """Return ``2 [log L(test) - log L(reference)]``, clamped at zero."""
    return max(
        0.0,
        2.0
        * (
            joint_likelihood_statistic(reference_likelihood)
            - joint_likelihood_statistic(test_likelihood)
        ),
    )


def draw_energy_hist_mev(histogram, ax, **kwargs):
    """Draw a measured-energy projection with its x-axis values in MeV."""

    projected = histogram.project("Em")
    projected.axes["Em"] = projected.axes["Em"].to(u.MeV)
    return projected.draw(ax, **kwargs)


def _project_em_phi_psichi(hist):
    if isinstance(hist, Histogram):
        projected = hist.project("Em", "Phi", "PsiChi")
    else:
        projected = hist
    return projected.copy()


def _open_response(response):
    if isinstance(response, FullDetectorResponse):
        return response

    response_path = str(Path(response).expanduser().resolve())
    cached_response = _OPEN_DETECTOR_RESPONSES.get(response_path)
    if cached_response is not None:
        return cached_response

    try:
        opened_response = FullDetectorResponse.open(response_path)
    except RuntimeError as exc:
        if "Response format is version 1" not in str(exc):
            raise

        original_rsp_version = FullDetectorResponse.rsp_version
        try:
            FullDetectorResponse.rsp_version = 1
            opened_response = FullDetectorResponse.open(response_path)
        finally:
            FullDetectorResponse.rsp_version = original_rsp_version

    _OPEN_DETECTOR_RESPONSES[response_path] = opened_response
    return opened_response


def clear_agn_response_caches():
    """Release detector and point-source responses retained by this module.

    Notebook users normally do not need this.  It is useful after switching to
    different response/orientation files repeatedly in one long-lived kernel.
    """

    _POINT_SOURCE_RESPONSE_CACHE.clear()
    _OPEN_DETECTOR_RESPONSES.clear()


def _point_source_response_cache_key(source_response):
    """Return an exact-geometry key for a reusable point-source response."""

    source = getattr(source_response, "_source", None)
    if source is None:
        return None

    coord = source.position.sky_coord.icrs
    instrument_response = getattr(source_response, "_response", None)
    detector_response = getattr(instrument_response, "_dr", None)
    orientation = getattr(source_response, "_sc_ori", None)
    axes = getattr(source_response, "axes", None)

    axis_signature = []
    for axis in axes:
        edges = np.asarray(axis)
        axis_signature.append(
            (
                type(axis).__name__,
                getattr(axis, "label", None),
                str(getattr(axis, "unit", "")),
                getattr(axis, "axis_scale", None),
                edges.dtype.str,
                edges.shape,
                edges.tobytes(),
                getattr(axis, "nside", None),
                str(getattr(axis, "scheme", "")),
                repr(getattr(axis, "coordsys", None)),
            )
        )

    # Object identities intentionally make this cache conservative.  Sharing
    # occurs only among plugins in the same notebook that use the same opened
    # detector response, the same spacecraft-history object, and identical
    # binned axes.  Different exposures can therefore never share a sky fold.
    return (
        id(detector_response),
        id(orientation),
        int(getattr(source_response, "_nside", 0) or 0),
        float(coord.ra.deg),
        float(coord.dec.deg),
        tuple(axis_signature),
    )


def _prime_point_source_response_cache(model_folding):
    """Install cached sky folds before the first likelihood evaluation."""

    model_folding._cache_source_responses()
    for source_response in model_folding._source_responses.values():
        key = _point_source_response_cache_key(source_response)
        if key is None or key not in _POINT_SOURCE_RESPONSE_CACHE:
            continue

        source_response._psr = _POINT_SOURCE_RESPONSE_CACHE[key]
        source_response._last_convolved_source_skycoord = (
            source_response._source.position.sky_coord.copy()
        )
        source_response._expectation = None
        source_response._last_convolved_source_dict = None
        _POINT_SOURCE_RESPONSE_CACHE.move_to_end(key)

    # _cache_source_responses() records the current model before an expectation
    # exists.  Invalidate only that dictionary snapshot so the first likelihood
    # call computes the spectrum with the installed sky response.
    model_folding._cached_model_dict = None
    model_folding._expectation.clear()


def _store_point_source_responses(model_folding):
    """Retain newly calculated sky folds for later plugins in this kernel."""

    for source_response in model_folding._source_responses.values():
        key = _point_source_response_cache_key(source_response)
        point_response = getattr(source_response, "_psr", None)
        if key is None or point_response is None:
            continue

        _POINT_SOURCE_RESPONSE_CACHE[key] = point_response
        _POINT_SOURCE_RESPONSE_CACHE.move_to_end(key)

    while len(_POINT_SOURCE_RESPONSE_CACHE) > _MAX_CACHED_POINT_SOURCE_RESPONSES:
        _POINT_SOURCE_RESPONSE_CACHE.popitem(last=False)


def open_spacecraft_history(filename, *args, **kwargs):
    """Open a spacecraft history, tolerating legacy ORI files missing ``EN``."""

    try:
        return SpacecraftHistory.open(filename, *args, **kwargs)
    except ValueError as exc:
        if ".ori file must end with final line 'EN'!" not in str(exc):
            raise

    source = Path(filename)
    contents = source.read_text()

    with tempfile.NamedTemporaryFile("w", suffix=".ori", delete=True) as tmp:
        tmp.write(contents)
        if not contents.endswith("\n"):
            tmp.write("\n")
        tmp.write("EN\n")
        tmp.flush()
        return SpacecraftHistory.open(tmp.name, *args, **kwargs)


def scale_spacecraft_livetime(sc_history, exposure_multiplier):
    """Scale exposure while preserving the sampled pointing distribution.

    Multiplying simulated source and background histograms represents a longer
    exposure only when the response integration uses the same multiplier.  A
    new spacecraft history with scaled interval livetimes is equivalent to
    repeating the selected pointing history without making the source brighter.
    """

    exposure_multiplier = float(exposure_multiplier)
    if not np.isfinite(exposure_multiplier) or exposure_multiplier <= 0:
        raise ValueError("Exposure multiplier must be finite and positive.")

    if np.isclose(exposure_multiplier, 1.0):
        return sc_history

    scaled_history = SpacecraftHistory(
        sc_history.obstime,
        sc_history.attitude,
        sc_history.location,
        livetime=sc_history.livetime * exposure_multiplier,
    )
    scaled_history.cache_earth_occ = sc_history.cache_earth_occ
    return scaled_history


def _background_rate_parameter(label, bkg_hist, sc_history, nuisance_param=None):
    livetime = sc_history.cumulative_livetime().to_value(u.s)
    if livetime <= 0:
        raise ValueError("Spacecraft history has non-positive cumulative livetime.")

    base_rate = hist_sum(bkg_hist) / livetime

    if nuisance_param is None:
        return Parameter(
            label,
            base_rate,
            min_value=0.0,
            max_value=max(base_rate * 5.0, np.finfo(float).tiny),
            delta=max(base_rate * 0.05, np.finfo(float).tiny),
            unit=u.Hz,
            desc="Background rate parameter for COSI",
        )

    value = float(getattr(nuisance_param, "value", 1.0))
    min_value = getattr(nuisance_param, "min_value", None)
    max_value = getattr(nuisance_param, "max_value", None)
    delta = getattr(nuisance_param, "delta", None)

    rate_value = max(base_rate * value, np.finfo(float).tiny)
    rate_min = None if min_value is None else max(base_rate * float(min_value), 0.0)
    rate_max = None if max_value is None else max(base_rate * float(max_value), rate_value)
    rate_delta = None if delta is None else max(abs(base_rate * float(delta)), np.finfo(float).tiny)

    return Parameter(
        label,
        rate_value,
        min_value=rate_min,
        max_value=rate_max,
        delta=rate_delta,
        unit=u.Hz,
        desc=getattr(nuisance_param, "desc", "Background rate parameter for COSI"),
    )


class EnergySlicePoissonLikelihood(PoissonLikelihood):
    """Poisson likelihood evaluated on a measured-energy slice."""

    def __init__(self, data, response, bkg, em_slice):
        super().__init__(data, response, bkg)
        self._em_slice = em_slice

    @property
    def nobservations(self):
        return self._data.data.slice[{"Em": self._em_slice}].contents.size

    def get_log_like(self):
        expectation = self._response.expectation(copy=self.has_bkg)

        if self.has_bkg:
            expectation += self._bkg.expectation(copy=False)

        expectation = expectation.slice[{"Em": self._em_slice}]
        data = self._data.data.slice[{"Em": self._em_slice}]

        expectation_contents = expectation.contents
        data_contents = data.contents

        if hasattr(expectation_contents, "todense"):
            expectation_contents = expectation_contents.todense()
        if hasattr(data_contents, "todense"):
            data_contents = data_contents.todense()
        if hasattr(expectation_contents, "value"):
            expectation_contents = expectation_contents.value
        if hasattr(data_contents, "value"):
            data_contents = data_contents.value

        expectation_contents = np.asarray(expectation_contents, dtype=float)
        data_contents = np.asarray(data_contents, dtype=float)

        return float(np.nansum(data_contents * np.log(expectation_contents) - expectation_contents))


class AGNCOSIPlugin(ThreeMLPluginInterface):
    """ThreeML plugin with source expectations exposed for old plot cells."""

    class _NuisanceParameterView(dict):
        def __init__(self, plugin, parameters):
            super().__init__(parameters)
            self._plugin = plugin

        def __getitem__(self, key):
            if dict.__contains__(self, key):
                return dict.__getitem__(self, key)

            prefixed_key = self._plugin._add_prefix_name(key)
            if dict.__contains__(self, prefixed_key):
                return dict.__getitem__(self, prefixed_key)

            raise KeyError(key)

    def __init__(self, name, likelihood, response, bkg=None):
        super().__init__(name, likelihood, response, bkg)
        self._expected_counts = {}
        self._signal = None
        self._agn_point_response_cache_primed = False

    @property
    def nuisance_parameters(self):
        return AGNCOSIPlugin._NuisanceParameterView(
            self,
            self._threeml_bkg_parameters,
        )

    def _refresh_expected_counts(self):
        source_responses = getattr(self._response, "_source_responses", {})
        expected_counts = {}
        signal = None

        for source_name, source_response in source_responses.items():
            expectation = source_response.expectation(copy=True)
            expected_counts[source_name] = expectation

            contents = expectation.project("Em", "Phi", "PsiChi").contents
            if hasattr(contents, "todense"):
                contents = contents.todense()
            if hasattr(contents, "value"):
                contents = contents.value
            contents = np.asarray(contents, dtype=float)

            signal = contents.copy() if signal is None else signal + contents

        self._expected_counts = expected_counts
        self._signal = signal

    def set_model(self, model):
        super().set_model(model)
        self._agn_point_response_cache_primed = False

    def get_log_like(self):
        if not self._agn_point_response_cache_primed:
            _prime_point_source_response_cache(self._response)
            self._agn_point_response_cache_primed = True

        value = super().get_log_like()
        _store_point_source_responses(self._response)
        self._refresh_expected_counts()
        return value


def make_cosi_plugin(
    name,
    dr,
    data,
    bkg,
    sc_orientation,
    nuisance_param=None,
    background_label=None,
    background_pseudocount=None,
    em_slice=None,
    **_,
):
    """Build the current binned COSI 3ML plugin from old notebook inputs.

    ``background_pseudocount`` is added to every background histogram cell
    before ``FreeNormBinnedBackground`` normalizes the template.  ``None``
    preserves the historical machine-tiny workaround; a finite positive value
    provides non-negligible support in cells that were empty in the finite
    background sample.
    """

    data_hist = _project_em_phi_psichi(data)
    bkg_hist = _project_em_phi_psichi(bkg)
    raw_bkg_hist = bkg_hist.copy()

    if background_pseudocount is None:
        background_pseudocount = np.finfo(float).tiny
    else:
        background_pseudocount = float(background_pseudocount)
        if (
            not np.isfinite(background_pseudocount)
            or background_pseudocount <= 0
        ):
            raise ValueError(
                "background_pseudocount must be finite and strictly positive."
            )

    # FreeNormBinnedBackground normalizes the template internally.  Adding the
    # pseudocount first preserves that behavior while removing exact zeros.
    bkg_hist += background_pseudocount

    response = _open_response(dr)
    data_interface = EmCDSBinnedData(data_hist)
    bkg_model = FreeNormBinnedBackground(
        {background_label or getattr(nuisance_param, "name", "background"): bkg_hist},
        sc_history=sc_orientation,
        copy=False,
    )

    instrument_response = BinnedInstrumentResponse(response, data_interface)
    point_source_response = BinnedThreeMLPointSourceResponse(
        data=data_interface,
        instrument_response=instrument_response,
        sc_history=sc_orientation,
        energy_axis=response.axes["Ei"],
        polarization_axis=response.axes["Pol"] if "Pol" in response.axes.labels else None,
        nside=2 * data_interface.axes["PsiChi"].nside,
    )
    model_folding = BinnedThreeMLModelFolding(
        data=data_interface,
        point_source_response=point_source_response,
    )

    if em_slice is None:
        likelihood = PoissonLikelihood(data_interface, model_folding, bkg_model)
    else:
        likelihood = EnergySlicePoissonLikelihood(data_interface, model_folding, bkg_model, em_slice)

    plugin = AGNCOSIPlugin(name, likelihood, model_folding, bkg_model)
    plugin._data = data_hist.copy()
    plugin._raw_bkg_hist = raw_bkg_hist
    plugin._bkg_hist = bkg_hist.copy()
    plugin._background_pseudocount = background_pseudocount

    for label in bkg_model.labels:
        plugin.bkg_parameter[label] = _background_rate_parameter(
            label,
            bkg_hist,
            sc_orientation,
            nuisance_param=nuisance_param,
        )

    return plugin


class COSIPlugin(AGNCOSIPlugin):
    """Drop-in wrapper for the removed COSI plugin, backed by modern cosipy."""

    def __init__(self, name, dr, data, bkg, sc_orientation, nuisance_param=None, **kwargs):
        plugin = make_cosi_plugin(
            name=name,
            dr=dr,
            data=data,
            bkg=bkg,
            sc_orientation=sc_orientation,
            nuisance_param=nuisance_param,
            **kwargs,
        )
        self.__dict__ = plugin.__dict__


class EnergyRangeCOSIPlugin(AGNCOSIPlugin):
    """Measured-energy-slice COSI plugin for bin-by-bin SED fits."""

    def __init__(self, name, dr, data, bkg, sc_orientation, em_slice, nuisance_param=None, **kwargs):
        plugin = make_cosi_plugin(
            name=name,
            dr=dr,
            data=data,
            bkg=bkg,
            sc_orientation=sc_orientation,
            nuisance_param=nuisance_param,
            em_slice=em_slice,
            **kwargs,
        )
        self.__dict__ = plugin.__dict__


def make_cosi_background_parameter(dataset_name):
    return Parameter(
        f"background_{dataset_name}",
        1.0,
        min_value=0.0,
        max_value=5.0,
        delta=0.05,
        desc="Background scale converted to a COSI background rate",
    )


def _summary_value(value):
    """Format numerical summary values compactly and reproducibly."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(numeric_value):
        return str(numeric_value)

    return f"{numeric_value:.10g}"


def _summary_case_value(values, case_label, default=None):
    if isinstance(values, Mapping):
        return values.get(case_label, default)
    if values is None:
        return default
    return values


def _injected_parameter_rows(model_or_components, prefix=""):
    """Return injected Astromodels parameters from a model, shape, or mapping."""

    if model_or_components is None:
        return []

    if isinstance(model_or_components, Mapping):
        rows = []
        for label, component in model_or_components.items():
            component_prefix = f"{prefix}.{label}" if prefix else str(label)
            rows.extend(_injected_parameter_rows(component, component_prefix))
        return rows

    parameters = getattr(model_or_components, "parameters", None)
    if not isinstance(parameters, Mapping):
        return []

    rows = []
    for name, parameter in parameters.items():
        parameter_name = f"{prefix}.{name}" if prefix else str(name)
        rows.append(
            (
                parameter_name,
                getattr(parameter, "value", "N/A"),
                getattr(parameter, "unit", ""),
                bool(getattr(parameter, "fix", False)),
            )
        )
    return rows


def _fitted_parameter_rows(fit_results, confidence_level):
    """Return fitted values and equal-tail errors, with a safe fallback."""

    uncertainty_warning = getattr(fit_results, "_agn_uncertainty_warning", None)
    if uncertainty_warning is not None:
        optimized_model = getattr(fit_results, "optimized_model", None)
        parameters = getattr(optimized_model, "free_parameters", {})
        rows = []
        for parameter_path, parameter in parameters.items():
            rows.append(
                (
                    str(parameter_path),
                    getattr(parameter, "value", "N/A"),
                    "N/A",
                    "N/A",
                    getattr(parameter, "unit", ""),
                )
            )
        return rows, uncertainty_warning

    try:
        frame = fit_results.get_data_frame(
            error_type="equal tail",
            cl=float(confidence_level),
        )
        rows = []
        for parameter_path, row in frame.iterrows():
            rows.append(
                (
                    str(parameter_path),
                    row.get("value", "N/A"),
                    row.get("negative_error", "N/A"),
                    row.get("positive_error", "N/A"),
                    row.get("unit", ""),
                )
            )
        return rows, None
    except Exception as exc:
        rows = []
        parameters = getattr(fit_results, "_free_parameters", None)
        if not isinstance(parameters, Mapping):
            optimized_model = getattr(fit_results, "optimized_model", None)
            parameters = getattr(optimized_model, "free_parameters", {})
        for parameter_path, parameter in parameters.items():
            rows.append(
                (
                    str(parameter_path),
                    getattr(parameter, "value", "N/A"),
                    "N/A",
                    "N/A",
                    getattr(parameter, "unit", ""),
                )
            )
        return rows, f"{type(exc).__name__}: {exc}"


def save_agn_fit_summary(
    output_path,
    fit_results,
    injected_models,
    ts_values,
    exposure_months,
    significance_values=None,
    extra_statistics=None,
    confidence_level=0.68,
    notes=None,
):
    """Write fitted and injected AGN parameters to a plain-text summary.

    Parameters may be supplied either as scalars or as mappings keyed by a fit
    case such as ``"ec200"``.  Fitted uncertainties are the equal-tail
    intervals stored in the ThreeML results at ``confidence_level``.  When a
    covariance or variate interval is unavailable, best-fit values are still
    written and the missing uncertainties are reported explicitly.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    case_labels = []
    for values in (fit_results, injected_models, ts_values, exposure_months):
        if isinstance(values, Mapping):
            for label in values:
                if label not in case_labels:
                    case_labels.append(label)

    if not case_labels:
        case_labels = ["global_fit"]

    lines = [
        "AGN GLOBAL FIT SUMMARY",
        f"Confidence level for fitted uncertainties: {100 * float(confidence_level):.1f}%",
    ]
    if notes:
        lines.extend([f"Notes: {notes}"])

    for case_label in case_labels:
        result = _summary_case_value(fit_results, case_label)
        injected = _summary_case_value(injected_models, case_label)
        ts_value = _summary_case_value(ts_values, case_label)
        exposure = _summary_case_value(exposure_months, case_label)
        significance = _summary_case_value(significance_values, case_label)

        if significance is None and ts_value is not None:
            significance = np.sqrt(max(float(ts_value), 0.0))

        lines.extend(
            [
                "",
                "=" * 88,
                f"FIT CASE: {case_label}",
                "=" * 88,
                f"Exposure months: {_summary_value(exposure) if exposure is not None else 'N/A'}",
                f"Global TS: {_summary_value(ts_value) if ts_value is not None else 'N/A'}",
                (
                    "Global significance (sqrt(TS)): "
                    f"{_summary_value(significance)} sigma"
                    if significance is not None
                    else "Global significance (sqrt(TS)): N/A"
                ),
            ]
        )

        case_extra_statistics = _summary_case_value(extra_statistics, case_label, {})
        if isinstance(case_extra_statistics, Mapping) and case_extra_statistics:
            lines.append("Additional statistics:")
            for label, value in case_extra_statistics.items():
                lines.append(f"  {label}: {_summary_value(value)}")

        lines.extend(["", "FITTED PARAMETERS (value, negative error, positive error, unit)"])
        if result is None:
            lines.append("  N/A: no global fit is performed for this case.")
        else:
            fitted_rows, error_message = _fitted_parameter_rows(result, confidence_level)
            if error_message:
                lines.append(f"  Uncertainty warning: {error_message}")
            if not fitted_rows:
                lines.append("  N/A: no free fitted parameters were found.")
            for parameter_path, value, negative_error, positive_error, unit in fitted_rows:
                lines.append(
                    "  "
                    f"{parameter_path}: {_summary_value(value)}, "
                    f"{_summary_value(negative_error)}, "
                    f"{_summary_value(positive_error)}, {unit}"
                )

        lines.extend(["", "INJECTED PARAMETERS (value, unit, fixed)"])
        injected_rows = _injected_parameter_rows(injected)
        if not injected_rows:
            lines.append("  N/A: no injected model was supplied.")
        for parameter_path, value, unit, fixed in injected_rows:
            lines.append(
                f"  {parameter_path}: {_summary_value(value)}, {unit}, {fixed}"
            )

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
