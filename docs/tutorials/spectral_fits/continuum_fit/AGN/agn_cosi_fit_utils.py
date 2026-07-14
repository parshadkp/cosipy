"""AGN spectral-fit helpers for the modern cosipy 3ML interface.

The notebooks in this directory were originally written against the removed
legacy COSI 3ML plugin.  These wrappers keep the notebook flow intact while
constructing the current binned likelihood stack explicitly.
"""

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


def hist_sum(hist):
    contents = hist.contents
    if hasattr(contents, "todense"):
        contents = contents.todense()
    if hasattr(contents, "value"):
        contents = contents.value
    return float(np.asarray(contents).sum())


def _project_em_phi_psichi(hist):
    if isinstance(hist, Histogram):
        projected = hist.project("Em", "Phi", "PsiChi")
    else:
        projected = hist
    return projected.copy()


def _open_response(response):
    if isinstance(response, FullDetectorResponse):
        return response

    try:
        return FullDetectorResponse.open(str(response))
    except RuntimeError as exc:
        if "Response format is version 1" not in str(exc):
            raise

    original_rsp_version = FullDetectorResponse.rsp_version
    try:
        FullDetectorResponse.rsp_version = 1
        return FullDetectorResponse.open(str(response))
    finally:
        FullDetectorResponse.rsp_version = original_rsp_version


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

    def get_log_like(self):
        value = super().get_log_like()
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
    em_slice=None,
    **_,
):
    """Build the current binned COSI 3ML plugin from old notebook inputs."""

    data_hist = _project_em_phi_psichi(data)
    bkg_hist = _project_em_phi_psichi(bkg)

    # Current FreeNormBinnedBackground normalizes internally; this tiny offset
    # mirrors the existing tutorial workaround for zero-probability bins.
    bkg_hist += np.finfo(float).tiny

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
    plugin._bkg_hist = bkg_hist.copy()

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
