"""Reusable point-source response adapters for interval-dependent exposure.

Catalog formats, source names and simulation-specific conventions belong to
the caller. These adapters never change source spectral normalizations.
"""
from copy import deepcopy

import astropy.units as u
import numpy as np
from astromodels import PointSource
from histpy import Histogram

from cosipy.response.threeml_point_source_response import BinnedThreeMLPointSourceResponse
from cosipy.spacecraftfile import SpacecraftHistory

__all__ = ["ZeroPointSourceResponse", "ComponentPointSourceResponse",
           "make_weighted_point_source_response"]

class ZeroPointSourceResponse:
    """No live exposure overlaps this component's emission."""

    def __init__(self, axes):
        self._zero = Histogram(axes)

    @property
    def axes(self):
        return self._zero.axes

    def set_source(self, source):
        if not isinstance(source, PointSource):
            raise TypeError("A point source is required")

    def copy(self):
        return ZeroPointSourceResponse(self.axes)

    def expectation(self, copy=True):
        return self._zero.copy() if copy else self._zero


class ComponentPointSourceResponse:
    """Sum independent temporal responses without splitting the catalog source.

    Child source objects are private copies, so the astromodels node hierarchy
    and fit parameter paths of the original source are preserved.
    """

    def __init__(self, responses):
        if not responses:
            raise ValueError("At least one component response is required")
        self._responses = {name: response.copy() for name, response in responses.items()}
        self._axes = next(iter(responses.values())).axes
        if any(response.axes != self._axes for response in responses.values()):
            raise ValueError("Component response axes do not match")
        self._source = None
        self._last_dict = None
        self._expectation = None

    @property
    def axes(self):
        return self._axes

    def copy(self):
        return ComponentPointSourceResponse(self._responses)

    def set_source(self, source):
        if not isinstance(source, PointSource) or set(source.components) != set(self._responses):
            raise ValueError("The source components do not match the temporal recipe")
        self._source = source
        self._last_dict = None

    def expectation(self, copy=True):
        if self._source is None:
            raise RuntimeError("Call set_source first")
        state = self._source.to_dict()
        if self._last_dict != state:
            total = Histogram(self.axes)
            for name, response in self._responses.items():
                component_source = PointSource(
                    f"{self._source.name}_{name}",
                    l=self._source.position.sky_coord.galactic.l.deg,
                    b=self._source.position.sky_coord.galactic.b.deg,
                    components=[deepcopy(self._source.components[name])],
                )
                response.set_source(component_source)
                total += response.expectation(copy=False)
            self._expectation = total
            self._last_dict = deepcopy(state)
        return self._expectation.copy() if copy else self._expectation


def make_weighted_point_source_response(base_history, weights, data,
                                        instrument_response, detector_response,
                                        exposure_multiplier=1.):
    """Build a lazy point-source response with weighted interval livetimes.

    Parameters
    ----------
    base_history : SpacecraftHistory
        Unscaled spacecraft history, already restricted to the analysis GTI.
    weights : array-like
        One finite, nonnegative dimensionless weight per history interval.
        For a time-averaged catalog amplitude, use
        ``profile.response_weights(starts, stops, catalog_mean)``. The catalog
        mean must refer to the amplitude's reference observation, not
        necessarily the analysis GTI. Attitude stays constant within intervals.
    data, instrument_response, detector_response
        The same binned data and instrument responses used for ordinary
        ``BinnedThreeMLPointSourceResponse`` construction. Incident energy and
        polarization axes are taken from ``detector_response``.
    exposure_multiplier : float
        Positive exposure repetition factor, applied only to livetimes.

    Returns
    -------
    BinnedThreeMLPointSourceResponse or ZeroPointSourceResponse
        A lazy response, or an exact zero when no weighted livetime remains.
        No input history, spectral amplitude or data histogram is modified.
    """
    if not np.isfinite(exposure_multiplier) or exposure_multiplier <= 0:
        raise ValueError("Exposure multiplier must be finite and positive")
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != base_history.livetime.shape
            or np.any(~np.isfinite(weights)) or np.any(weights < 0)):
        raise ValueError("Require one finite nonnegative weight per history interval")
    weighted_live = base_history.livetime * weights * exposure_multiplier
    if not np.all(np.isfinite(weighted_live.to_value(u.s))):
        raise ValueError("Weighted livetimes must be finite")
    if not np.any(weighted_live.to_value(u.s) > 0):
        return ZeroPointSourceResponse(data.axes)
    history = SpacecraftHistory(base_history.obstime, base_history.attitude,
                                base_history.location, livetime=weighted_live)
    history.cache_earth_occ = base_history.cache_earth_occ
    return BinnedThreeMLPointSourceResponse(
        data=data, instrument_response=instrument_response, sc_history=history,
        energy_axis=detector_response.axes['Ei'],
        polarization_axis=(detector_response.axes['Pol']
                           if 'Pol' in detector_response.axes.labels else None),
        nside=2*data.axes['PsiChi'].nside,
    )
