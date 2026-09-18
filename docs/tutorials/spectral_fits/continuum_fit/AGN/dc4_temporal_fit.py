"""Response adapters for the all-DC4 light-curve spectral-fit tutorial.

Temporal factors are already in the catalog amplitudes. Only response
livetimes are changed here; the catalog spectra are never rescaled.
"""
from array import array
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astromodels import PointSource
from histpy import Histogram
from cosipy.response import BinnedThreeMLPointSourceResponse
from cosipy.spacecraftfile import SpacecraftHistory
from dc4_lightcurve_response import LightcurveProfile


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


def read_profile(row):
    path = Path(row['lightcurve'])
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != row.get('lightcurve sha256'):
        raise ValueError(f"Light curve changed since catalog generation: {path}")
    opener = gzip.open if path.suffix == '.gz' else open
    times, rates = array('d'), array('d')
    with opener(path, 'rt') as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[0] == 'IP' and fields[1].upper() != 'LINLIN':
                raise ValueError(f"Unsupported interpolation in {path}")
            if fields and fields[0] == 'DP':
                times.append(float(fields[1]))
                rates.append(float(fields[2]))
    return LightcurveProfile(
        np.frombuffer(times, dtype=float), np.frombuffer(rates, dtype=float),
        repeating=row['repeating'], mode=row['normalization mode'],
        bins=int(row['simulation bins']),
    )


def load_recipe(path, catalog_path, model):
    recipe = json.loads(Path(path).read_text())
    if recipe.get('schema') != 'dc4-component-lightcurve-response-v1':
        raise ValueError("Unsupported temporal-response recipe schema")
    if Path(recipe['catalog']).resolve() != Path(catalog_path).resolve():
        raise ValueError("The response recipe belongs to a different catalog")
    if recipe['selection'] not in {'full', 'NGC4151_GTI'} or recipe['time_basis'] != 'elapsed':
        raise ValueError("Expected a full-observation or NGC4151 GTI elapsed-time catalog")
    if set(row['source'] for row in recipe['components']) != set(model.sources):
        raise ValueError("Recipe and catalog source lists differ")
    if any(row['selection'] != recipe['selection'] for row in recipe['components']):
        raise ValueError("Mixed selections in response recipe")
    keys = [(r['source'], r['component']) for r in recipe['components']]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate source/component recipe entries")
    return recipe


def component_name(source, label):
    if label in source.components:
        return label
    # v1 recipes store MEGAlib labels rather than astromodels component names.
    crab_names = {'Crab_Nebula': 'nebula', 'Crab_P1': 'peak1',
                  'Crab_Brg': 'bridge', 'Crab_P2': 'peak2'}
    if source.name == 'crab' and crab_names.get(label) in source.components:
        return crab_names[label]
    raise ValueError(f"Unknown temporal component {source.name}/{label}")


def make_temporal_responses(model, recipe, base_history, gti, data,
                            instrument_response, detector_response,
                            exposure_multiplier=1., full_history=None):
    """Build per-source/component responses from the unscaled GTI history.

    Validate temporal factors over the CATALOG selection, which can differ
    from the analysis GTI. Full catalogs require the original full_history:
    response weights remain L(t)/mean_full while exposure is GTI restricted.
    Exposure repetitions multiply response livetimes, not fluxes.
    Responses are computed lazily by cosipy and reused throughout fitting.
    """
    if not np.isfinite(exposure_multiplier) or exposure_multiplier <= 0:
        raise ValueError("Exposure multiplier must be finite and positive")
    starts, stops = np.asarray(gti.tstart_list.unix), np.asarray(gti.tstop_list.unix)
    duration = float(np.sum(stops-starts))
    if duration <= 0 or np.any(stops <= starts):
        raise ValueError("The analysis GTI must have positive duration")
    catalog_selection = recipe.get('selection', 'NGC4151_GTI')
    if catalog_selection == 'full':
        if full_history is None:
            raise ValueError("Full-observation catalogs require the original full_history")
        reference_starts = np.array([full_history.obstime[0].unix])
        reference_stops = np.array([full_history.obstime[-1].unix])
        if np.any(starts < reference_starts[0]-1e-5) or np.any(stops > reference_stops[0]+1e-5):
            raise ValueError("The analysis GTI extends beyond the catalog observation")
    elif catalog_selection == 'NGC4151_GTI':
        reference_starts, reference_stops = starts, stops
    else:
        raise ValueError(f"Unsupported catalog selection: {catalog_selection}")
    reference_duration = float(np.sum(reference_stops-reference_starts))
    if not np.isfinite(reference_duration) or reference_duration <= 0:
        raise ValueError("Invalid catalog reference duration")
    history_starts, history_stops = base_history.intervals_tstart.unix, base_history.intervals_tstop.unix
    rows_by_source = {}
    for row in recipe['components']:
        rows_by_source.setdefault(row['source'], []).append(row)
    overrides, diagnostics = {}, []
    for key, rows in rows_by_source.items():
        source = model.sources[key]
        for row in rows:
            if not np.isclose(row['selected seconds'], reference_duration, rtol=0., atol=1e-4):
                raise ValueError(f"Reference duration differs from the catalog: {key}")
        if not any(row['lightcurve'] for row in rows):
            continue
        if not isinstance(source, PointSource):
            raise NotImplementedError(f"A temporal extended response is required for {key}")
        if any(r['component'] == 'all' for r in rows) and len(rows) != 1:
            raise ValueError(f"Invalid component grouping for {key}")
        component_responses = {}
        for row in rows:
            applied = float(row['applied factor'])
            if not np.isfinite(applied) or applied <= 0:
                raise ValueError(f"Invalid applied factor for {key}")
            if row['lightcurve']:
                profile = read_profile(row)
                mean = float(profile.integral(reference_starts, reference_stops).sum()/reference_duration)
                gti_mean = float(profile.integral(starts, stops).sum()/duration)
                weights = profile.response_weights(history_starts, history_stops, applied)
                del profile
            else:
                mean, gti_mean, weights = 1., 1., np.ones(base_history.nintervals)
            if not np.isclose(mean, row['time factor'], rtol=1e-8, atol=1e-14):
                raise ValueError(f"Temporal factor differs from catalog for {key}/{row['component']}")
            if not np.isclose(applied, max(mean, 1e-30), rtol=1e-8, atol=0.):
                raise ValueError(f"Applied factor differs from catalog for {key}")
            weighted_live = base_history.livetime * weights * exposure_multiplier
            if not np.any(weighted_live.to_value(u.s) > 0):
                response = ZeroPointSourceResponse(data.axes)
            else:
                history = SpacecraftHistory(base_history.obstime, base_history.attitude,
                                            base_history.location, livetime=weighted_live)
                history.cache_earth_occ = base_history.cache_earth_occ
                response = BinnedThreeMLPointSourceResponse(
                    data=data, instrument_response=instrument_response, sc_history=history,
                    energy_axis=detector_response.axes['Ei'],
                    polarization_axis=detector_response.axes['Pol'] if 'Pol' in detector_response.axes.labels else None,
                    nside=2*data.axes['PsiChi'].nside,
                )
            diagnostics.append({'source': key, 'component': row['component'],
                                'normalization mode': row['normalization mode'],
                                'simulation bins': row['simulation bins'],
                                'catalog selection': catalog_selection,
                                'catalog time factor': mean,
                                'analysis GTI time factor': gti_mean,
                                'response denominator': applied,
                                'weighted response seconds': float(weighted_live.to_value(u.s).sum())})
            if row['component'] == 'all':
                overrides[key] = response
            else:
                name = component_name(source, row['component'])
                if name in component_responses:
                    raise ValueError(f"Duplicate mapped component for {key}/{name}")
                component_responses[name] = response
        if component_responses:
            if set(component_responses) != set(source.components):
                raise ValueError(f"Recipe does not cover all components of {key}")
            overrides[key] = ComponentPointSourceResponse(component_responses)
    return overrides, diagnostics
