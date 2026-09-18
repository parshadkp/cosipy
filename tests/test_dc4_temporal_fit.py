"""Small-response tests; no DC4 event files or response cubes required."""
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import astropy.units as u


@pytest.fixture(scope='module')
def api():
    import json
    from types import ModuleType
    from copy import deepcopy
    import astropy.units as u
    from astromodels import PointSource
    from histpy import Histogram
    path = Path(__file__).resolve().parents[1] / (
        'docs/tutorials/spectral_fits/continuum_fit/AGN/'
        'NGC4151_DC4_Spectral_Fit_Source_Cat_lightcurve.ipynb')
    module = ModuleType('dc4_tutorial_setup')
    module.__dict__.update(Path=Path, np=np, u=u, PointSource=PointSource,
                           Histogram=Histogram, deepcopy=deepcopy)
    for cell in json.loads(path.read_text())['cells']:
        if 'dc4-setup' in cell.get('metadata', {}).get('tags', []):
            exec(compile(''.join(cell['source']), str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture
def setup(api):
    from histpy import Axis, Axes, HealpixAxis
    from astromodels import PointSource, Powerlaw, SpectralComponent
    axes = Axes([Axis([100., 200., 400.], unit=u.keV, label='Em'),
                 Axis([0., 180.], unit=u.deg, label='Phi'),
                 HealpixAxis(nside=1, scheme='ring', coordsys='galactic', label='PsiChi')])
    source = PointSource('crab', l=10., b=20., components=[
        SpectralComponent('nebula', Powerlaw(K=1., index=-2., piv=100.)),
        SpectralComponent('peak1', Powerlaw(K=2., index=-2., piv=100.)),
    ])
    class ToyResponse:
        def __init__(self, weight=1., **kwargs):
            self.axes = axes
            self.weight = (kwargs['sc_history'].livetime.to_value(u.s).sum()
                           if 'sc_history' in kwargs else weight)
        def copy(self):
            return ToyResponse(self.weight)
        def set_source(self, source):
            self.source = source
        def expectation(self, copy=True):
            values = sum(np.asarray(c.shape(np.array([150., 300.]))) for c in self.source.components.values())
            return api.Histogram(self.axes, contents=np.broadcast_to(values[:, None, None], (2, 1, 12))*self.weight)
    return axes, source, ToyResponse


def test_component_response_updates_and_keeps_original_nodes(api, setup):
    axes, source, response = setup
    adapter = api.ComponentPointSourceResponse({'nebula': response(1.), 'peak1': response(3.)})
    adapter.set_source(source)
    before = source.to_dict()
    first = adapter.expectation().contents.copy()
    assert source.to_dict() == before
    assert source.components['peak1'].shape.K.path.startswith('crab.spectrum.peak1.')
    source.components['peak1'].shape.K.value *= 2.
    second = adapter.expectation().contents
    np.testing.assert_allclose(second/first, 13./7.)
    independent = adapter.copy()
    clone = api.deepcopy(source)
    independent.set_source(clone)
    clone.components['nebula'].shape.K.value = 10.
    assert not np.allclose(independent.expectation().contents, adapter.expectation().contents)


def test_component_coverage_and_zero_response(api, setup):
    axes, source, response = setup
    adapter = api.ComponentPointSourceResponse({'nebula': response()})
    with pytest.raises(ValueError, match='components'):
        adapter.set_source(source)
    zero = api.ZeroPointSourceResponse(axes)
    zero.set_source(source)
    assert zero.expectation().contents.sum() == 0
    changed = zero.expectation()
    changed += 1.
    assert zero.expectation().contents.sum() == 0


def test_hash_verification(api, tmp_path):
    path = tmp_path/'lc.dat'
    path.write_text('IP LINLIN\nDP 1 1\nDP 2 1\nEN\n')
    row = {'lightcurve': str(path), 'lightcurve sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'repeating': False, 'normalization mode': 'global_megalib', 'simulation bins': 1}
    profile = api.LightcurveProfile.from_dat(
        path, expected_sha256=row['lightcurve sha256'])
    np.testing.assert_allclose(profile.integral([0.], [4.]), [1.])
    path.write_text('DP 1 2\nDP 2 2\n')
    with pytest.raises(ValueError, match='changed'):
        api.LightcurveProfile.from_dat(path, expected_sha256=row['lightcurve sha256'])


def test_temporal_exposure_scaling_and_zero_overlap(api, setup, tmp_path, monkeypatch):
    axes, _, response = setup
    from astromodels import Model, PointSource, Powerlaw
    source = PointSource('transient', l=1., b=2., spectral_shape=Powerlaw(K=.25, piv=100.))
    model = Model(source)
    path = tmp_path/'lc.dat'
    path.write_text('DP 1 1\nDP 2 1\n')
    row = {'source': 'transient', 'component': 'all', 'selection': 'NGC4151_GTI',
           'selected seconds': 4., 'time factor': .25, 'applied factor': .25,
           'lightcurve': str(path), 'lightcurve sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'normalization mode': 'global_megalib', 'simulation bins': 1, 'repeating': False}
    gti = SimpleNamespace(tstart_list=SimpleNamespace(unix=np.array([0.])),
                          tstop_list=SimpleNamespace(unix=np.array([4.])))
    history = SimpleNamespace(intervals_tstart=SimpleNamespace(unix=np.array([0., 2.])),
                              intervals_tstop=SimpleNamespace(unix=np.array([2., 4.])),
                              nintervals=2, livetime=np.array([2., 2.])*u.s,
                              obstime=None, attitude=None, location=None, cache_earth_occ=False)
    from cosipy.response import threeml_temporal_response as temporal_api
    monkeypatch.setattr(temporal_api, 'SpacecraftHistory', lambda *a, **k: SimpleNamespace(livetime=k['livetime']))
    monkeypatch.setattr(temporal_api, 'BinnedThreeMLPointSourceResponse', response)
    data = SimpleNamespace(axes=axes)
    # No polarization and only Ei is inspected by the response factory.
    from histpy import Axes, Axis
    detector = SimpleNamespace(axes=Axes([Axis([100.,200.,400.],unit=u.keV,label='Ei')]))
    state = source.to_dict()
    for multiplier in [1., 8.]:
        overrides, diagnostics = api.make_temporal_responses(
            model, {'components':[row]}, history, gti, data, None, detector, multiplier)
        assert np.isclose(overrides['transient'].weight, 4.*multiplier)
        assert np.isclose(diagnostics[0]['catalog time factor'], .25)
        assert source.to_dict() == state  # Amplitudes are never multiplied again.
    # A full-mean catalog can be restricted to a new GTI without changing K.
    # The burst lies wholly inside [0,2], so mean_GTI=.5 but mean_full=.25.
    cut_history = SimpleNamespace(
        intervals_tstart=SimpleNamespace(unix=np.array([0.])),
        intervals_tstop=SimpleNamespace(unix=np.array([2.])),
        nintervals=1, livetime=np.array([2.])*u.s,
        obstime=None, attitude=None, location=None, cache_earth_occ=False)
    cut_gti = SimpleNamespace(tstart_list=SimpleNamespace(unix=np.array([0.])),
                              tstop_list=SimpleNamespace(unix=np.array([2.])))
    full_history = SimpleNamespace(obstime=[SimpleNamespace(unix=0.), SimpleNamespace(unix=4.)])
    full_row = dict(row, selection='full')
    full_recipe = {'selection':'full', 'components':[full_row]}
    with pytest.raises(ValueError, match='full_history'):
        api.make_temporal_responses(model, full_recipe, cut_history, cut_gti, data, None, detector)
    for multiplier in [1., 8.]:
        overrides, diagnostics = api.make_temporal_responses(
            model, full_recipe, cut_history, cut_gti, data, None, detector,
            multiplier, full_history=full_history)
        full_expected = source.spectrum.main.Powerlaw.K.value * overrides['transient'].weight
        assert np.isclose(full_expected, multiplier)
        assert diagnostics[0]['catalog time factor'] == .25
        assert diagnostics[0]['analysis GTI time factor'] == .5
        assert diagnostics[0]['response denominator'] == .25
        assert source.to_dict() == state
        # The equivalent GTI-calibrated catalog gives the same prediction.
        gti_row = dict(row, **{'selected seconds':2., 'time factor':.5, 'applied factor':.5})
        gti_overrides, _ = api.make_temporal_responses(
            model, {'components':[gti_row]}, cut_history, cut_gti, data, None, detector, multiplier)
        assert np.isclose(full_expected, .5*gti_overrides['transient'].weight)
    off_gti = SimpleNamespace(tstart_list=SimpleNamespace(unix=np.array([2.])),
                              tstop_list=SimpleNamespace(unix=np.array([4.])))
    off_history = SimpleNamespace(**vars(cut_history))
    off_history.intervals_tstart = SimpleNamespace(unix=np.array([2.]))
    off_history.intervals_tstop = SimpleNamespace(unix=np.array([4.]))
    off_overrides, off_diagnostics = api.make_temporal_responses(
        model, full_recipe, off_history, off_gti, data, None, detector, full_history=full_history)
    assert isinstance(off_overrides['transient'], api.ZeroPointSourceResponse)
    assert off_diagnostics[0]['analysis GTI time factor'] == 0.
    assert source.to_dict() == state  # Full-observation amplitude stays nonzero.
    outside = SimpleNamespace(tstart_list=SimpleNamespace(unix=np.array([3.])),
                              tstop_list=SimpleNamespace(unix=np.array([5.])))
    with pytest.raises(ValueError, match='beyond'):
        api.make_temporal_responses(model, full_recipe, cut_history, outside, data, None, detector,
                                    full_history=full_history)
    row['time factor'] = .5
    with pytest.raises(ValueError, match='Temporal factor'):
        api.make_temporal_responses(model, {'components':[row]}, history, gti, data, None, detector)
    row.update({'time factor':0., 'applied factor':1e-30})
    gti.tstart_list.unix = np.array([10.]); gti.tstop_list.unix=np.array([14.])
    history.intervals_tstart.unix=np.array([10.,12.]); history.intervals_tstop.unix=np.array([12.,14.])
    overrides, _ = api.make_temporal_responses(model, {'components':[row]}, history, gti, data, None, detector)
    assert isinstance(overrides['transient'], api.ZeroPointSourceResponse)


def test_recipe_pairing(api, tmp_path):
    catalog = tmp_path/'catalog.yaml'
    recipe = {'schema':'dc4-component-lightcurve-response-v1', 'catalog':str(catalog),
              'selection':'NGC4151_GTI', 'time_basis':'elapsed',
              'components':[{'source':'a','component':'all','selection':'NGC4151_GTI'}]}
    path=tmp_path/'catalog.response.json'; path.write_text(json.dumps(recipe))
    assert api.load_recipe(path,catalog,SimpleNamespace(sources={'a':None})) == recipe
    with pytest.raises(ValueError, match='different catalog'):
        api.load_recipe(path,tmp_path/'other.yaml',SimpleNamespace(sources={'a':None}))
    recipe['selection']='full'
    recipe['components'][0]['selection']='full'
    path.write_text(json.dumps(recipe))
    assert api.load_recipe(path,catalog,SimpleNamespace(sources={'a':None}))['selection']=='full'
