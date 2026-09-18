"""Public temporal APIs work independently of any DC4 notebook or recipe."""
import gzip
import hashlib
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from histpy import Axis, Axes, HealpixAxis

from cosipy.response import LightcurveProfile, make_weighted_point_source_response
from cosipy.response import threeml_temporal_response as temporal


@pytest.mark.parametrize('compressed', [False, True])
def test_lightcurve_file_reader(tmp_path, compressed):
    path = tmp_path / ('profile.dat.gz' if compressed else 'profile.dat')
    text = '# example\nIP LINLIN # interpolation\nDP 1 2\nDP 3 2\nEN\n'
    if compressed:
        with gzip.open(path, 'wt') as stream:
            stream.write(text)
    else:
        path.write_text(text)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    profile = LightcurveProfile.from_dat(path, mode='raw', expected_sha256=digest)
    np.testing.assert_allclose(profile.integral([0.], [10.]), [4.])
    with pytest.raises(ValueError, match='changed'):
        LightcurveProfile.from_dat(path, expected_sha256='incorrect')


@pytest.mark.parametrize('text', ['IP LOGLOG\n', 'IP\n', 'DP 1\n', 'DP 1 -1\nDP 2 1\n'])
def test_malformed_or_unsupported_lightcurve(tmp_path, text):
    path = tmp_path / 'profile.dat'
    path.write_text(text)
    with pytest.raises(ValueError):
        LightcurveProfile.from_dat(path)


@pytest.fixture
def inputs():
    axes = Axes([Axis([100., 200.], label='Em', unit=u.keV),
                 Axis([0., 180.], label='Phi', unit=u.deg),
                 HealpixAxis(nside=1, scheme='ring', coordsys='galactic', label='PsiChi')])
    history = SimpleNamespace(obstime=None, attitude=None, location=None,
                              livetime=np.array([2., 3.])*u.s, cache_earth_occ=True)
    detector = SimpleNamespace(axes=Axes([Axis([100., 200.], label='Ei', unit=u.keV)]))
    return history, SimpleNamespace(axes=axes), detector


def test_weighted_history_is_independent(inputs, monkeypatch):
    history, data, detector = inputs
    monkeypatch.setattr(temporal, 'SpacecraftHistory',
                        lambda *args, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(temporal, 'BinnedThreeMLPointSourceResponse',
                        lambda **kwargs: SimpleNamespace(**kwargs))
    weights = np.array([.5, 2.])
    response = make_weighted_point_source_response(
        history, weights, data, None, detector, exposure_multiplier=8.)
    np.testing.assert_allclose(response.sc_history.livetime.to_value(u.s), [8., 48.])
    np.testing.assert_allclose(history.livetime.to_value(u.s), [2., 3.])
    np.testing.assert_allclose(weights, [.5, 2.])
    assert response.sc_history.cache_earth_occ is True


def test_zero_weight_avoids_response_construction(inputs, monkeypatch):
    history, data, detector = inputs
    def unexpected(*args, **kwargs):
        raise AssertionError('Zero exposure must not construct a spacecraft response')
    monkeypatch.setattr(temporal, 'SpacecraftHistory', unexpected)
    response = make_weighted_point_source_response(history, [0., 0.], data, None, detector)
    assert isinstance(response, temporal.ZeroPointSourceResponse)
    assert response.expectation().contents.sum() == 0.


@pytest.mark.parametrize('weights', [[1.], [-1., 1.], [np.nan, 1.], [np.inf, 1.]])
def test_invalid_interval_weights(inputs, weights):
    history, data, detector = inputs
    with pytest.raises(ValueError, match='weight'):
        make_weighted_point_source_response(history, weights, data, None, detector)


@pytest.mark.parametrize('multiplier', [0., -1., np.nan, np.inf])
def test_invalid_exposure(inputs, multiplier):
    history, data, detector = inputs
    with pytest.raises(ValueError, match='Exposure'):
        make_weighted_point_source_response(history, [1., 1.], data, None, detector,
                                            exposure_multiplier=multiplier)
