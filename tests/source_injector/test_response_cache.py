from types import SimpleNamespace
import os
import shutil

import astropy.units as u
from astromodels import Cutoff_powerlaw, Model, PointSource
import numpy as np
import pytest

from cosipy import SourceInjector, SpacecraftHistory, test_data
from cosipy.response import FullDetectorResponse
from scoords import Attitude


@pytest.fixture
def injection(monkeypatch, request):
    history = SpacecraftHistory.open(test_data.path / "20280301_2s.fits")
    spectrum = Cutoff_powerlaw()
    spectrum.K.value = 17
    spectrum.index.value = -2.2
    spectrum.xc.value = 400
    model = Model(PointSource("target", l=50, b=-45, spectral_shape=spectrum))
    response_name = getattr(request, "param", "test_full_detector_response.h5")
    injector = SourceInjector(test_data.path / response_name)
    builds = []
    original = FullDetectorResponse.get_point_source_response

    def counted(response, *args, **kwargs):
        builds.append(kwargs["coord"])
        return original(response, *args, **kwargs)

    monkeypatch.setattr(FullDetectorResponse, "get_point_source_response", counted)
    return SimpleNamespace(history=history, spectrum=spectrum, model=model,
                           injector=injector, builds=builds)


def inject(state, fluctuate=False, injector=None):
    return (injector or state.injector).inject_model(
        state.model, orientation=state.history, fluctuate=fluctuate,
    )


@pytest.mark.parametrize("injection", [
    "test_full_detector_response.h5", "test_polarization_response.h5",
], indirect=True)
def test_cache_reuses_response_but_refolds_spectrum(injection):
    state = injection
    initial = inject(state)
    state.spectrum.K.value *= 2
    doubled = inject(state)
    np.testing.assert_allclose(doubled.contents, 2 * initial.contents, rtol=1e-13)
    state.spectrum.xc.value = 800
    changed = inject(state)
    assert len(state.builds) == 1
    assert not np.array_equal(changed.contents, doubled.contents)

    fresh = inject(state, injector=SourceInjector(state.injector.response_path))
    np.testing.assert_array_equal(changed.contents, fresh.contents)


def test_cache_invalidates_for_inplace_exposure_change(injection):
    state = injection
    initial = inject(state)
    state.history.livetime[:] *= 0.5
    changed = inject(state)
    assert len(state.builds) == 2
    np.testing.assert_allclose(changed.contents, 0.5 * initial.contents, rtol=1e-13)


def test_cache_invalidates_for_attitude_change(injection, monkeypatch):
    state = injection
    inject(state)
    coord = state.model.point_sources["target"].position.sky_coord
    with FullDetectorResponse.open(state.injector.response_path) as response:
        mapping = state.history.get_scatt_map(response.nside * 2, target_coord=coord)
    mapping.attitudes = Attitude.from_rotvec(
        mapping.attitudes.as_rotvec() + np.array([0.2, 0, 0]) * u.rad,
        frame=mapping.attitudes.frame,
    )
    monkeypatch.setattr(state.history, "get_scatt_map", lambda *args, **kwargs: mapping)
    changed = inject(state)
    assert len(state.builds) == 2
    fresh = inject(state, injector=SourceInjector(state.injector.response_path))
    np.testing.assert_array_equal(changed.contents, fresh.contents)


def test_cache_is_bounded_to_last_position(injection):
    state = injection
    initial = inject(state)
    position = state.model.point_sources["target"].position
    position.l.value = 80
    inject(state)
    position.l.value = 50
    returned = inject(state)
    assert len(state.builds) == 3
    np.testing.assert_array_equal(initial.contents, returned.contents)


def test_cache_distinguishes_output_coordinate_frames(injection):
    state = injection
    inject(state)
    coord = state.model.point_sources["target"].position.sky_coord.icrs
    changed = state.injector.inject_point_source(
        state.spectrum, coord, orientation=state.history,
    )
    assert len(state.builds) == 2
    fresh = SourceInjector(state.injector.response_path).inject_point_source(
        state.spectrum, coord, orientation=state.history,
    )
    assert changed.axes == fresh.axes
    np.testing.assert_array_equal(changed.contents, fresh.contents)


def test_cache_invalidates_for_earth_occultation_setting(injection):
    state = injection
    inject(state)
    state.injector.inject_model(
        state.model, orientation=state.history, fluctuate=False, earth_occ=False,
    )
    assert len(state.builds) == 2


def test_cache_invalidates_when_response_file_changes(injection, tmp_path):
    state = injection
    path = tmp_path / "response.h5"
    shutil.copyfile(state.injector.response_path, path)
    state.injector.response_path = path
    initial = inject(state)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    changed = inject(state)
    assert len(state.builds) == 2
    np.testing.assert_array_equal(initial.contents, changed.contents)


def test_cached_response_never_caches_poisson_counts(injection):
    state = injection
    state.spectrum.K.max_value = 1e8
    state.spectrum.K.value = 1e5
    previous = np.random.get_state()
    try:
        np.random.seed(42)
        first = inject(state, fluctuate=True)
        second = inject(state, fluctuate=True)
        assert not np.array_equal(first.contents, second.contents)
        np.random.seed(42)
        repeated = inject(state, fluctuate=True)
        np.testing.assert_array_equal(first.contents, repeated.contents)
        deterministic = inject(state)
        assert len(state.builds) == 1
        fresh = inject(state, injector=SourceInjector(state.injector.response_path))
        np.testing.assert_array_equal(deterministic.contents, fresh.contents)
    finally:
        np.random.set_state(previous)
