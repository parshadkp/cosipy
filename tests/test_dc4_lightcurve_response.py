"""Portable tests of temporal profiles, without DC4 data or responses."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def profile_class():
    from cosipy.response.temporal_profile import LightcurveProfile
    return LightcurveProfile


def test_parallel_keeps_inactive_chunks_off(profile_class):
    times = np.arange(9.)
    rates = np.array([1., 1., 1., 0., 0., 0., 0., 0., 0.])
    parallel = profile_class(times, rates, mode="parallel_megalib", bins=2)
    global_profile = profile_class(times, rates)
    assert parallel.active_chunks == 1
    np.testing.assert_allclose(parallel.integral([0., 4.], [4., 8.]), [4., 0.])
    np.testing.assert_allclose(global_profile.integral([0.], [8.]), [8.])


@pytest.mark.parametrize("samples,bins", [(9, 2), (10, 3), (12, 3), (11, 3)])
def test_parallel_chunk_coverage(profile_class, samples, bins):
    profile = profile_class(np.arange(float(samples)), np.ones(samples),
                            mode="parallel_megalib", bins=bins)
    np.testing.assert_allclose(profile.integral([0.], [samples-1.]), [samples-1.])


def test_subinterval_burst_not_missed_and_mean_not_applied_twice(profile_class):
    profile = profile_class([7., 7.1, 7.2], [0., 2., 0.])
    starts, stops = np.array([0., 15.]), np.array([15., 30.])
    mean = profile.integral([0.], [30.])[0]/30.
    weighted = profile.response_weights(starts, stops, mean)
    # Folding with scaled amplitude and normalized temporal response is
    # identical to folding the injected amplitude with the physical profile.
    np.testing.assert_allclose(mean*weighted*(stops-starts), profile.integral(starts, stops))
    assert weighted[0] > 0 and weighted[1] == 0


def test_time_response_recovers_exposure_correlation(profile_class):
    profile = profile_class([0., 1.], [1., 1.], mode="raw")
    starts, stops = np.array([0., 1.]), np.array([1., 2.])
    area = np.array([10., 1.])
    mean = .5
    ordinary_scaled = mean*np.sum(area*(stops-starts))
    weighted = mean*np.sum(area*(stops-starts)*profile.response_weights(starts, stops, mean))
    assert ordinary_scaled == 5.5
    assert weighted == 10.


def test_periodic_precision_and_zero_overlap(profile_class):
    periodic = profile_class([0., .033], [3., 3.], repeating=True)
    np.testing.assert_allclose(periodic.integral([1837507000.], [1837507015.]), [15.], atol=1e-6)
    finite = profile_class([0., 1.], [1., 1.])
    np.testing.assert_array_equal(finite.response_weights([10.], [20.], 1e-30), [0.])


def test_raw_mode_preserves_duty_cycle(profile_class):
    p = profile_class([0., 1., 2., 10.], [1., 1., 0., 0.], mode="raw")
    np.testing.assert_allclose(p.integral([0.], [10.]), [1.5])


def test_invalid_profile_rejected(profile_class):
    with pytest.raises(ValueError):
        profile_class([0., 1.], [1., -1.])
    with pytest.raises(ValueError):
        profile_class([0., 1.], [1., 1.], repeating=True, mode="parallel_megalib")
