"""Portable tests of notebook time integration, without DC4 data or responses."""
import ast
import gzip
import json
import importlib.util
from array import array
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest


@pytest.fixture(scope="module")
def helpers():
    path = Path(__file__).resolve().parents[1] / (
        "docs/tutorials/spectral_fits/continuum_fit/AGN/"
        "Source_catalog_generator_all_DC4_lightcurve.ipynb"
    )
    notebook = json.loads(path.read_text())
    wanted = {"read_lightcurve", "normalized_lc_integral", "selection_intervals"}
    functions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(line for line in cell["source"] if not line.startswith("%"))
        functions.extend(node for node in ast.parse(text).body
                         if isinstance(node, ast.FunctionDef) and node.name in wanted)
    assert {node.name for node in functions} == wanted
    namespace = {"np": np, "u": u, "Path": Path, "gzip": gzip, "array": array}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_full_burst_and_no_overlap(helpers):
    integrate = helpers["normalized_lc_integral"]
    times = np.array([10., 20.])
    rates = np.array([3., 3.])  # Raw scale must cancel under MEGAlib normalization.
    np.testing.assert_allclose(integrate(times, rates, [0., 30.], [100., 40.]), [10., 0.])


def test_partial_triangular_curve(helpers):
    integrate = helpers["normalized_lc_integral"]
    times, rates = np.array([0., 5., 10.]), np.array([0., 2., 0.])
    np.testing.assert_allclose(integrate(times, rates, [0., 8.], [2., 10.]), [.8, .8])


def test_gti_denominator_excludes_gaps(helpers):
    gti = SimpleNamespace(tstart_list=SimpleNamespace(unix=np.array([0., 15.])),
                          tstop_list=SimpleNamespace(unix=np.array([12., 18.])))
    starts, stops, weights, duration = helpers["selection_intervals"](None, gti)
    assert duration == 15.
    integral = helpers["normalized_lc_integral"](np.array([10., 20.]), np.ones(2), starts, stops)
    assert np.isclose(np.sum(integral*weights)/duration, 5./15.)


def test_periodic_curve_at_unix_times(helpers):
    actual = helpers["normalized_lc_integral"](
        np.array([0., .033]), np.ones(2), [1837507000.], [1837507015.], repeating=True)
    np.testing.assert_allclose(actual, [15.], atol=1e-6)


def test_livetime_excludes_zero_livetime_gaps(helpers):
    history = SimpleNamespace(intervals_tstart=SimpleNamespace(unix=np.array([0., 10., 90.])),
                              intervals_tstop=SimpleNamespace(unix=np.array([10., 90., 100.])),
                              livetime=np.array([5., 0., 10.])*u.s)
    starts, stops, weights, duration = helpers["selection_intervals"](history, basis="livetime")
    np.testing.assert_allclose(starts, [0., 90.])
    np.testing.assert_allclose(weights, [.5, 1.])
    assert duration == 15.


@pytest.mark.parametrize("compressed", [False, True])
def test_lightcurve_reader(helpers, tmp_path, compressed):
    text = "# comment\nIP LINLIN\nDP 0 1\nDP 5 2\nEN\n"
    path = tmp_path / ("lc.dat.gz" if compressed else "lc.dat")
    if compressed:
        with gzip.open(path, "wt") as stream:
            stream.write(text)
    else:
        path.write_text(text)
    times, rates = helpers["read_lightcurve"](path)
    np.testing.assert_array_equal(times, [0., 5.])
    np.testing.assert_array_equal(rates, [1., 2.])


def test_bad_lightcurve_rejected(helpers, tmp_path):
    path = tmp_path / "bad.dat"
    path.write_text("DP 0 1\nDP 1 -2\n")
    with pytest.raises(ValueError, match="Invalid"):
        helpers["read_lightcurve"](path)


@pytest.fixture(scope="module")
def profile_class():
    path = Path(__file__).resolve().parents[1] / (
        "docs/tutorials/spectral_fits/continuum_fit/AGN/dc4_lightcurve_response.py"
    )
    spec = importlib.util.spec_from_file_location("dc4_lc_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LightcurveProfile


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
