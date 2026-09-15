###
### Test integration code for assorted Astromodels spectral functions
###

import numpy as np

from astromodels import (
    Constant,
    Line,
    Quadratic,
    Cubic,
    Quartic,
    Band,
    Band_grbm,
    Powerlaw,
    Cutoff_powerlaw,
    Gaussian,
    DiracDelta,
    StepFunction,
    StepFunctionUpper,
    GenericFunction,
)

from cosipy.threeml.custom_functions import Band_Eflux, SpecFromDat

from cosipy.response.integrals import get_integral_values
import pytest


@pytest.mark.parametrize("normalization", [1.0e-30, 1.0, 1036.4060853665721])
def test_tabulated_integral_support_and_scaling(tmp_path, normalization):
    table = tmp_path / "spectrum.dat"
    table.write_text("IP LINLIN\nDP 1 2\nDP 2 4\nDP 4 0\nEN\n")
    spectrum = SpecFromDat(K=normalization, dat=table)
    edges = np.array([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    # SpecFromDat normalizes by the left-rectangle integral (10 here).
    reference = np.array([0.0, 0.125, 0.175, 0.3, 0.1, 0.0])
    actual = get_integral_values(spectrum, edges)
    np.testing.assert_allclose(actual / normalization, reference, rtol=1e-14)
    np.testing.assert_allclose(
        get_integral_values(spectrum + spectrum, edges) / normalization,
        2 * reference, rtol=1e-14,
    )


@pytest.mark.parametrize("index", [-3.0, -2.53, -1.0, -0.9, 0.0, 0.5, 1.0, 2.0, 3.0])
def test_cutoff_powerlaw_integer_and_fractional_indices(index):
    # MGF070222 has index=+1 and xc=1287/(2+1)=429 keV.
    spectrum = Cutoff_powerlaw(index=index, piv=100.0, xc=429.0, K=0.01)
    edges = np.geomspace(100.0, 10000.0, 11)
    actual = get_integral_values(spectrum, edges)
    reference = get_integral_values(spectrum, edges, force_quad=True)
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0)
    np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-15)


def test_composite_integral_preserves_expression():
    edges = np.array([100.0, 200.0, 400.0])
    spectrum = Powerlaw(K=0.01, index=-2.0, piv=100.0)
    one = get_integral_values(spectrum, edges)
    np.testing.assert_allclose(get_integral_values(spectrum + spectrum, edges), 2 * one)
    np.testing.assert_allclose(
        get_integral_values((2 * spectrum) + 0.003, edges),
        2 * one + 0.003 * np.diff(edges),
    )

def test_integrate():

    x = np.geomspace(10, 10000, num=11)

    ## Constant
    const = Constant(k=1e-1)
    v     = get_integral_values(const, x)
    v_q   = get_integral_values(const, x, force_quad=True)
    assert np.allclose(v, v_q)

    ## Line
    line = Line(a=1e-1, b=4e-5)
    v    = get_integral_values(line, x)
    v_q  = get_integral_values(line, x, force_quad=True)
    assert np.allclose(v, v_q)

    ## Quadratic
    quad = Quadratic(a=1e-1, b=4e-5, c=1e-9)
    v    = get_integral_values(quad, x)
    v_q  = get_integral_values(quad, x, force_quad=True)
    assert np.allclose(v, v_q)

    ## Cubic
    cubic = Cubic(a=1e-1, b=4e-5, c=1e-9, d=4e-13)
    v    = get_integral_values(cubic, x)
    v_q  = get_integral_values(cubic, x, force_quad=True)
    assert np.allclose(v, v_q)

    ## Quartic
    quartic = Quartic(a=1e-1, b=4e-5, c=1e-9, d=4e-13, e=1e-17)
    v    = get_integral_values(quartic, x)
    v_q  = get_integral_values(quartic, x, force_quad=True)
    assert np.allclose(v, v_q)

    ## Test generic function integration (falls back to adaptive
    ## quadrature)
    gen = GenericFunction()
    gen.set_function(lambda x: 1e-1) # like Constant()
    v = get_integral_values(gen, x)
    v0 = get_integral_values(const, x)
    assert np.allclose(v, v0)
    ## Powerlaw
    pl = Powerlaw(index=-1, piv=100., K=400.)
    v   = get_integral_values(pl, x)
    v_q = get_integral_values(pl, x, force_quad=True)
    assert np.allclose(v, v_q)

    ## Cutoff_powerlaw

    # test all three possible branches based on value of index
    # (> -1, negative integer, non-integer < -1)
    pl = Cutoff_powerlaw(index=-2.53, piv=100., xc = 400., K=10.)
    v   = get_integral_values(pl, x)
    v_q = get_integral_values(pl, x, force_quad=True)
    assert np.allclose(v, v_q)

    pl = Cutoff_powerlaw(index=-2.0, piv=100., xc = 400., K=10.)
    v   = get_integral_values(pl, x)
    v_q = get_integral_values(pl, x, force_quad=True)
    assert np.allclose(v, v_q)

    pl = Cutoff_powerlaw(index=-0.9, piv=100., xc = 400., K=10.)
    v   = get_integral_values(pl, x)
    v_q = get_integral_values(pl, x, force_quad=True)
    assert np.allclose(v, v_q)

    ## Band

    # test all three possible branches based on value of cutoff
    # (less than all of x, greater than all of x, in the middle)
    bs = Band(alpha=-0.5, beta=-2.53, piv=100, xp=500, K=10)
    v   = get_integral_values(bs, x)
    v_q = get_integral_values(bs, x, force_quad=True)
    assert np.allclose(v, v_q)

    bs = Band(alpha=-0.5, beta=-2.53, piv=100, xp=10, K=10)
    bs.xp.min_value = 5.
    bs.xp.value = 5.
    v   = get_integral_values(bs, x)
    v_q = get_integral_values(bs, x, force_quad=True)
    assert np.allclose(v, v_q)

    bs = Band(alpha=-0.5, beta=-2.53, piv=100, xp=50000, K=10)
    v   = get_integral_values(bs, x)
    v_q = get_integral_values(bs, x, force_quad=True)
    assert np.allclose(v, v_q)

    # test cutoff falling right on bin boundary, as opposed
    # to between two boundaries
    x_simple = np.array([10, 100, 1000, 10000])
    bs = Band(alpha=-1, beta=-2, piv=100, xp=1000, K=10)
    v   = get_integral_values(bs, x_simple)
    v_q = get_integral_values(bs, x_simple, force_quad=True)
    assert np.allclose(v, v_q)

    ## Band_grbm (NOT THE SAME AS Band!)
    bs = Band_grbm(alpha=-0.5, beta=-2.53, piv=100, xc=500, K=10)
    v   = get_integral_values(bs, x)
    v_q = get_integral_values(bs, x, force_quad=True)
    assert np.allclose(v, v_q)

    ## Band_Eflux
    bs = Band_Eflux(a=100, b=600, alpha=-0.5, beta=-2.53, E0=500, K=10)
    v   = get_integral_values(bs, x)
    v_q = get_integral_values(bs, x, force_quad=True)
    assert np.allclose(v, v_q)

    ## Gaussian

    x = np.linspace(-10,10,num=21)
    g = Gaussian(mu=1.1, sigma=2.3, F=2.)
    v   = get_integral_values(g, x)
    v_q = get_integral_values(g, x, force_quad=True)
    assert np.allclose(v, v_q)

    ################################################################

    # These need to be compared to an explicit result, as they can't
    # be numerically integrated accurately

    x = np.arange(-10, 11, dtype=np.float64)

    ## StepFunction
    step = StepFunction(upper_bound=5, lower_bound=-5, value=3)
    v = get_integral_values(step, x)
    v0 = [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0]
    assert np.allclose(v, v0)

    step = StepFunction(upper_bound=5.5, lower_bound=-5.5, value=3)
    v = get_integral_values(step, x)
    v0 = [0, 0, 0, 0, 1.5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1.5, 0, 0, 0, 0]
    assert np.allclose(v, v0)

    v = get_integral_values(step, x, force_quad=True) # ignored for Step
    assert np.allclose(v, v0)

    ## StepFunctionUpper
    step = StepFunctionUpper(upper_bound=5, lower_bound=-5, value=3)
    v = get_integral_values(step, x)
    v0 = [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0]
    assert np.allclose(v, v0)

    step = StepFunction(upper_bound=5.5, lower_bound=-5.5, value=3)
    v = get_integral_values(step, x)
    v0 = [0, 0, 0, 0, 1.5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1.5, 0, 0, 0, 0]
    assert np.allclose(v, v0)

    v = get_integral_values(step, x, force_quad=True) # ignored for StepUpper
    assert np.allclose(v, v0)

    ## DiracDelta
    delta = DiracDelta(value=3, zero_point=1.5)
    v = get_integral_values(delta, x)
    v0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.allclose(v, v0)

    delta = DiracDelta(value=3, zero_point=1)
    v = get_integral_values(delta, x)
    v0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.allclose(v, v0)

    v = get_integral_values(delta, x, force_quad=True) # ignored for Delta
    assert np.allclose(v, v0)


def test_integrate_additive_composite_with_narrow_gaussians():
    """Narrow lines inside broad bins must not be missed by quadrature."""

    edges = np.array([100.0, 1000.0, 1200.0, 1400.0, 10000.0])
    line_low = Gaussian(mu=1173.3, sigma=0.44, F=2.7e-6)
    line_high = Gaussian(mu=1332.6, sigma=0.50, F=2.7e-6)

    values = get_integral_values(line_low + line_high, edges)

    assert np.allclose(values, [0.0, 2.7e-6, 2.7e-6, 0.0])
    assert np.isclose(values.sum(), 5.4e-6)
