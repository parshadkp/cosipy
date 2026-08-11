import astropy.units as u
import numpy as np

from cosipy.image_deconvolution import AllSkyImageModel
from cosipy.image_deconvolution.spectral_deconvolution import (
    differential_flux_from_model,
)


def test_differential_flux_from_model_integrates_solid_angle():
    model = AllSkyImageModel(
        nside=1,
        energy_edges=np.array([100.0, 200.0, 400.0]) * u.keV,
    )
    model[:, 0] = 2.0 * model.unit
    model[:, 1] = 3.0 * model.unit

    flux = differential_flux_from_model(model)

    expected = np.array([2.0 * 4 * np.pi / 100.0, 3.0 * 4 * np.pi / 200.0])
    assert np.allclose(flux.value, expected)
    assert flux.unit == 1 / (u.cm**2 * u.s * u.keV)
