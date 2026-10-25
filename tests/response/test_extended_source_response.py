from cosipy import test_data
from cosipy.response import ExtendedSourceResponse
from cosipy.image_deconvolution import AllSkyImageModel

from astromodels import Gaussian, Gaussian_on_sphere, ExtendedSource, Model, load_model
import astropy.units as u
import numpy as np
import importlib
from types import SimpleNamespace
from astropy.coordinates import Galactic

from cosipy.threeml.custom_functions import GalpropHealpixModel

extended_response_path = test_data.path/"test_precomputed_response.h5"

def test_open():
    resp = ExtendedSourceResponse.open(extended_response_path)

def test_get_expectation():

    resp = ExtendedSourceResponse.open(extended_response_path)

    nside = resp.axes['NuLambda'].nside
    energy_edges = resp.axes['Ei'].edges

    allsky_imagemodel = AllSkyImageModel(nside = nside,
                                         energy_edges = energy_edges,
                                         label_image = 'NuLambda')

    hist = resp.get_expectation(allsky_imagemodel)

    assert isinstance(hist[:], u.quantity.Quantity) == True

def test_get_expectation_from_astromodel():

    resp = ExtendedSourceResponse.open(extended_response_path)

    # Define spectrum:
    F = 4e-2 / u.cm / u.cm / u.s
    mu = 511*u.keV
    sigma = 0.85*u.keV
    spectrum = Gaussian()
    spectrum.F.value = F.value
    spectrum.F.unit = F.unit
    spectrum.mu.value = mu.value
    spectrum.mu.unit = mu.unit
    spectrum.sigma.value = sigma.value
    spectrum.sigma.unit = sigma.unit

    # Define morphology:
    morphology = Gaussian_on_sphere(lon0 = 0, lat0 = 0, sigma = 5)

    # Define source:
    extended_model = ExtendedSource('gaussian', spectral_shape=spectrum, spatial_shape=morphology)

    # Calculate the expectation
    hist = resp.get_expectation_from_astromodel(extended_model)

    assert isinstance(hist[:], u.quantity.Quantity) == True

def test_get_expectation_from_astromodel_3d():

    resp = ExtendedSourceResponse.open(extended_response_path)

    # Upload source model:
    extended_model = load_model(test_data.path/'galprop_model.yaml')
    extended_model.galprop_source.spatial_shape._fitsfile = test_data.path/'ics_isotropic_healpix_54_0780000f.gz'
    extended_model.galprop_source.spatial_shape.set_version(54)

    # Calculate the expectation
    hist = resp.get_expectation_from_astromodel(extended_model.galprop_source)

    assert isinstance(hist[:], u.quantity.Quantity) == True


def test_spatial_response_factorization():
    resp = ExtendedSourceResponse.open(extended_response_path)

    spectrum = Gaussian()
    spectrum.F.value = 4e-2
    spectrum.F.unit = 1 / u.cm**2 / u.s
    spectrum.mu.value = 511.0
    spectrum.mu.unit = u.keV
    spectrum.sigma.value = 0.85
    spectrum.sigma.unit = u.keV

    morphology = Gaussian_on_sphere(lon0=0, lat0=0, sigma=5)
    source = ExtendedSource(
        "gaussian_factorized",
        spectral_shape=spectrum,
        spatial_shape=morphology,
    )

    full = resp.get_expectation_from_astromodel(source)
    spatial_response = resp.get_spatial_response_from_astromodel(source)
    factorized = resp.get_expectation_from_spatial_response(
        source,
        spatial_response,
    )

    assert full.axes == factorized.axes
    assert full.unit == factorized.unit
    assert np.allclose(
        full.contents,
        factorized.contents,
        rtol=1e-10,
        atol=1e-12,
    )



def test_galprop_unit_expectation(monkeypatch):
    # Keep this unit test lightweight by mocking the GALPROP integration and
    # Histogram construction. The existing 3D test above exercises the real
    # response/data path.
    response_module = importlib.import_module(
        "cosipy.response.ExtendedSourceResponse"
    )

    galprop = GalpropHealpixModel()
    galprop.K.value = 2.3
    source = SimpleNamespace(spatial_shape=galprop)

    unit_flux = np.ones((2, 3))
    flux_map = object()
    expected = object()

    def fake_integrated_model(source, image_axis, energy_axis):
        galprop.intg_flux = unit_flux

    monkeypatch.setattr(
        response_module,
        "get_integrated_extended_model_3d",
        fake_integrated_model,
    )
    monkeypatch.setattr(
        response_module,
        "Histogram",
        lambda *args, **kwargs: flux_map,
    )

    fake_response = SimpleNamespace(
        axes=(object(), object()),
        _exp_unit=object(),
        get_expectation=lambda model: expected,
    )

    result = ExtendedSourceResponse.get_galprop_unit_expectation(
        fake_response,
        source,
    )

    assert result is expected
    assert galprop.K.value == 2.3


def test_galprop_native_energy_knot_integration(monkeypatch):
    """A narrow in-bin template feature must not vanish at response edges."""
    response_module = importlib.import_module("cosipy.response.functions_3d")
    custom_module = importlib.import_module("cosipy.threeml.custom_functions")

    class FakeGalprop:
        def __init__(self):
            self.K = SimpleNamespace(value=2.0)
            self._file_loaded = True
            self.energy = np.array([0.1, 0.4, 0.5, 0.6, 1.0]) * u.MeV
            self.calls = 0

        def evaluate(self, longitude, latitude, energies, norm):
            self.calls += 1
            spectrum = np.interp(
                energies.to_value(u.MeV),
                self.energy.to_value(u.MeV),
                [0.0, 0.0, 10.0, 0.0, 0.0],
            )
            return spectrum[None, :] / (u.MeV * u.s * u.cm**2 * u.sr)

    monkeypatch.setattr(custom_module, "GalpropHealpixModel", FakeGalprop)
    monkeypatch.setattr(
        response_module, "Histogram",
        lambda axes, contents, **kwargs: SimpleNamespace(contents=contents),
    )
    galprop = FakeGalprop()
    source = SimpleNamespace(
        spatial_shape=galprop,
        spectrum=SimpleNamespace(
            main=SimpleNamespace(Constant=SimpleNamespace(k=SimpleNamespace(free=True)))
        ),
    )
    image_axis = SimpleNamespace(
        coordsys=Galactic(), npix=1,
        pix2ang=lambda pixels, lonlat: (np.zeros(1) * u.deg, np.zeros(1) * u.deg),
    )
    energy_axis = SimpleNamespace(edges=np.array([0.1, 1.0]) * u.MeV, nbins=1)

    first = response_module.get_integrated_extended_model_3d(
        source, image_axis, energy_axis
    )
    # The triangle between 0.4 and 0.6 MeV has unit integrated intensity.
    np.testing.assert_allclose(first.contents, [[2.0]])
    assert galprop.calls == 1

    galprop.K.value = 3.0
    second = response_module.get_integrated_extended_model_3d(
        source, image_axis, energy_axis
    )
    np.testing.assert_allclose(second.contents, [[3.0]])
    assert galprop.calls == 1
