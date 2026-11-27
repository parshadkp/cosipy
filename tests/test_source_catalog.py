"""Runtime FITS model and energy-integration checks; no notebook dependency."""
from pathlib import Path
import os
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import Galactic
from astromodels import Gaussian, Model, PointSource
from histpy import Axis, HealpixAxis
import numpy as np
import pytest
from cosipy.response import PointSourceResponse
from cosipy.response.integrals import get_integral_values
from cosipy.threeml.custom_functions import SpecFromFits


@pytest.fixture(scope='session')
def catalog_directory():
    default=Path.home()/'Library/CloudStorage/OneDrive-ClemsonUniversity/COSI/Radio_Quiet_AGN/DC4_Files/dc4_source_catalog'
    folder=Path(os.environ.get('COSIPY_DC4_CATALOG_DIR',str(default))).expanduser()
    if not (folder/'dc4_catalog.yaml').is_file():
        pytest.skip('External DC4 catalog unavailable; set COSIPY_DC4_CATALOG_DIR to test the delivered models')
    return folder


def write_spectral_template(path, energy, flux):
    table=fits.BinTableHDU.from_columns([
        fits.Column(name='ENERGY',format='D',unit='keV',array=energy),
        fits.Column(name='FLUX',format='D',unit='ph cm-2 s-1 keV-1',array=flux),
    ],name='SPECTRAL_MODEL')
    fits.HDUList([fits.PrimaryHDU(),table]).writeto(path)
    return path

@pytest.mark.parametrize('index', [-5., -1.75, 0., 1., 5.])
def test_cpl_integration_across_fitting_index_bounds(index):
    from astromodels import Cutoff_powerlaw
    spectrum = Cutoff_powerlaw(index=index, piv=200., xc=200., K=1.e-5)
    edges = np.geomspace(100., 2000., 11)
    exact = get_integral_values(spectrum, edges)
    assert np.all(np.isfinite(exact)) and np.all(exact >= 0.)
    assert exact == pytest.approx(get_integral_values(spectrum, edges, force_quad=True), rel=1.e-7)

def test_narrow_tabulated_line_and_additive_line(tmp_path):
    path = write_spectral_template(tmp_path/'line.fits',
                                   [100, 510, 511, 512, 1000], [0, 0, 3, 0, 0])
    shape = SpecFromFits(fits_file=str(path))
    # Neither response-bin edge sees the line; the integral must retain it.
    assert get_integral_values(shape, [100., 1000.]) == pytest.approx([3.])
    combined = shape + Gaussian(F=2., mu=478., sigma=.1)
    assert get_integral_values(combined, [100., 1000.]) == pytest.approx([5.])
    shape.K.value = 2.
    assert get_integral_values(shape, [100., 1000.]) == pytest.approx([6.])

def test_loglog_native_integration_and_zero_support(tmp_path):
    energy = np.array([100., 300., 1000.])
    shape = SpecFromFits(fits_file=str(write_spectral_template(
        tmp_path/'pl.fits', energy, 1.e4/energy**2)), interpolation='LOGLOG')
    assert np.asarray(shape(np.array([200.]))) == pytest.approx([.25])
    assert get_integral_values(shape, [10., 100., 1000., 2000.]) == pytest.approx([0., 90., 0.])

def test_fits_unit_metadata_and_quantity_evaluation(tmp_path):
    path = write_spectral_template(tmp_path/'spectrum.fits', [100, 200], [1, 2])
    shape = SpecFromFits(fits_file=str(path))
    Model(PointSource('test', l=0, b=0, spectral_shape=shape))
    actual = shape(np.array([.1, .2])*u.MeV)
    assert actual.to_value(u.cm**-2*u.s**-1*u.keV**-1) == pytest.approx([1, 2])
    with fits.open(path, checksum=True) as h:
        for hdu in h:
            assert hdu.verify_checksum() in (1,2)
            assert hdu.verify_datasum() in (1,2)

def test_fits_flux_folds_to_dimensionless_counts(tmp_path):
    shape = SpecFromFits(fits_file=str(write_spectral_template(
        tmp_path/'spectrum.fits', [100, 1000], [.01, .01])))
    Model(PointSource('test', l=0, b=0, spectral_shape=shape))
    axes = [Axis([100, 1000], label='Ei', unit=u.keV),
            Axis([100, 1000], label='Em', unit=u.keV),
            Axis([0, 180], label='Phi', unit=u.deg),
            HealpixAxis(nside=1, coordsys=Galactic(), label='PsiChi')]
    psr = PointSourceResponse(axes, contents=np.full((1, 1, 1, 12), 100./12),
                              unit=u.cm**2*u.s)
    result = psr.get_expectation(shape)
    assert result.unit == u.dimensionless_unscaled
    assert np.sum(result.contents).to_value(u.dimensionless_unscaled) == pytest.approx(900.)


def test_loglog_change_revalidates_cached_spectrum(tmp_path):
    path=write_spectral_template(tmp_path/'zero.fits',[100,200,300],[1,0,1])
    shape=SpecFromFits(fits_file=str(path))
    assert shape(np.array([200.]))[0]==0
    shape.interpolation.value='LOGLOG'
    with pytest.raises(ValueError,match='strictly positive'):
        shape(np.array([200.]))


def test_spectral_deliverables_have_only_minimal_table_headers(catalog_directory):
    folder=catalog_directory
    forbidden={'INTERP','EXTRAP','DATE','OBJECT','TSTART','TSTOP','SRCSHA','LCFILE','LCSHA','CONVVER','HDUCLASS'}
    for path in folder.glob('*.fits'):
        with fits.open(path) as hdul:
            if 'SPECTRAL_MODEL' not in hdul:
                continue
            assert hdul[1].columns.names==['ENERGY','FLUX']
            assert not forbidden.intersection(hdul[1].header)
