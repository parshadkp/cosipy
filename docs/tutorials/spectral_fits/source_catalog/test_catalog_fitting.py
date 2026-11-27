"""Fitting notebook checks, maintained only on SourceCatTest."""
from pathlib import Path
import ast
import json
import os
import shutil
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import Galactic
from astromodels import ExtendedSource, Gaussian, Model, PointSource, SpectralComponent
from histpy import Axis, HealpixAxis
import numpy as np
import pytest
from cosipy.response import PointSourceResponse
from cosipy.response.functions_3d import get_integrated_extended_model_3d
from cosipy.response.integrals import get_integral_values
from cosipy.threeml.custom_functions import GalpropHealpixModel, SpecFromFits

NOTEBOOK=Path(__file__).with_name("DC4_Static_Source_Catalog_Spectral_Fit.ipynb")



@pytest.fixture(scope="session")
def catalog_directory():
    default=Path.home()/'Library/CloudStorage/OneDrive-ClemsonUniversity/COSI/Radio_Quiet_AGN/DC4_Files/dc4_source_catalog'
    folder=Path(os.environ.get('COSIPY_DC4_CATALOG_DIR',str(default))).expanduser()
    if not (folder/'dc4_catalog.yaml').is_file():
        pytest.skip('External DC4 catalog unavailable; set COSIPY_DC4_CATALOG_DIR to test the delivered models')
    return folder

def notebook_definitions():
    """Exercise the actual readable notebook functions without fitting any data."""
    namespace={}
    notebook=json.loads(NOTEBOOK.read_text())
    for cell in notebook['cells']:
        if cell['cell_type']!='code':
            continue
        tree=ast.parse(''.join(cell['source']))
        nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(NOTEBOOK),'exec'),namespace)
    return namespace

@pytest.fixture(autouse=True)
def analysis_functions(monkeypatch):
    definitions=notebook_definitions()
    for name in ('load_source_catalog','free_source_normalization'):
        monkeypatch.setitem(globals(),name,definitions[name])

def write_spectral_template(path, energy, flux):
    table=fits.BinTableHDU.from_columns([
        fits.Column(name='ENERGY',format='D',unit='keV',array=energy),
        fits.Column(name='FLUX',format='D',unit='ph cm-2 s-1 keV-1',array=flux),
    ],name='SPECTRAL_MODEL')
    fits.HDUList([fits.PrimaryHDU(),table]).writeto(path)
    return path

def test_catalog_portable_after_original_directory_removed(tmp_path, monkeypatch):
    original = tmp_path/'original'
    original.mkdir()
    path = write_spectral_template(original/'spec.fits', [100, 200], [1, 2])
    model = Model(PointSource('test', l=0, b=0,
                              spectral_shape=SpecFromFits(fits_file=str(path))))
    model.save(original/'catalog.yaml', overwrite=True)
    yaml_path = original/'catalog.yaml'
    yaml_path.write_text(yaml_path.read_text().replace(str(path), path.name))
    copied = tmp_path/'copied'
    shutil.copytree(original, copied)
    shutil.rmtree(original)
    monkeypatch.chdir(tmp_path)
    model = load_source_catalog(copied/'catalog.yaml')
    assert np.asarray(model.test.spectrum.main.shape([100., 200.])) == pytest.approx([1., 2.])

def test_free_one_normalization_preserves_components():
    source = PointSource('test', l=0, b=0, components=[
        SpectralComponent('one', Gaussian(F=2., mu=200., sigma=10.)),
        SpectralComponent('two', Gaussian(F=3., mu=400., sigma=20.)),
    ])
    model = Model(source)
    for parameter in model.parameters.values():
        parameter.fix = True
    anchor = free_source_normalization(model, 'test')
    assert len(model.free_parameters) == 1
    anchor.value *= 2
    assert source.spectrum.two.Gaussian.F.value == pytest.approx(6.)

def test_native_events_and_background_receive_same_interval(tmp_path):
    from histpy import Axes
    read_catalog_observation=notebook_definitions()['read_catalog_observation']
    dense_counts=notebook_definitions()['dense_counts']
    events = fits.BinTableHDU.from_columns([
        fits.Column(name=name, format='D', array=values) for name, values in {
            'TimeTags': [0., 1., 2.], 'Energies': [200., 300., 400.],
            'Phi': [.5, .5, .5], 'Chi galactic': [0., 0., 0.],
            'Psi galactic': [0., 0., 0.],
        }.items()
    ])
    path = tmp_path/'events.fits'
    fits.HDUList([fits.PrimaryHDU(), events]).writeto(path)
    axes = Axes([Axis([100, 1000], label='Em', unit=u.keV),
                 Axis([0, 180], label='Phi', unit=u.deg),
                 HealpixAxis(nside=1, coordsys=Galactic(), label='PsiChi')])
    full = read_catalog_observation(path, axes=axes)
    selected = read_catalog_observation(path, axes=axes, start=1., stop=2.)
    background = read_catalog_observation(path, axes=axes, start=1., stop=2.)
    assert dense_counts(full).sum() == 3
    assert dense_counts(selected).sum() == 1  # Half-open time interval.
    assert np.array_equal(dense_counts(selected), dense_counts(background))
    full.write(tmp_path/'binned.hdf5')
    with pytest.raises(ValueError, match='time-resolved event FITS'):
        read_catalog_observation(tmp_path/'binned.hdf5', start=1., stop=2.)

def test_response_cache_changes_with_actual_selected_history(tmp_path):
    from cosipy import test_data
    from cosipy.response import ExtendedSourceResponse
    from cosipy.spacecraftfile import SpacecraftHistory
    cached_extended_response=notebook_definitions()['cached_extended_response']

    history = SpacecraftHistory.open(test_data.path/'20280301_2s.fits')
    orientation_file, response_file = tmp_path/'orientation', tmp_path/'response'
    orientation_file.write_text('input orientation')
    response_file.write_text('input response')
    sky = HealpixAxis(nside=1, coordsys=Galactic(), label='NuLambda')
    axes = [sky, Axis([100, 1000], label='Ei', unit=u.keV),
            Axis([100, 1000], label='Em', unit=u.keV),
            Axis([0, 180], label='Phi', unit=u.deg),
            HealpixAxis(nside=1, coordsys=Galactic(), label='PsiChi')]

    class Response:
        calls = 0

        def get_extended_source_response(self, history, **kwargs):
            self.calls += 1
            return ExtendedSourceResponse(axes, contents=np.ones((12, 1, 1, 1, 12)),
                                          unit=u.cm**2*u.s)

    response = Response()
    options = dict(orientation_path=orientation_file, response_path=response_file,
                   nside_image=1, nside_scatt_map=2)
    _, first = cached_extended_response(response, history, tmp_path/'cache', **options)
    _, reused = cached_extended_response(response, history, tmp_path/'cache', **options)
    assert first == reused and response.calls == 1
    partial_history = history.select_interval(stop=history.obstime[0]+.5*u.s)
    _, changed = cached_extended_response(response, partial_history, tmp_path/'cache', **options)
    assert changed != first and response.calls == 2
    assert not list((tmp_path/'cache').glob('*.json'))

def test_delivered_catalog_population_and_flat_assets(catalog_directory):
    import yaml
    folder = catalog_directory
    model = load_source_catalog(folder/'dc4_catalog.yaml')
    assert (len(model.point_sources), len(model.extended_sources)) == (32, 19)
    assert not model.free_parameters
    assert not any(name.startswith(('grb_', 'mgf_')) for name in model.sources)
    assert 'four_c_21p35' in model.sources and 'three_c_454p3' in model.sources
    assert 'four_c_21p35_flare' not in model.sources
    assert len(model.crab.components) == 4
    content = yaml.safe_load((folder/'dc4_catalog.yaml').read_text())
    assert len(content) == 51
    assert not any(p.is_dir() for p in folder.iterdir())
    for path in folder.iterdir():
        assert path.suffix in ('.yaml', '.fits')
        if path.suffix == '.fits':
            with fits.open(path, checksum=True) as hdul:
                assert all(h.verify_checksum() in (1,2) and h.verify_datasum() in (1,2) for h in hdul)

def test_fitting_notebook_has_no_catalog_helper_imports():
    text=NOTEBOOK.read_text()
    assert 'cosipy.util.catalog_fitting' not in text
    assert 'cosipy.util.source_catalog' not in text
    assert 'cosipy.util.megalib' not in text
    assert 'dc4_catalog_tools' not in text
