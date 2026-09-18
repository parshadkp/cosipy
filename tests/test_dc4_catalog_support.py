"""Small, portable tests of tutorial asset preparation and native-event GTIs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from astropy.io import fits


@pytest.fixture(scope='module')
def support():
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


def test_spatial_conversion_units_and_no_overwrite(support, tmp_path):
    source = tmp_path / 'map.dat'
    text = 'PA -180 180\nTA 0 180\nEA 100 1000\n'
    text += ''.join(f'AP {i} {j} {k} {k+1}\n'
                    for i in range(2) for j in range(2) for k in range(2))
    source.write_text(text)
    output = tmp_path / 'template.fits'
    support.megalib_map_to_galprop_fits(source, output)
    with fits.open(output) as hdul:
        assert hdul['SKYMAP'].header['NSIDE'] == 8
        assert hdul['SKYMAP'].header['MAPVERS'] == 'perMeV_v3'
        table = np.column_stack([hdul['SKYMAP'].data[n] for n in hdul['SKYMAP'].columns.names])
        assert table.shape == (768, 6)
        np.testing.assert_allclose(table[:, 2:4], np.tile([1000., 2000.], (768, 1)))
        np.testing.assert_allclose(table[:, [0, 1, 4, 5]], 0.)
        np.testing.assert_allclose(hdul['ENERGIES'].data['ENERGY'][2:4], [.1, 1.])
    before = output.read_bytes(), output.stat().st_mtime_ns
    source.unlink()  # Reuse does not require reconverting the input.
    assert support.ensure_spatial_template('unused', output, tmp_path)[1] == 'loaded'
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before


def test_missing_template_resolves_original_source_map(support, tmp_path, monkeypatch):
    library = tmp_path / 'Source_Library'
    directory = library / 'DC3/sources/example'
    directory.mkdir(parents=True)
    (directory / 'example.source').write_text(
        'DataChallenge.Source Test\nTest.Beam FarFieldNormalizedEnergyBeam sky.dat\n')
    (directory / 'sky.dat').write_text('map placeholder')
    monkeypatch.setitem(support.SPATIAL_DEFINITIONS, 'example', ('DC3', 'example'))
    calls = []
    def convert(input_path, output_path, **kwargs):
        calls.append((input_path, output_path))
        return output_path
    monkeypatch.setattr(support, 'megalib_map_to_galprop_fits', convert)
    output = tmp_path / 'example.fits'
    assert support.ensure_spatial_template('example', output, library)[1] == 'created'
    assert calls == [(directory / 'sky.dat', output)]
    assert support.source_library_path('/old/cosi-sim/Source_Library/DC3/sources/example/sky.dat', library) == str(directory / 'sky.dat')


def test_native_event_gti_and_binning(support, tmp_path):
    from histpy import Histogram
    config = tmp_path / 'agn.yaml'
    config.write_text(yaml.safe_dump({'energy_bins': [100., 200., 400.],
                                     'phi_pix_size': 90., 'nside': 1, 'scheme': 'ring'}))
    path = tmp_path / 'events.fits'
    times = np.array([0., 1., 1.999, 2., 3., 4., 5.])
    columns = [fits.Column(name='TimeTags', format='D', array=times)]
    columns += [fits.Column(name=name, format='D', array=np.full(times.size, value))
                for name, value in [('Energies', 150.), ('Phi', np.pi / 4),
                                     ('Chi galactic', 155.), ('Psi galactic', 75.)]]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    gti = SimpleNamespace(tstart_list=SimpleNamespace(unix=np.array([1., 3.])),
                          tstop_list=SimpleNamespace(unix=np.array([2., 4.])) )
    hist = support.load_selected_histogram(path, gti, config, already_selected=False)
    assert hist.project('Em').to_dense().contents.sum() == 3.
    assert hist.axes['PsiChi'].nside == 1
    binned = tmp_path / 'selected.hdf5'
    hist.write(binned)
    loaded = support.load_selected_histogram(binned, gti, config)
    np.testing.assert_allclose(loaded.to_dense().contents, hist.to_dense().contents)
    with pytest.raises(ValueError, match='native event FITS'):
        support.load_selected_histogram(binned, gti, config, already_selected=False)


def test_dat_reader_accepts_variable_trailing_columns(tmp_path):
    from cosipy.threeml.custom_functions import SpecFromDat
    path = tmp_path / 'spectrum.dat'
    path.write_text('# header\nIP LINLIN\nDP 100 2\nDP 200 1 trailing note\nDP 400 0\nEN\n')
    spectrum = SpecFromDat(K=1., dat=str(path))
    np.testing.assert_allclose(spectrum(np.array([100., 200., 400.])), [2./400, 1./400, 0.])


def test_source_specific_folding_overrides_only_named_source():
    from astromodels import Model, PointSource, Powerlaw
    from histpy import Histogram, Axis, Axes
    from cosipy.response import BinnedThreeMLModelFolding
    axes = Axes([Axis([0., 1., 2.], label='Em')])
    class Response:
        def __init__(self, scale):
            self.axes, self.scale = axes, scale
        def copy(self):
            return Response(self.scale)
        def set_source(self, source):
            self.source = source
        def expectation(self, copy=True):
            return Histogram(axes, contents=np.ones(2) * self.scale * self.source.spectrum.main.Powerlaw.K.value)
    model = Model(PointSource('steady', l=0., b=0., spectral_shape=Powerlaw(K=1.)),
                  PointSource('variable', l=0., b=0., spectral_shape=Powerlaw(K=2.)))
    folding = BinnedThreeMLModelFolding(SimpleNamespace(axes=axes), Response(1.),
                                      source_specific_point_source_responses={'variable': Response(3.)})
    folding.set_model(model)
    np.testing.assert_allclose(folding.expectation().contents, [7., 7.])
    model.variable.spectrum.main.Powerlaw.K.value = 4.
    np.testing.assert_allclose(folding.expectation().contents, [13., 13.])
