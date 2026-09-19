"""Ratio-only plotting can run from a snapshot without any COSI response inputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / 'docs/tutorials/spectral_fits/continuum_fit/AGN'


def notebook(name):
    return json.loads((DIRECTORY / name).read_text())


def cell_code(nb, identifier):
    return ''.join(next(c for c in nb['cells'] if c['id'] == identifier)['source'])


@pytest.fixture
def api():
    nb = notebook('Source_catalog_DC4_lightcurve_comparison_plots.ipynb')
    scope = {}
    exec(cell_code(nb, 'plot-helpers'), scope)
    yield scope
    plt.close('all')


@pytest.fixture
def snapshot():
    sources = ['maxi_j1348', 'co_nova_continuum', 'grb_bn090424592', 'mgf_051103']
    records = []
    for selection in ['full', 'NGC4151_GTI']:
        for i, source in enumerate(sources):
            # Partial and completely empty FITS selections exercise undefined ratios.
            fits = [10., 0.] if selection == 'full' or i == 0 else [0., 0.]
            records.append({'selection': selection, 'source': source, 'counts': {
                'FITS': fits,
                'Parallel chunks + LC response': [0., 2.],
                'No chunks + LC response': [20., 1.],
                'Parallel chunks, duration only': [5., 0.],
            }})
    return {'schema': 'dc4-lightcurve-count-comparison-v1',
            'energy_edges_keV': [100., 200., 400.], 'variable_sources': sources,
            'records': records, 'created_utc': 'test-snapshot', 'provenance': {'test': True}}


def test_ratios_preserve_zero_models_and_mask_zero_denominators(api):
    ratios = api['model_fits_ratio']([0., 4., 1.], [2., 2., 0.])
    np.testing.assert_allclose(ratios[:2], [0., 2.])
    assert np.isnan(ratios[2])


def test_plots_only_show_ratios(api, snapshot):
    figures = api['plot_total_ratios'](snapshot)
    assert len(figures) == 2 and all(len(fig.axes) == 1 for fig in figures)
    np.testing.assert_allclose(figures[0].axes[0].lines[0].get_ydata(), [.2] * 4)
    figure = api['plot_energy_ratios'](snapshot, snapshot['variable_sources'])
    assert len(figure.axes) == 8
    for ax in [*(f.axes[0] for f in figures), *figure.axes]:
        assert ax.get_ylabel() == 'Model / FITS'
    assert any('Ratios undefined' in text.get_text() for ax in figure.axes for text in ax.texts)


def test_standalone_fresh_namespace(api, snapshot, tmp_path, monkeypatch):
    (tmp_path / 'source_catalog_DC4_lightcurve_comparison.json').write_text(json.dumps(snapshot))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, 'show', lambda: None)
    scope = {'__name__': '__main__'}
    for cell in notebook('Source_catalog_DC4_lightcurve_comparison_plots.ipynb')['cells']:
        if cell['cell_type'] == 'code':
            exec(''.join(cell['source']), scope)
    assert len(scope['total_ratio_figures']) == 2
    assert len(scope['energy_ratio_figure'].axes) == 8
    # The plotter has no source-model or instrument-response bindings.
    assert 'detector_response' not in scope and 'simulation_units' not in scope


def test_missing_cache_explains_export_step(api, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nb = notebook('Source_catalog_DC4_lightcurve_comparison_plots.ipynb')
    with pytest.raises(FileNotFoundError, match='Save a lightweight comparison cache'):
        exec(cell_code(nb, 'plot-load-cache'), api)


def test_cache_export_roundtrip(api, snapshot, tmp_path):
    nb = notebook('Source_catalog_generator_all_DC4_lightcurve.ipynb')
    spectra = {(r['selection'], r['source']): {k: np.array(v) for k, v in r['counts'].items()}
               for r in snapshot['records']}
    api.update(comparison_spectra=spectra,
               measurement_axes={'Em': SimpleNamespace(edges=SimpleNamespace(
                   to_value=lambda unit: np.array(snapshot['energy_edges_keV'])))},
               injection_blocks={k: [{'Lightcurve': ['File', 'False', 'example.dat']}]
                                 for k in snapshot['variable_sources']},
               catalog_paths={'full': tmp_path / 'full.yaml', 'NGC4151_GTI': tmp_path / 'gti.yaml'},
               CATALOG_DIRECTORY=tmp_path, ORIENTATION_PATH=Path('observation.ori'),
               RESPONSE_PATH=Path('response.h5'), AGN_BINNING_CONFIG=Path('agn.yaml'),
               TIME_BASIS='elapsed', PARALLEL_SIMULATION_BINS=1000,
               PARALLEL_SIMULATION_BINS_BY_SOURCE={'co_nova_continuum': 500},
               LIGHTCURVE_MODE_BY_SOURCE={}, u=SimpleNamespace(keV='keV'))
    exec(cell_code(nb, 'lc-export-comparison-cache'), api)
    result = json.loads((tmp_path / 'source_catalog_DC4_lightcurve_comparison.json').read_text())
    assert result['records'] == snapshot['records']
    assert result['energy_edges_keV'] == snapshot['energy_edges_keV']
    assert result['provenance']['parallel_bins_by_source'] == {'co_nova_continuum': 500}


def test_invalid_snapshot_rejected(api, snapshot):
    snapshot['records'][0]['counts']['FITS'][0] = -1.
    with pytest.raises(ValueError, match='Invalid count spectrum'):
        api['validate_comparison_snapshot'](snapshot)


def test_generator_and_standalone_use_same_plots():
    generator = notebook('Source_catalog_generator_all_DC4_lightcurve.ipynb')
    standalone = notebook('Source_catalog_DC4_lightcurve_comparison_plots.ipynb')
    for left, right in [('lc-ratio-plot-helpers', 'plot-helpers'),
                        ('lc-total-comparison', 'plot-total'),
                        ('lc-energy-comparison', 'plot-energy')]:
        assert cell_code(generator, left) == cell_code(standalone, right)
