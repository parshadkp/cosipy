"""Generation checks: only on SourceCatTest."""
from pathlib import Path
import sys
import numpy as np
import pytest
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import Galactic
from astromodels import ExtendedSource
from histpy import Axis, HealpixAxis
from cosipy.response.functions_3d import get_integrated_extended_model_3d
from cosipy.threeml.custom_functions import GalpropHealpixModel
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dc4_catalog_tools.generation import _read_dp_table, time_average_factor
from dc4_catalog_tools.megalib import megalib_map_to_galprop_fits


@pytest.mark.parametrize('compressed', [False, True])
def test_streaming_event_spectra_match_exact_time_selection(tmp_path, compressed):
    from dc4_catalog_tools.comparison import simulated_energy_counts
    path=tmp_path/('events.fits.gz' if compressed else 'events.fits')
    table=fits.BinTableHDU.from_columns([
        fits.Column(name='TimeTags',format='D',array=[0.,1.,2.,3.]),
        fits.Column(name='Energies',format='D',unit='keV',array=[150.,250.,350.,450.]),
    ])
    fits.HDUList([fits.PrimaryHDU(),table]).writeto(path)
    edges=np.array([100.,200.,300.,400.,500.])
    assert np.array_equal(simulated_energy_counts(path,edges,chunk_rows=2),[1,1,1,1])
    assert np.array_equal(simulated_energy_counts(path,edges,start=1.,stop=3.,chunk_rows=2),[0,1,1,0])


def test_simulation_manifest_groups_states_once():
    from dc4_catalog_tools.simulation_files import SIMULATION_FILES
    assert len(SIMULATION_FILES)==51
    assert len(SIMULATION_FILES['four_c_21p35'])==2
    assert len(SIMULATION_FILES['three_c_454p3'])==2
    assert len(SIMULATION_FILES['crab'])==1
    assert not any(k.startswith(('grb_','mgf_')) for k in SIMULATION_FILES)


def test_count_cache_changes_with_livetime_and_simulated_events(tmp_path):
    from astromodels import Model, PointSource, Powerlaw
    from cosipy import test_data
    from cosipy.response import FullDetectorResponse
    from cosipy.spacecraftfile import SpacecraftHistory
    from dc4_catalog_tools.comparison import compare_catalog_sources
    history=SpacecraftHistory.open(test_data.path/'20280301_2s.fits')
    response=FullDetectorResponse.open(test_data.path/'test_full_detector_response.h5')
    catalog=Model(PointSource('test',l=50,b=-45,spectral_shape=Powerlaw(K=.01,piv=100,index=-2)))
    events=tmp_path/'events.fits'

    def write_event(energy):
        table=fits.BinTableHDU.from_columns([
            fits.Column(name='TimeTags',format='D',array=[history.tstart.unix+.5]),
            fits.Column(name='Energies',format='D',array=[energy]),
        ])
        fits.HDUList([fits.PrimaryHDU(),table]).writeto(events,overwrite=True)

    def run(selected):
        return compare_catalog_sources(catalog,{'test':[events]},['test'],response,selected,
                                       tmp_path/'cache',model_cache_key='fixture')[1]['test']

    write_event(150.)
    first=run(history)
    doubled=SpacecraftHistory(history.obstime,history.attitude,history.location,livetime=history.livetime*2)
    second=run(doubled)
    assert second['model']==pytest.approx(first['model']*2)
    assert np.array_equal(second['simulation'],first['simulation'])
    write_event(350.)
    third=run(history)
    assert third['simulation'].sum()==1
    assert not np.array_equal(third['simulation'],first['simulation'])
    assert len(list((tmp_path/'cache').glob('*.npz')))==3


def test_short_flare_full_mean_is_not_duty_cycle_twice():
    # A brief pulse is already scaled to mean one over the supplied LC span.
    assert time_average_factor([0, 8, 9, 10], [0, 0, 100, 0], 0, 10) == pytest.approx(1.)
    assert time_average_factor([0, 8, 9, 10], [0, 0, 100, 0], 0, 20) == pytest.approx(.5)

def test_finite_profile_zero_overlap_and_zero_curve():
    assert time_average_factor([10, 11, 12], [0, 1, 0], 20, 30) == 0.
    assert time_average_factor([10, 11, 12], [0, 0, 0], 0, 30) == 0.

def test_periodic_profile_partial_cycles():
    # Mean-one triangular profile: area in [0,.5] is .25, not .5.
    t, y = [0, 1, 2], [0, 2, 0]
    assert time_average_factor(t, y, 0, .5, repeating=True) == pytest.approx(.5)
    assert time_average_factor(t, y, 0, 2000, repeating=True) == pytest.approx(1.)
    assert time_average_factor(t, y, -2, 2, repeating=True) == pytest.approx(1.)

@pytest.mark.parametrize('times,flux,start,stop', [([0,0],[1,2],0,1),([0,1],[1,-1],0,1),([0,1],[1,2],1,1),([0,np.nan],[1,2],0,1)])
def test_invalid_lightcurves(times, flux, start, stop):
    with pytest.raises(ValueError):
        time_average_factor(times, flux, start, stop)

def test_megalib_zero_flux_loglog_fallback(tmp_path):
    path = tmp_path/'spectrum.dat'
    path.write_text('IP LOGLOG\nDP 100 1\nDP 200 0\nDP 300 1\n')
    _, _, mode = _read_dp_table(path, spectral=True)
    assert mode == 'LINLIN'

def test_megalib_map_keeps_line_area_at_native_knots(tmp_path):
    path = tmp_path/'map.dat'
    rows = ['PA -180 180', 'TA 0 180', 'EA 510 511 512']
    for i in range(2):
        for j in range(2):
            for k, value in enumerate([0, 3, 0]):
                rows.append(f'AP {i} {j} {k} {value}')
    path.write_text('\n'.join(rows))
    output = megalib_map_to_galprop_fits(path, tmp_path/'map.fits', nside=1,
                                       oversample_nside=8, energy_bounds=(100, 1000))
    spatial = GalpropHealpixModel()
    spatial._fitsfile = str(output)
    model = ExtendedSource('diffuse', spatial_shape=spatial)
    sky = HealpixAxis(nside=1, coordsys=Galactic(), label='NuLambda')
    energy = Axis([100, 1000], label='Ei', unit=u.keV)
    flux = get_integrated_extended_model_3d(model, sky, energy)
    assert flux.contents[:, 0].value == pytest.approx(np.full(12, 3.))
    # A previous zero normalization cannot poison the cached unit template.
    spatial.K.value = 0.
    assert np.all(get_integrated_extended_model_3d(model, sky, energy).contents == 0.)
    spatial.K.value = 2.
    assert get_integrated_extended_model_3d(model, sky, energy).contents[:, 0] == pytest.approx(np.full(12, 6.))
    # A different Ei grid must trigger new integration.
    fine = Axis([100, 511, 1000], label='Ei', unit=u.keV)
    actual = get_integrated_extended_model_3d(model, sky, fine).contents.value
    assert actual == pytest.approx(np.full((12, 2), 3.))
