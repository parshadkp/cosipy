"""Optional DC4 simulation checks, kept on SourceCatTest rather than in COSIpy."""
from copy import deepcopy
import gzip
import hashlib
from pathlib import Path

import astropy.units as u
from astropy.io import fits
import numpy as np
from astromodels import Model
from histpy import Axes, HealpixAxis, Histogram

from cosipy.data_io import EmCDSBinnedData
from cosipy.response import (BinnedInstrumentResponse, BinnedThreeMLModelFolding,
    BinnedThreeMLPointSourceResponse, BinnedThreeMLExtendedSourceResponse)


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def simulated_energy_counts(path, energy_edges_keV, *, start=None, stop=None, chunk_rows=100000):
    """Histogram native DC4 energies without decompressing a whole event table.

    Read ordinary fixed-width FITS rows in blocks. Only the energy and time
    columns are used. The file must already have the agreed DC4 quality cuts.
    ``start``/``stop`` use the full orientation range, not a target pointing cut.
    """
    path = Path(path)
    if (start is None) != (stop is None) or (start is not None and stop <= start):
        raise ValueError('Supply both increasing Unix limits, or neither')
    with fits.open(path, lazy_load_hdus=True) as hdul:
        hdu = hdul[1]
        rows, row_bytes = hdu.header['NAXIS2'], hdu.header['NAXIS1']
        offset = hdu.fileinfo()['datLoc']
        dtype = hdu.columns.dtype.newbyteorder('>')
        if dtype.itemsize != row_bytes or dtype.hasobject:
            raise ValueError('Expected a fixed-width DC4 event table')
        column = hdu.columns['Energies']
        scale, zero = column.bscale or 1., column.bzero or 0.
        energy_unit = u.Unit(column.unit) if column.unit else u.keV
        to_keV = energy_unit.to(u.keV)
    counts = np.zeros(len(energy_edges_keV)-1, dtype=np.int64)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        stream.seek(offset)
        for first in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows-first)
            payload = stream.read(count*row_bytes)
            if len(payload) != count*row_bytes:
                raise ValueError(f'Truncated FITS event table: {path}')
            events = np.frombuffer(payload, dtype=dtype, count=count)
            selected = np.ones(count, dtype=bool)
            if start is not None:
                selected = (events['TimeTags'] >= start) & (events['TimeTags'] < stop)
            energy = (events['Energies'][selected]*scale+zero)*to_keV
            if np.any(~np.isfinite(energy)):
                raise ValueError(f'Nonfinite energies: {path}')
            counts += np.histogram(energy, energy_edges_keV)[0]
    return counts


def compare_catalog_sources(catalog, source_files, names, detector_response, history,
                            cache_directory, *, model_cache_key, extended_response=None,
                            extended_response_factory=None):
    """Fold static catalog models and compare the same measured-energy bins.

    Observed events are never used to adjust the catalog. Grouped AGN event
    files are summed once. Per-source NPZ caches contain only numerical results;
    their key changes when an event file, catalog asset, history or response does.
    """
    axes = Axes([detector_response.axes['Em'], detector_response.axes['Phi'],
                 HealpixAxis(nside=detector_response.axes['PsiChi'].nside,
                             coordsys='galactic', label='PsiChi')])
    data = EmCDSBinnedData(Histogram(axes))
    point_response = BinnedThreeMLPointSourceResponse(
        data, BinnedInstrumentResponse(detector_response, data), history,
        detector_response.axes['Ei'], polarization_axis=None,
        nside=2*axes['PsiChi'].nside,
    )
    energy_edges = axes['Em'].edges.to_value(u.keV)
    # Include in-memory changes as well as the caller's input-file fingerprints.
    # Changing a source amplitude or history must not reuse an old count spectrum.
    model_and_history = hashlib.sha256(model_cache_key.encode())
    model_and_history.update(repr(catalog.to_dict()).encode())
    model_and_history.update(np.asarray(history.obstime.unix, dtype=np.float64).tobytes())
    model_and_history.update(np.asarray(history.livetime.to_value(u.s), dtype=np.float64).tobytes())
    model_key = model_and_history.hexdigest()
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for number, name in enumerate(names, 1):
        print(f'[{number}/{len(names)}] {name}', flush=True)
        paths = [Path(p) for p in source_files[name]]
        key = hashlib.sha256((model_key + name + ''.join(file_hash(p) for p in paths)).encode()).hexdigest()
        cache = cache_directory/f'{name}_{key[:16]}.npz'
        if cache.exists():
            with np.load(cache) as saved:
                results[name] = {'model':saved['model'], 'simulation':saved['simulation']}
            continue
        source = deepcopy(catalog.sources[name])
        for component in source.components.values():
            if hasattr(component.polarization, 'degree'):
                component.polarization.degree.value = 0.
        if name in catalog.extended_sources and extended_response is None:
            if extended_response_factory is None:
                raise ValueError('Supply the full-observation extended response for an extended source')
            extended_response = extended_response_factory()
        extended = (BinnedThreeMLExtendedSourceResponse(data, extended_response)
                    if name in catalog.extended_sources else None)
        folding = BinnedThreeMLModelFolding(data, point_response, extended)
        folding.set_model(Model(source))
        model_counts = np.asarray(folding.expectation().project('Em').contents, dtype=float)
        if np.any(~np.isfinite(model_counts)) or np.any(model_counts < 0):
            raise ValueError(f'Invalid model prediction: {name}')
        simulated = sum((simulated_energy_counts(
            path, energy_edges, start=history.tstart.unix, stop=history.tstop.unix,
        ) for path in paths), start=np.zeros(len(energy_edges)-1, dtype=np.int64))
        np.savez_compressed(cache, model=model_counts, simulation=simulated)
        results[name] = {'model':model_counts, 'simulation':simulated}
    return energy_edges, results
