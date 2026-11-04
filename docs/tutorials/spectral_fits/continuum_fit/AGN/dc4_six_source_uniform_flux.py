#!/usr/bin/env python3
"""Compare full-light-curve and 1000-chunk incident fluence for six DC4 sources.

Run: python dc4_six_source_uniform_flux.py
     python dc4_six_source_uniform_flux.py --source-library /path/to/Source_Library

This uses only cosi-sim .source definitions and their light curves. Each .source
Flux gives the total photon flux of its Spectrum; all selected spectra are
within 100–10,000 keV. With a uniform response over that whole band, spectral
shape does not change total fluence. No detector response or FITS data are used.
"""

from __future__ import annotations

import argparse
import os
from array import array
from functools import lru_cache
from pathlib import Path

import numpy as np


# Rows to print. The two energy components in each MAXI .source are summed.
SOURCE_FILES = {
    "4C+21.35 (no flare)": ("DC3/sources/Extragalactic/4C21p35_noflare/4C21p35_noflare.source",),
    "4C+21.35 (flare)": ("DC3/sources/Extragalactic/4C21p35_flare/4C21p35_flare.source",),
    "3C 454.3 (low)": ("DC4/sources/Extragalactic/3C454p3_low/3C454p3_low.source",),
    "3C 454.3 (high)": ("DC4/sources/Extragalactic/3C454p3_high/3C454p3_high.source",),
    "MAXI J1820+070": ("DC3/sources/Galactic/MAXIJ1820/MAXIJ1820.source",),
    "MAXI J1348-630": ("DC3/sources/Galactic/MAXIJ1348/MAXIJ1348.source",),
    "PSR B1259-63": ("DC3/sources/Galactic/PSRB1259/PSRB1259.source",),
    "CO nova (continuum)": ("DC3/sources/Galactic/nova_co_continuum/nova_co_continuum.source",),
    "CO nova (511 keV)": ("DC3/sources/Galactic/nova_co_511keV/nova_co_511keV.source",),
    "CO nova (478 keV)": ("DC3/sources/Galactic/nova_co_478keV/nova_co_478keV.source",),
}
N_CHUNKS = 1000
ENERGY_MIN_KEV, ENERGY_MAX_KEV = 100.0, 10_000.0


def read_components(source_file: Path):
    """Get each component's Flux, Spectrum, and Lightcurve from one .source."""
    components = {}
    for line in source_file.read_text().splitlines():
        words = line.split("#", 1)[0].split()
        if words and "." in words[0]:
            component, field = words[0].rsplit(".", 1)
            if field in {"Flux", "Spectrum", "Lightcurve"}:
                components.setdefault(component, {})[field] = words[1:]
    if not components or any(set(fields) != {"Flux", "Spectrum", "Lightcurve"}
                             for fields in components.values()):
        raise ValueError(f"Missing Flux, Spectrum, or Lightcurve: {source_file}")
    return components.values()


def check_spectrum(source_dir: Path, description):
    """Confirm the component spectrum lies in the uniform-response energy band."""
    if description[0] == "File":
        energies = []
        for line in (source_dir / description[1]).read_text().splitlines():
            words = line.split()
            if words and words[0] == "DP":
                energies.append(float(words[1]))
        if not energies or energies[0] < ENERGY_MIN_KEV or energies[-1] > ENERGY_MAX_KEV:
            raise ValueError(f"Spectrum extends outside 100–10,000 keV: {source_dir}")
    elif description[0] == "Gaussian":
        center, sigma = map(float, description[1:3])
        if center - 5 * sigma < ENERGY_MIN_KEV or center + 5 * sigma > ENERGY_MAX_KEV:
            raise ValueError(f"Gaussian line extends outside 100–10,000 keV: {source_dir}")
    else:
        raise ValueError(f"Unsupported spectrum: {description}")


@lru_cache(maxsize=None)
def lightcurve_durations(path: Path):
    """Return full duration and duration of chunks containing any emission."""
    times, rates = array("d"), array("d")
    for line in path.read_text().splitlines():
        words = line.split()
        if words and words[0] == "DP":
            times.append(float(words[1]))
            rates.append(float(words[2]))
    t, y = np.asarray(times), np.asarray(rates)
    if len(t) <= N_CHUNKS or np.any(np.diff(t) <= 0) or np.any(y < 0):
        raise ValueError(f"Invalid light curve: {path}")

    # This matches cosi-sim's make_orientation_bins.py: adjacent chunks share
    # an endpoint, and a remainder becomes one additional chunk (1001 here).
    size, remainder = divmod(len(t), N_CHUNKS)
    chunks = [slice(i * size, (i + 1) * size + 1) for i in range(N_CHUNKS)]
    if remainder > 1:
        chunks.append(slice(N_CHUNKS * size, len(t)))

    active_seconds = 0.0
    for chunk in chunks:
        tc, yc = t[chunk], y[chunk]
        area = np.sum((yc[:-1] + yc[1:]) * np.diff(tc) / 2)
        if area > 0:  # Cosima normalizes this chunk to mean one.
            active_seconds += tc[-1] - tc[0]
    return float(t[-1] - t[0]), active_seconds


def fluences(source_library: Path):
    """Calculate incident photons/cm² for each displayed row."""
    rows = []
    observation_seconds = None
    for label, files in SOURCE_FILES.items():
        no_chunks = parallel = 0.0
        for filename in files:
            source_file = source_library / filename
            for fields in read_components(source_file):
                check_spectrum(source_file.parent, fields["Spectrum"])
                if fields["Lightcurve"][:2] != ["File", "false"]:
                    raise ValueError(f"Expected a nonrepeating light curve: {source_file}")
                lc_file = source_file.parent / fields["Lightcurve"][2]
                full_seconds, active_seconds = lightcurve_durations(lc_file)
                if observation_seconds is None:
                    observation_seconds = full_seconds
                elif full_seconds != observation_seconds:
                    raise ValueError(f"Light-curve durations differ: {lc_file}")

                flux = float(fields["Flux"][0])  # Integrated ph/cm²/s.
                no_chunks += flux * full_seconds  # One full-LC unit mean.
                parallel += flux * active_seconds  # Unit mean per active chunk.
        rows.append((label, no_chunks, parallel))
    return rows


def main():
    sibling_library = (Path(__file__).resolve().parents[6] /
                       "cosi-sim/cosi_sim/Source_Library")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-library", type=Path,
                        default=Path(os.environ.get("COSI_SIM_SOURCE_LIBRARY", sibling_library)))
    args = parser.parse_args()

    print(f"{'Component':24} {'No chunks (ph/cm²)':>22} {'1000 chunks (ph/cm²)':>22}")
    for label, no_chunks, parallel in fluences(args.source_library):
        print(f"{label:24} {no_chunks:22.3f} {parallel:22.3f}")


if __name__ == "__main__":
    main()
