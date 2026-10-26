"""Convert coupled MEGAlib spatial/energy tables into GALPROP HEALPix FITS.

The input PA/TA axes are Galactic longitude and colatitude in degrees; EA is
in keV and AP is differential intensity per keV. Output intensities are per
MeV on a Galactic RING HEALPix grid, suitable for GalpropHealpixModel.
"""
import gzip
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits

__all__ = ["read_normalized_energy_beam", "megalib_map_to_galprop_fits"]
SPATIAL_TEMPLATE_FORMAT_TAG = "perMeV_v3"


def read_normalized_energy_beam(path):
    """Read plain/gzipped PA/TA/EA/AP data into axes and a dense intensity cube."""
    opener = gzip.open if Path(path).suffix == ".gz" else open
    pa = ta = ea = None
    entries = []
    with opener(path, "rt") as stream:
        for raw_line in stream:
            tokens = raw_line.split()
            if not tokens:
                continue
            if tokens[0] == "PA":
                pa = np.asarray(tokens[1:], dtype=float)
            elif tokens[0] == "TA":
                ta = np.asarray(tokens[1:], dtype=float)
            elif tokens[0] == "EA":
                ea = np.asarray(tokens[1:], dtype=float)
            elif tokens[0] == "AP":
                entries.append(
                    (
                        int(tokens[1]),
                        int(tokens[2]),
                        int(tokens[3]),
                        float(tokens[4]),
                    )
                )
    if pa is None or ta is None or ea is None:
        raise ValueError(f"Missing PA/TA/EA axes in {path}")
    values = np.zeros((pa.size, ta.size, ea.size), dtype=np.float32)
    for i_pa, i_ta, i_ea, value in entries:
        values[i_pa, i_ta, i_ea] = value
    return pa, ta, ea, values


def _nearest_axis_indices(axis, values):
    order = np.argsort(axis)
    sorted_axis = axis[order]
    insertion = np.searchsorted(sorted_axis, values)
    insertion = np.clip(insertion, 1, sorted_axis.size - 1)
    left = sorted_axis[insertion - 1]
    right = sorted_axis[insertion]
    choose_right = np.abs(values - right) < np.abs(values - left)
    nearest = insertion - 1 + choose_right.astype(int)
    return order[nearest]


def megalib_map_to_galprop_fits(
    input_path,
    output_path,
    nside=8,
    overwrite=False,
    *,
    oversample_nside=64,
    energy_bounds=None,
):
    """Convert a Galactic MEGAlib map, saving or reusing a spatial FITS template.

    Sample the PA/TA grid at ``max(nside, oversample_nside)`` and area-average
    onto ``nside``. Convert keV energies/intensities to MeV and add zero anchors
    outside the simulated energy range to suppress extrapolated line tails.
    ``energy_bounds``, if supplied, is a pair of positive increasing keV values
    to which the outer zero anchors extend (e.g. the response's energy range).
    It does not truncate the input spectrum. Existing output is reused unless
    ``overwrite=True``; cache compatibility is the caller's responsibility.
    Returns the resolved output Path. No instrument response is generated.
    """
    output_path = Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        return output_path

    pa, ta, energy_kev, values = read_normalized_energy_beam(input_path)
    sampling_nside = max(int(nside), int(oversample_nside))
    npix = hp.nside2npix(sampling_nside)
    longitude, latitude = hp.pix2ang(
        sampling_nside,
        np.arange(npix),
        lonlat=True,
    )
    longitude = ((longitude + 180.0) % 360.0) - 180.0
    theta = 90.0 - latitude
    i_pa = _nearest_axis_indices(pa, longitude)
    i_ta = _nearest_axis_indices(ta, theta)
    sampled_values = values[i_pa, i_ta, :]

    if sampling_nside == int(nside):
        healpix_values = sampled_values
    else:
        # The table stores intensity, so power=0 preserves the area-weighted
        # mean intensity and therefore the total flux after pixel-area folding.
        healpix_values = np.column_stack(
            [
                hp.ud_grade(
                    sampled_values[:, energy_index],
                    nside_out=int(nside),
                    order_in="RING",
                    order_out="RING",
                    power=0,
                )
                for energy_index in range(sampled_values.shape[1])
            ]
        ).astype(np.float32, copy=False)

    # GalpropHealpixModel interpolates at response-bin edges.  Add zero
    # anchors outside the simulated band to avoid extrapolated line tails.
    energy_mev = energy_kev / 1000.0
    # MEGAlib energies and differential intensities are per keV, whereas
    # GalpropHealpixModel expects energy in MeV and intensity per MeV.
    # Preserve flux under the coordinate change: f_MeV = 1000 * f_keV.
    healpix_values *= 1000.0
    if np.any(np.diff(energy_mev) <= 0.0):
        raise ValueError(f"Energy axis is not strictly increasing in {input_path}")
    epsilon = 1.0e-6
    lower_inner = energy_mev[0] * (1.0 - epsilon)
    upper_inner = energy_mev[-1] * (1.0 + epsilon)
    if energy_bounds is None:
        energy_bounds = (energy_kev[0], energy_kev[-1])
    bounds = np.asarray(energy_bounds, dtype=float)
    if (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
            or bounds[0] <= 0 or bounds[1] <= bounds[0]):
        raise ValueError("energy_bounds must be two positive increasing keV values")
    lower_outer = min(bounds[0] / 1000.0, lower_inner)
    upper_outer = max(bounds[1] / 1000.0, upper_inner)
    if lower_outer >= lower_inner:
        lower_outer = lower_inner * (1.0 - epsilon)
    if upper_outer <= upper_inner:
        upper_outer = upper_inner * (1.0 + epsilon)
    energy_mev = np.concatenate(
        ([lower_outer, lower_inner], energy_mev, [upper_inner, upper_outer])
    )
    zero_columns = np.zeros((healpix_values.shape[0], 2), dtype=np.float32)
    healpix_values = np.column_stack(
        (
            zero_columns,
            healpix_values,
            zero_columns,
        )
    )

    skymap_columns = [
        fits.Column(
            name=f"ENERGY{index:03d}",
            format="E",
            unit="ph cm-2 s-1 sr-1 MeV-1",
            array=healpix_values[:, index],
        )
        for index in range(energy_mev.size)
    ]
    skymap = fits.BinTableHDU.from_columns(
        skymap_columns,
        name="SKYMAP",
    )
    energies = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="ENERGY",
                format="E",
                unit="MeV",
                array=energy_mev,
            )
        ],
        name="ENERGIES",
    )
    skymap.header["NSIDE"] = nside
    skymap.header["ORDERING"] = "RING"
    skymap.header["COORDSYS"] = "GAL"
    skymap.header["MAPVERS"] = SPATIAL_TEMPLATE_FORMAT_TAG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), skymap, energies]).writeto(
        output_path,
        overwrite=overwrite,
    )
    return output_path
