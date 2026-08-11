"""Richardson--Lucy sky and spectrum reconstruction helpers.

The implementation follows the multi-energy image reconstruction demonstrated
in the COSIpy v0.2.1 Crab ScAtt tutorial.  The model is a two-dimensional
``(sky pixel, incident energy)`` histogram, so the reconstructed spectrum is
obtained by integrating each incident-energy map over solid angle.
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from histpy import Histogram
from mhealpy import HealpixBase
from tqdm.auto import tqdm
from yayc import Configurator

from cosipy.event_selection import GoodTimeInterval
from cosipy.response import ExtendedSourceResponse, FullDetectorResponse
from cosipy.spacecraftfile import SpacecraftHistory

from .algorithms.RichardsonLucyAdvanced import RichardsonLucyAdvanced
from .data_interfaces.dataIF_COSI_DC2 import DataIF_COSI_DC2
from .data_interfaces.data_interface_collection import DataInterfaceCollection
from .models.allskyimage import AllSkyImageModel


def prepare_galactic_histograms(
    source_data: str | Path | Histogram,
    background: str | Path | Histogram,
    *,
    data_contains_background: bool = False,
) -> tuple[Histogram, Histogram, Histogram]:
    """Load, project, and align Galactic-CDS source and background data.

    The supplied DC4 background has a time axis while the simulated source has
    already been accumulated over the selected observation.  The time axis is
    therefore projected out before the histograms are combined.
    """

    source = (
        Histogram.open(source_data)
        if isinstance(source_data, (str, Path))
        else source_data.copy()
    )
    bkg = (
        Histogram.open(background)
        if isinstance(background, (str, Path))
        else background.copy()
    )

    source = source.project("Em", "Phi", "PsiChi")
    bkg = bkg.project("Em", "Phi", "PsiChi")
    source.axes["Em"].axis_scale = bkg.axes["Em"].axis_scale
    source = source.to(unit=bkg.unit, update=False)

    if source.axes != bkg.axes:
        raise ValueError("Source and projected background axes do not match.")
    if getattr(source.axes["PsiChi"].coordsys, "name", None) != "galactic":
        raise ValueError("This workflow requires a Galactic PsiChi axis.")

    event = source.copy()
    if not data_contains_background:
        event += bkg

    return source, bkg, event


def select_orientation_for_pointing_cut(
    orientation: str | Path | SpacecraftHistory,
    source_coord: SkyCoord,
    fov_cut: u.Quantity | None,
) -> SpacecraftHistory:
    """Apply the same point-source field-of-view selection used for the data."""

    history = (
        SpacecraftHistory.open(orientation)
        if isinstance(orientation, (str, Path))
        else orientation
    )
    if fov_cut is None:
        return history

    gti = GoodTimeInterval.from_pointing_cut(
        source_coord,
        history,
        fov_cut,
        earth_occ=False,
    )
    return history.apply_gti(gti)


def build_galactic_response(
    detector_response: str | Path,
    orientation: SpacecraftHistory,
    output_path: str | Path,
    *,
    nside_image: int = 2,
    nside_scatt_map: int = 4,
    earth_occ: bool = True,
    dtype=np.float32,
    overwrite: bool = False,
) -> ExtendedSourceResponse:
    """Build or open an orientation-convolved Galactic imaging response.

    Point-source response rows are cached individually next to ``output_path``.
    This makes the expensive response construction resumable.
    """

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return ExtendedSourceResponse.open(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output_path.parent / f"{output_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_base = parts_dir / "psr_"
    image_grid = HealpixBase(
        nside=nside_image,
        coordsys="galactic",
        scheme="ring",
    )

    with FullDetectorResponse.open(detector_response, dtype=dtype) as detector:
        for pixel in tqdm(range(image_grid.npix), desc="Galactic response pixels"):
            part_path = parts_dir / f"psr_{pixel:05d}.hdf5"
            if part_path.exists() and not overwrite:
                continue
            point_response = detector.get_point_source_response_per_image_pixel(
                pixel,
                orientation,
                coordsys="galactic",
                nside_image=nside_image,
                nside_scatt_map=nside_scatt_map,
                earth_occ=earth_occ,
            )
            point_response.write(part_path, overwrite=True)

        galactic_response = FullDetectorResponse.merge_psr_to_extended_source_response(
            part_base,
            coordsys="galactic",
            nside_image=nside_image,
        )

    galactic_response.write(output_path, overwrite=True)
    return galactic_response


def differential_flux_from_model(model: AllSkyImageModel):
    """Integrate reconstructed maps and divide by incident-energy widths."""

    return model.total_flux() / model.axes["Ei"].widths


def run_richardson_lucy_spectral_deconvolution(
    event: Histogram,
    background: Histogram,
    response: ExtendedSourceResponse,
    *,
    iteration_max: int = 20,
    initial_flux: float = 1e-4,
    acceleration_max: float = 5.0,
    response_weighting_index: float = 0.5,
    smoothing_fwhm: u.Quantity = 3 * u.deg,
    background_range: tuple[float, float] = (0.01, 10.0),
    optimize_background: bool = True,
    stopping_threshold: float = 0.01,
):
    """Reconstruct sky maps in every incident-energy bin with accelerated RL."""

    dataset = DataIF_COSI_DC2.load(
        name="spectral_deconvolution",
        event_binned_data=event,
        dict_bkg_binned_data={"background": background},
        rsp=response,
        coordsys_conv_matrix=None,
    )
    model = AllSkyImageModel(
        nside=response.axes["NuLambda"].nside,
        energy_edges=response.axes["Ei"].edges,
        scheme=response.axes["NuLambda"].scheme,
        coordsys="galactic",
    )
    model[:] = initial_flux * model.unit

    parameters = Configurator(
        {
            "iteration_max": int(iteration_max),
            "minimum_flux": {
                "value": 0.0,
                "unit": "cm-2 s-1 sr-1",
            },
            "acceleration": {
                "activate": True,
                "algorithm": "MaxStep",
                "accel_factor_max": float(acceleration_max),
                "accel_bkg_norm": False,
            },
            "response_weighting": {
                "activate": True,
                "index": float(response_weighting_index),
            },
            "smoothing": {
                "activate": True,
                "FWHM": {
                    "value": smoothing_fwhm.to_value(u.deg),
                    "unit": "deg",
                },
            },
            "stopping_criteria": {
                "statistics": "log-likelihood",
                "threshold": float(stopping_threshold),
            },
            "background_normalization_optimization": {
                "activate": bool(optimize_background),
                "range": {"background": list(background_range)},
            },
            "save_results": {"activate": False},
        }
    )

    algorithm = RichardsonLucyAdvanced(
        initial_model=model,
        dataset=DataInterfaceCollection([dataset]),
        mask=None,
        parameter=parameters,
    )
    algorithm.initialization()
    for _ in tqdm(range(iteration_max), desc="Richardson-Lucy iterations"):
        if algorithm.iteration():
            break
    algorithm.finalization()
    return algorithm
