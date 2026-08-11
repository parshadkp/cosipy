#!/usr/bin/env python
"""Run Richardson--Lucy sky/spectrum reconstruction for NGC 4151 DC4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord

from cosipy.image_deconvolution import (
    build_galactic_response,
    differential_flux_from_model,
    prepare_galactic_histograms,
    run_richardson_lucy_spectral_deconvolution,
    select_orientation_for_pointing_cut,
)


DEFAULT_DATA = (
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/COSI/"
    "Radio_Quiet_AGN/GammaRay/Paper_Models/"
    "NGC4151_ec_1000_DC4_COSI_cpl_60_fovCut.hdf5"
)
DEFAULT_BACKGROUND = (
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/COSI/"
    "Radio_Quiet_AGN/DC4_Files/Background/"
    "Total_DC4_BG_3months_binned_data_filtered_with_SAAcut_withSAAbck_"
    "NGC4151_60deg_fov_cut.hdf5"
)
DEFAULT_RESPONSE = (
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/COSI/"
    "Radio_Quiet_AGN/DC4_Files/"
    "ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered."
    "nonsparse.binnedimaging.imagingresponse.h5"
)
DEFAULT_ORIENTATION = (
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/COSI/"
    "Radio_Quiet_AGN/DC4_Files/"
    "DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits"
)
DEFAULT_LARGE_DATA_DIR = (
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/COSI/"
    "Radio_Quiet_AGN/SED-analysis"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--background", default=DEFAULT_BACKGROUND)
    parser.add_argument("--response", default=DEFAULT_RESPONSE)
    parser.add_argument("--orientation", default=DEFAULT_ORIENTATION)
    parser.add_argument("--output-dir", default="outputs/ngc4151_rl_deconvolution")
    parser.add_argument("--large-data-dir", default=DEFAULT_LARGE_DATA_DIR)
    parser.add_argument("--nside-image", type=int, default=2)
    parser.add_argument("--nside-scatt-map", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--initial-flux", type=float, default=1e-4)
    parser.add_argument("--acceleration-max", type=float, default=5.0)
    parser.add_argument("--response-weighting-index", type=float, default=0.5)
    parser.add_argument("--smoothing-fwhm-deg", type=float, default=3.0)
    parser.add_argument("--stopping-threshold", type=float, default=0.01)
    parser.add_argument("--truth-normalization", type=float)
    parser.add_argument("--truth-index", type=float)
    parser.add_argument("--truth-cutoff-kev", type=float)
    parser.add_argument("--overwrite-response", action="store_true")
    parser.add_argument("--data-contains-background", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    large_data_dir = Path(args.large_data_dir)
    large_data_dir.mkdir(parents=True, exist_ok=True)
    source_coord = SkyCoord(l=155.077, b=75.063, frame="galactic", unit="deg")

    source, background, event = prepare_galactic_histograms(
        args.data,
        args.background,
        data_contains_background=args.data_contains_background,
    )
    orientation = select_orientation_for_pointing_cut(
        args.orientation,
        source_coord,
        60 * u.deg,
    )
    response_path = large_data_dir / (
        f"ngc4151_galactic_response_nside{args.nside_image}_"
        f"scatt{args.nside_scatt_map}.hdf5"
    )
    response = build_galactic_response(
        args.response,
        orientation,
        response_path,
        nside_image=args.nside_image,
        nside_scatt_map=args.nside_scatt_map,
        overwrite=args.overwrite_response,
    )

    try:
        algorithm = run_richardson_lucy_spectral_deconvolution(
            event,
            background,
            response,
            iteration_max=args.iterations,
            initial_flux=args.initial_flux,
            acceleration_max=args.acceleration_max,
            response_weighting_index=args.response_weighting_index,
            smoothing_fwhm=args.smoothing_fwhm_deg * u.deg,
            stopping_threshold=args.stopping_threshold,
        )
        model = algorithm.results[-1]["model"]
        model_path = output_dir / "reconstructed_model.hdf5"
        model.write(model_path, overwrite=True)

        differential_flux = differential_flux_from_model(model)
        edges = model.axes["Ei"].edges.to_value(u.keV)
        centers = model.axes["Ei"].centers.to_value(u.keV)
        flux = differential_flux.to_value(1 / (u.cm**2 * u.s * u.keV))
        spectrum_path = output_dir / "reconstructed_spectrum.csv"
        with spectrum_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["e_min_keV", "e_max_keV", "e_ref_keV", "dnde_per_cm2_s_keV"])
            writer.writerows(zip(edges[:-1], edges[1:], centers, flux))

        likelihood = [float(np.sum(item["log-likelihood"])) for item in algorithm.results]
        background_norm = [
            float(item["background_normalization"]["background"])
            for item in algorithm.results
        ]
        diagnostics = {
            "iterations": len(algorithm.results),
            "selected_livetime_s": orientation.cumulative_livetime().to_value(u.s),
            "source_counts": float(source.to_dense(copy=False).contents.sum()),
            "background_counts": float(background.to_dense(copy=False).contents.sum()),
            "log_likelihood": likelihood,
            "background_normalization": background_norm,
            "nside_image": args.nside_image,
            "nside_scatt_map": args.nside_scatt_map,
        }
        diagnostics_path = output_dir / "diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].plot(range(1, len(likelihood) + 1), likelihood)
        axes[0].set(xlabel="Iteration", ylabel="Poisson log-likelihood")
        axes[1].plot(range(1, len(background_norm) + 1), background_norm)
        axes[1].set(xlabel="Iteration", ylabel="Background normalization")
        figure.savefig(output_dir / "convergence.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        xerr = np.vstack((centers - edges[:-1], edges[1:] - centers))
        axis.errorbar(centers, flux, xerr=xerr, fmt="o", label="RL reconstructed")
        truth_parameters = (
            args.truth_normalization,
            args.truth_index,
            args.truth_cutoff_kev,
        )
        if all(value is not None for value in truth_parameters):
            truth_energy = np.geomspace(edges[0], edges[-1], 300)
            truth_flux = (
                args.truth_normalization
                * truth_energy**args.truth_index
                * np.exp(-truth_energy / args.truth_cutoff_kev)
            )
            axis.plot(truth_energy, truth_flux, label="Input cutoff power law")
        axis.set(xscale="log", yscale="log", xlabel="Energy (keV)")
        axis.set_ylabel(r"$dN/dE$ (cm$^{-2}$ s$^{-1}$ keV$^{-1}$)")
        axis.legend()
        figure.savefig(output_dir / "reconstructed_spectrum.png", dpi=180)
        plt.close(figure)

        figure, axes = plt.subplots(2, 5, figsize=(20, 8))
        for energy_index in range(model.axes["Ei"].nbins):
            plt.axes(axes.flat[energy_index])
            hp.mollview(
                model.contents[:, energy_index].value,
                title=f"{edges[energy_index]:g}-{edges[energy_index + 1]:g} keV",
                unit=str(model.unit),
                hold=True,
            )
            hp.projscatter(
                source_coord.galactic.l.deg,
                source_coord.galactic.b.deg,
                lonlat=True,
                color="red",
                marker="*",
            )
        figure.savefig(output_dir / "reconstructed_maps.png", dpi=150)
        plt.close(figure)

        print(f"Response: {response_path}")
        print(f"Model: {model_path}")
        print(f"Spectrum: {spectrum_path}")
        print(f"Diagnostics: {diagnostics_path}")
    finally:
        close_response = getattr(response, "close", None)
        if close_response is not None:
            close_response()


if __name__ == "__main__":
    main()
