"""Apply the NGC 4151 pointing cut and save unbinned event files."""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from cosipy import UnBinnedData
from cosipy.event_selection import GoodTimeInterval
from cosipy.spacecraftfile import SpacecraftHistory


TUTORIAL_DIR = Path(__file__).resolve().parent
CONFIG_FILE = TUTORIAL_DIR / "inputs.yaml"

DATA_FILE = TUTORIAL_DIR / (
    "dc4_mock_dataset_3months_unbinned_data_filtered_with_SAAcut_"
    "time_ordered.fits.gz"
)
BACKGROUND_FILE = TUTORIAL_DIR / (
    "Total_DC4_BG_3months_unbinned_data_filtered_with_SAAcut_"
    "withSAAbck.fits.gz"
)
ORIENTATION_FILE = TUTORIAL_DIR / (
    "DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits"
)

DATA_OUTPUT_PREFIX = TUTORIAL_DIR / (
    "dc4_mock_dataset_3months_unbinned_data_filtered_with_SAAcut_"
    "time_ordered_NGC4151_cut"
)
BACKGROUND_OUTPUT_PREFIX = TUTORIAL_DIR / (
    "Total_DC4_BG_3months_unbinned_data_filtered_with_SAAcut_"
    "withSAAbck_NGC4151_cut"
)
ORIENTATION_OUTPUT_FILE = TUTORIAL_DIR / (
    "DC4_final_530km_3_month_with_slew_15sbins_"
    "GalacticEarth_SAA_NGC4151_cut.fits"
)

SOURCE_NAME = "NGC 4151"
SOURCE_COORD = SkyCoord(l=155.07 * u.deg, b=75.06 * u.deg, frame="galactic")
MAX_OFFAXIS = 60 * u.deg
EARTH_OCC = True


def output_path(prefix: Path, output_format: str) -> Path:
    """Return the filename appended by ``write_unbinned_output``."""
    if output_format == "fits":
        return Path(f"{prefix}.fits.gz")
    if output_format == "hdf5":
        return Path(f"{prefix}.hdf5")
    raise ValueError(f"Unsupported unbinned output format: {output_format}")


def cut_and_write_events(
    input_file: Path,
    output_prefix: Path,
    source_gti: GoodTimeInterval,
    label: str,
) -> None:
    """Apply the pointing-cut GTI and write the selected events unbinned."""
    events = UnBinnedData(str(CONFIG_FILE))
    events.cosi_dataset = events.get_dict(str(input_file))

    in_fov_mask = source_gti.contains(events.cosi_dataset["TimeTags"])
    n_in_fov = np.count_nonzero(in_fov_mask)
    n_out_fov = len(in_fov_mask) - n_in_fov

    if n_in_fov == 0:
        raise RuntimeError(f"The pointing cut selected no {label} events")

    events.cosi_dataset = {
        key: values[in_fov_mask]
        for key, values in events.cosi_dataset.items()
    }
    selected_time_tags = events.cosi_dataset["TimeTags"]
    events.tmin = float(np.min(selected_time_tags))
    events.tmax = float(np.max(selected_time_tags))

    events.write_unbinned_output(str(output_prefix))
    saved_file = output_path(output_prefix, events.unbinned_output)

    print(f"{label} events in FOV: {n_in_fov:,}")
    print(f"{label} events outside FOV: {n_out_fov:,}")
    print(f"Wrote: {saved_file}")


def main() -> None:
    orientation = SpacecraftHistory.open(ORIENTATION_FILE)
    source_gti = GoodTimeInterval.from_pointing_cut(
        SOURCE_COORD,
        orientation,
        MAX_OFFAXIS,
        earth_occ=EARTH_OCC,
    )
    pointing_cut_orientation = orientation.apply_gti(source_gti)

    gti_livetime = pointing_cut_orientation.cumulative_livetime().to_value(u.s)
    total_livetime = orientation.cumulative_livetime().to_value(u.s)
    livetime_fraction = gti_livetime / total_livetime

    print(f"Source: {SOURCE_NAME}")
    print(f"FOV cut: off-axis <= {MAX_OFFAXIS.to_value(u.deg):.1f} deg")
    print(f"GTI intervals: {len(source_gti)}")
    print(f"GTI livetime: {gti_livetime:,.1f} s")
    print(f"Total livetime: {total_livetime:,.1f} s")
    print(f"Livetime fraction in FOV: {livetime_fraction:.4f}")

    cut_and_write_events(DATA_FILE, DATA_OUTPUT_PREFIX, source_gti, "Mock-data")
    cut_and_write_events(
        BACKGROUND_FILE,
        BACKGROUND_OUTPUT_PREFIX,
        source_gti,
        "Background",
    )

    pointing_cut_orientation.write_fits(ORIENTATION_OUTPUT_FILE, overwrite=True)
    print(f"Wrote: {ORIENTATION_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
