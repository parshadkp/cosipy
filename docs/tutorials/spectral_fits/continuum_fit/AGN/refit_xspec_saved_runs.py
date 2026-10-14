"""Re-fit saved XSPEC sessions in a selected NGC 4151 simulation directory.

Run from any working directory. Existing saved fits are never overwritten.
Use --dry-run to inspect the XSPEC commands before using --execute.
"""

import argparse
import os
from pathlib import Path
import subprocess


DEFAULT_SIMULATION_DIR = Path(
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/"
    "COSI/Radio_Quiet_AGN/XRay/Simulations/Ecut1000_XMM20_NuSTAR25"
)
DEFAULT_RELXILL_DIR = Path("/Users/parshadkp/Software/XSPEC_models/relxill")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-dir", type=Path, default=DEFAULT_SIMULATION_DIR)
    parser.add_argument(
        "--fit-file", type=Path, action="append",
        help="Saved .xcm relative to the run directory; default: every Fits/*.xcm",
    )
    parser.add_argument("--relxill-dir", type=Path, default=DEFAULT_RELXILL_DIR)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print commands only")
    mode.add_argument("--execute", action="store_true", help="Run XSPEC fits")
    args = parser.parse_args()

    run_dir = args.simulation_dir.expanduser().resolve()
    if not (run_dir / "FakeData").is_dir() or not (run_dir / "Fits").is_dir():
        parser.error(f"Not a simulation run with FakeData/ and Fits/: {run_dir}")
    if not args.relxill_dir.is_dir():
        parser.error(f"relxill directory does not exist: {args.relxill_dir}")
    fits = args.fit_file
    if fits is None:
        fits = [
            Path("Fits") / path.name
            for path in sorted((run_dir / "Fits").glob("*.xcm"))
        ]
    if not fits:
        parser.error(f"No Fits/*.xcm found in {run_dir}")

    for fit in fits:
        fit_path = (run_dir / fit).resolve()
        if not fit_path.is_file() or not fit_path.is_relative_to(run_dir):
            parser.error(f"Fit file is missing or outside the run: {fit}")
        output = run_dir / "Fits" / "Refits" / f"{fit_path.stem}_refit.xcm"
        if output.exists():
            parser.error(f"Refusing to overwrite existing refit: {output}")
        commands = "\n".join(
            [
                f"lmod relxill {args.relxill_dir.resolve()}",
                "query yes",
                f"@{fit_path.relative_to(run_dir)}",
                "statistic cstat",
                "fit 1000",
                f"save all {output.relative_to(run_dir)}",
                "quit",
                "y",
                "",
            ]
        )
        if args.dry_run:
            print(f"# {fit_path.relative_to(run_dir)} -> {output.relative_to(run_dir)}")
            print(commands)
            continue

        output.parent.mkdir(exist_ok=True)
        environment = os.environ.copy()
        environment["RELXILL_TABLE_PATH"] = str(args.relxill_dir.resolve())
        result = subprocess.run(
            ["xspec", "-"], input=commands, text=True, cwd=run_dir,
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )
        log_path = output.with_suffix(".log")
        log_path.write_text(result.stdout)
        if result.returncode or not output.is_file():
            raise RuntimeError(f"XSPEC failed for {fit}; inspect {log_path}")
        print(f"Saved {output} and {log_path}")


if __name__ == "__main__":
    main()
