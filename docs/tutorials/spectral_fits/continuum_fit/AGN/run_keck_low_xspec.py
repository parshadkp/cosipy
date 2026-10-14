"""Reproducible XSPEC simulation of the Keck low interval of NGC 4151.

Run ``python run_keck_low_xspec.py --stage all``. Each stage can be resumed
independently. Existing science products are never overwritten. The two
reflection/continuum flux targets refer to separate components, not to the
total model flux. This run intentionally has no instrumental backgrounds.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


SIMULATIONS = Path(
    "/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/"
    "COSI/Radio_Quiet_AGN/XRay/Simulations"
)
DEFAULT_RUN = SIMULATIONS / "Ecut1000_KeckLow_XMM25_NuSTAR30"
MODEL_GRID = "energies extend high 1000 500 log"
SOURCE_MODEL = Path("/Users/parshadkp/Software/XSPEC_models/ngc4151_keck_noWA_fluxmatched.xcm")
RELXILL = Path("/Users/parshadkp/Software/XSPEC_models/relxill")
TARGET_PLC = 10 ** -10.22
TARGET_IDR = 10 ** -9.86
TRUTH_FILE = Path("Fits/ngc4151_truth_Ecut1000_KeckLow.xcm")
FIT_NAMES = ("xmm", "xn", "cosi", "joint")
CONTOUR_LEVELS = (2.30, 4.605, 9.210)
SEEDS = {"xmm": 4151001, "nustar": 4151002, "cosi": 4151003}
INSTRUMENTS = {
    "xmm": {
        "rmf": SIMULATIONS / "XMM/pn-thin-5-ao26.rmf",
        "arf": SIMULATIONS / "XMM/pn-thin-5-ao26.arf",
        "exposure": 25000,
        "fake": "ngc4151_xmm_25ks_KeckLow.fak",
        "ignore": "1-500,2000-4096",
    },
    "nustar": {
        "rmf": SIMULATIONS / "NuSTAR/point_sources_for_proposers_v3/nustar.rmf",
        "arf": SIMULATIONS / "NuSTAR/point_sources_for_proposers_v3/point_60arcsecRad_1arcminOA.arf",
        "exposure": 30000,
        "fake": "ngc4151_nustar_30ks_KeckLow.fak",
        "ignore": "1-35,1935-4096",
    },
    "cosi": {
        "rmf": SIMULATIONS / "COSI/cosi_ngc4151.rmf",
        "arf": SIMULATIONS / "COSI/cosi_ngc4151.arf",
        "exposure": 980415,
        "fake": "ngc4151_cosi_980415s_KeckLow.fak",
        "ignore": "10",
    },
}


def fresh_log(run, name):
    path = run / "Logs" / f"{name}.log"
    index = 2
    while path.exists():
        path = run / "Logs" / f"{name}_retry{index}.log"
        index += 1
    return path


def persist_model_grid(session):
    """XSPEC save all omits the energies command; restore it on reload."""
    contents = session.read_text()
    if MODEL_GRID not in contents.splitlines():
        session.write_text(contents.rstrip("\n") + f"\n{MODEL_GRID}\n")


def xspec(run, name, commands, timeout=3600):
    log_path = fresh_log(run, name)
    script = "\n".join([*commands, "quit", "y", ""])
    env = os.environ.copy()
    env["RELXILL_TABLE_PATH"] = str(RELXILL)
    print(f"XSPEC: {name} (log: {log_path})", flush=True)
    try:
        result = subprocess.run(
            ["xspec", "-"], input=script, text=True, cwd=run, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        log_path.write_bytes(output if isinstance(output, bytes) else output.encode())
        raise RuntimeError(f"XSPEC timed out; inspect {log_path}") from error
    log_path.write_text(result.stdout)
    if result.returncode or "invalid command name" in result.stdout:
        raise RuntimeError(f"XSPEC failed ({result.returncode}); inspect {log_path}")
    return result.stdout


def model_commands(plc_norm, idr_norm):
    return [
        f"lmod relxill {RELXILL}",
        f"@{SOURCE_MODEL}",
        "statistic cstat",
        "newpar 3 0.93",
        "newpar 4 15.7",
        "newpar 13 0.00016",
        "newpar 15 1000",
        "newpar 16 1.73",
        f"newpar 18 {plc_norm:.10g}",
        f"newpar 34 {idr_norm:.10g}",
    ]


def model_flux(run, name, plc_norm, idr_norm, isolate):
    commands = model_commands(plc_norm, idr_norm)
    commands += ["dummyrsp 0.1 10000 3000 log"]
    if isolate == "plc":
        commands += ["newpar 13 1e-20", "newpar 34 1e-20"]
    elif isolate == "idr":
        commands += ["newpar 13 1e-20", "newpar 18 1e-20"]
    commands += ["flux 20 40"]
    output = xspec(run, name, commands, timeout=300)
    matches = re.findall(r"Model Flux\s+\S+ photons \((\S+) ergs/cm\^2/s\)", output)
    if not matches:
        raise RuntimeError(f"No 20-40 keV model flux in {name} log")
    return float(matches[-1])


def ensure_layout(run):
    for directory in ("FakeData", "Fits", "PlotData", "Logs"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    for instrument, config in INSTRUMENTS.items():
        for kind in ("rmf", "arf"):
            target = config[kind]
            if not target.is_file():
                raise FileNotFoundError(target)
            link = run / "FakeData" / f"{instrument}.{kind}"
            if not link.exists() and not link.is_symlink():
                link.symlink_to(os.path.relpath(target, link.parent))
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Unexpected response link: {link}")


def write_run_readme(run, metadata):
    readme = run / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "# NGC 4151 Keck low-interval simulation\n\n"
        "This is a Keck-inspired, **no-warm-absorber** simulation, not a direct "
        "reproduction of Keck et al.'s complete spectral model. It uses "
        "XMM/pn 25 ks, NuSTAR/FPMA 30 ks, and COSI 980,415 s. "
        "No instrumental backgrounds were added; fit contours will be optimistic.\n\n"
        "Before each fake spectrum, fit, contour scan, and observed SED export, "
        f"XSPEC applies `{MODEL_GRID}`. This evaluates the reflection model "
        "beyond the individual response's upper energy boundary before folding "
        "it through the response. Without this extension, the XMM-pn model near "
        "Fe-K/9 keV was artificially below the NuSTAR model.\n\n"
        "The injected model has Gamma = 1.73 and Ecut = 1000 keV. Its separate "
        "20-40 keV direct and inner-reflection component fluxes match Keck "
        "et al. (2015) time interval 4, Table 6. The p34 inner-reflection "
        "normalization is unusually large for this installed relxill version; "
        "use the measured component flux and inspected spectral shape, not "
        "p34 alone, when comparing with the paper.\n\n"
        "**Model-range warning:** The installed relxill package declares a "
        "1000-keV upper Ecut limit for its linked xillver components. The "
        "injected spectrum is at this boundary. Cutoff-contour portions above "
        "1000 keV are exploratory, not validated reflection-model constraints. "
        "Do not quote their upper limits as publication-ready results.\n\n"
        f"- Direct flux: {metadata['plc_flux_20_40']:.6g} erg cm^-2 s^-1; "
        f"p18 = {metadata['plc_norm_p18']:.8g}.\n"
        f"- Inner-reflection flux: {metadata['idr_flux_20_40']:.6g} "
        f"erg cm^-2 s^-1; p34 = {metadata['idr_norm_p34']:.8g}.\n"
        f"- Full-model 20-40 keV flux: {metadata['total_flux_20_40']:.6g} "
        "erg cm^-2 s^-1, including distant reflection.\n\n"
        "`FakeData/` contains the independent Poisson realizations; `Fits/` "
        "contains the four XSPEC sessions; `PlotData/` contains joint SED "
        "and Gamma-Ecut contour inputs and figures; `Logs/` records each XSPEC "
        "batch. Run XSPEC from this directory when reloading the `.xcm` files.\n\n"
        "Reference: https://arxiv.org/pdf/1504.07950\n"
    )


def truth(run):
    metadata_path = run / "truth_metadata.json"
    saved = run / TRUTH_FILE
    if saved.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        persist_model_grid(saved)
        write_run_readme(run, metadata)
        return metadata
    if saved.exists() or metadata_path.exists():
        raise RuntimeError("Partial truth outputs exist; inspect them before resuming")

    plc_guess, idr_guess = 0.02288, 0.00391
    plc_measured = model_flux(run, "plc_calibration", plc_guess, idr_guess, "plc")
    idr_measured = model_flux(run, "idr_calibration", plc_guess, idr_guess, "idr")
    plc_norm = plc_guess * TARGET_PLC / plc_measured
    idr_norm = idr_guess * TARGET_IDR / idr_measured
    plc_checked = model_flux(run, "plc_check", plc_norm, idr_norm, "plc")
    idr_checked = model_flux(run, "idr_check", plc_norm, idr_norm, "idr")
    for target, measured in ((TARGET_PLC, plc_checked), (TARGET_IDR, idr_checked)):
        if abs(measured / target - 1) > 0.01:
            raise RuntimeError("Component 20-40 keV flux differs from Keck target by >1%")

    commands = model_commands(plc_norm, idr_norm)
    commands += [
        f"save all {TRUTH_FILE}",
        "dummyrsp 0.3 10000 3000 log",
        "setplot energy keV",
        "setplot noadd",
        "setplot command wdata PlotData/truth_model_components.qdp",
        "plot eemodel",
    ]
    output = xspec(run, "truth_model", commands, timeout=300)
    if not saved.is_file() or not (run / "PlotData/truth_model_components.qdp").is_file():
        raise RuntimeError("Truth model or validation QDP was not saved")
    persist_model_grid(saved)
    total_flux = model_flux(run, "total_flux_check", plc_norm, idr_norm, "total")
    metadata = {
        "paper": "Keck et al. 2015, time-resolved interval 4, Tables 6-7",
        "model": "Keck-inspired, no warm absorbers; XSPEC relxill 2.9.0",
        "truth_gamma": 1.73,
        "truth_ecut_kev": 1000,
        "partial_covering_nh_1e22": 15.7,
        "covering_fraction": 0.93,
        "distant_reflection_norm": 0.00016,
        "plc_norm_p18": plc_norm,
        "idr_norm_p34": idr_norm,
        "plc_flux_20_40": plc_checked,
        "idr_flux_20_40": idr_checked,
        "total_flux_20_40": total_flux,
        "instrumental_backgrounds": False,
        "model_energy_grid": MODEL_GRID,
        "random_seeds": SEEDS,
        "shape_validation": "Compared isolated components and Fe-K/reflection shape with Keck Figure 5; qualitative pass, but p34 normalization is version-dependent.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    write_run_readme(run, metadata)
    if "***Error" in output:
        raise RuntimeError("Truth model emitted an XSPEC error")
    print(f"Truth: p18={plc_norm:.7g}, p34={idr_norm:.7g}", flush=True)
    return metadata


def simulate(run):
    if not (run / TRUTH_FILE).is_file():
        raise RuntimeError("Run --stage truth first")
    for instrument, config in INSTRUMENTS.items():
        fake = run / "FakeData" / config["fake"]
        if fake.exists():
            continue
        rmf = f"FakeData/{instrument}.rmf"
        arf = f"FakeData/{instrument}.arf"
        commands = [
            f"lmod relxill {RELXILL}", f"@{TRUTH_FILE}",
            MODEL_GRID, "statistic cstat", f"xset seed {SEEDS[instrument]}",
            "fakeit none", rmf, arf, "y", "",
            f"FakeData/{config['fake']}",
            f"{config['exposure']}, 1, {config['exposure']}",
        ]
        output = xspec(run, f"fake_{instrument}", commands, timeout=600)
        if not fake.is_file() or "No background will be applied" not in output:
            raise RuntimeError(f"fakeit failed for {instrument}")
        print(f"Saved {fake}", flush=True)


def fit_commands(run, combination):
    instruments = {
        "xmm": ("xmm",),
        "xn": ("xmm", "nustar"),
        "cosi": ("cosi",),
        "joint": ("xmm", "nustar", "cosi"),
    }[combination]
    data_spec = " ".join(
        f"{number}:{number} FakeData/{INSTRUMENTS[name]['fake']}"
        for number, name in enumerate(instruments, 1)
    )
    commands = [
        f"lmod relxill {RELXILL}", f"@{TRUTH_FILE}",
        f"data {data_spec}",
    ]
    for number, name in enumerate(instruments, 1):
        commands += [
            f"response 1:{number} FakeData/{name}.rmf",
            f"arf 1:{number} FakeData/{name}.arf",
        ]
    commands += [MODEL_GRID, "ignore bad"]
    commands += [
        f"ignore {number}:{INSTRUMENTS[name]['ignore']}"
        for number, name in enumerate(instruments, 1)
    ]
    commands += [
        "statistic cstat", "query yes", "freeze 22", "thaw 15 16 18",
    ]
    for group in range(2, len(instruments) + 1):
        constant = 1 + (group - 1) * 34
        commands += [f"untie {constant}", f"newpar {constant} 1", f"thaw {constant}"]
    return commands


def fit(run):
    for combination in FIT_NAMES:
        saved = run / "Fits" / f"fit_{combination}.xcm"
        summary = run / "Fits" / f"fit_{combination}.json"
        if saved.exists() and summary.exists():
            persist_model_grid(saved)
            continue
        if saved.exists() or summary.exists():
            raise RuntimeError(f"Partial fit outputs exist for {combination}")
        commands = fit_commands(run, combination)
        commands += [
            "fit 200", "tclout param 15", 'puts "BEST_ECUT $xspec_tclout"',
            "tclout param 16", 'puts "BEST_GAMMA $xspec_tclout"',
            "tclout param 18", 'puts "BEST_NORM $xspec_tclout"',
            "tclout stat", 'puts "BEST_CSTAT $xspec_tclout"',
            f"save all Fits/fit_{combination}.xcm",
        ]
        output = xspec(run, f"fit_{combination}", commands, timeout=3600)
        if not saved.is_file():
            raise RuntimeError(f"XSPEC did not save fit {combination}")
        persist_model_grid(saved)
        values = {}
        for key in ("ECUT", "GAMMA", "NORM", "CSTAT"):
            matches = re.findall(rf"BEST_{key}\s+([+\-\d.eE]+)", output)
            if not matches:
                raise RuntimeError(f"Missing {key} result for {combination}")
            values[key.lower()] = float(matches[-1])
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"Non-finite fit result for {combination}")
        summary.write_text(json.dumps(values, indent=2) + "\n")
        print(f"Fit {combination}: Gamma={values['gamma']:.4f}, Ecut={values['ecut']:.1f}", flush=True)


def read_grid(run, prefix):
    folder = run / "PlotData"
    gamma, ecut, delta = (
        np.atleast_1d(np.loadtxt(folder / f"{prefix}_2d_{field}.txt"))
        for field in ("gamma", "ecut", "delc")
    )
    if gamma.size != ecut.size or ecut.size != delta.size or not gamma.size:
        raise ValueError(f"Incomplete XSPEC grid: {prefix}")
    if not all(np.all(np.isfinite(values)) for values in (gamma, ecut, delta)):
        raise ValueError(f"Non-finite XSPEC grid: {prefix}")
    delta -= np.min(delta)
    return gamma, ecut, delta


def edge_minimum(gamma, ecut, delta):
    return min(edge_values(gamma, ecut, delta).values())


def edge_values(gamma, ecut, delta):
    return {
        "gamma_low": float(np.min(delta[gamma == gamma.min()])),
        "gamma_high": float(np.min(delta[gamma == gamma.max()])),
        "ecut_low": float(np.min(delta[ecut == ecut.min()])),
        "ecut_high": float(np.min(delta[ecut == ecut.max()])),
    }


def expand_bounds(bounds, grid):
    elo, ehi, glo, ghi = bounds
    edge = edge_values(*grid)
    if edge["ecut_low"] <= CONTOUR_LEVELS[-1]:
        elo = max(20.0, elo / 1.8)
    if edge["ecut_high"] <= CONTOUR_LEVELS[-1]:
        ehi = min(10000.0, ehi * 1.8)
    if edge["gamma_low"] <= CONTOUR_LEVELS[-1]:
        glo = max(1.3, glo - 0.1)
    if edge["gamma_high"] <= CONTOUR_LEVELS[-1]:
        ghi = min(2.2, ghi + 0.1)
    return elo, ehi, glo, ghi


def publish_grid(run, source, destination):
    if source == destination:
        return
    folder = run / "PlotData"
    for field in ("gamma", "ecut", "delc"):
        current = folder / f"{destination}_2d_{field}.txt"
        if current.exists():
            archived = folder / f"{destination}_narrow_2d_{field}.txt"
            if archived.exists():
                raise RuntimeError(f"Refusing to overwrite archived scan: {archived}")
            current.rename(archived)
        shutil.copyfile(folder / f"{source}_2d_{field}.txt", current)


def scan_once(run, combination, prefix, bounds, steps):
    folder = run / "PlotData"
    outputs = [folder / f"{prefix}_2d_{field}.txt" for field in ("gamma", "ecut", "delc")]
    if all(path.is_file() and path.stat().st_size for path in outputs):
        return read_grid(run, prefix)
    if any(path.exists() for path in outputs):
        raise RuntimeError(f"Partial grid exists: {prefix}")
    elo, ehi, glo, ghi = bounds
    ecut_steps, gamma_steps = (steps, steps) if isinstance(steps, int) else steps
    commands = [
        f"lmod relxill {RELXILL}", f"@Fits/fit_{combination}.xcm",
        MODEL_GRID, "statistic cstat", "query yes", "fit 100",
        f"steppar best log 15 {elo:.9g} {ehi:.9g} {ecut_steps} nolog 16 {glo:.9g} {ghi:.9g} {gamma_steps}",
    ]
    for field, tcl_name in (("gamma", "16"), ("ecut", "15"), ("delc", "delstat")):
        commands += [
            f"tclout steppar {tcl_name}",
            f'set outfile [open "PlotData/{prefix}_2d_{field}.txt" w]',
            "puts $outfile $xspec_tclout",
            "close $outfile",
        ]
    xspec(run, f"scan_{prefix}", commands, timeout=14400)
    return read_grid(run, prefix)


def scan(run):
    for combination in ("xn", "cosi", "joint"):
        if not (run / "Fits" / f"fit_{combination}.xcm").is_file():
            raise RuntimeError(f"Run --stage fit before scanning {combination}")
        if combination == "xn":
            refine_xn(run)
            continue
        status_path = run / "PlotData" / f"{combination}_grid_status.json"
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if not previous["open_contours"]:
                print(f"Contour {combination}: already complete", flush=True)
                continue
        bounds = (100.0, 5000.0, 1.55, 1.95)
        coarse = scan_once(run, combination, f"{combination}_coarse", bounds, 12)
        for retry in range(2):
            if edge_minimum(*coarse) > CONTOUR_LEVELS[-1]:
                break
            bounds = expand_bounds(bounds, coarse)
            coarse = scan_once(run, combination, f"{combination}_coarse_wide{retry+1}", bounds, 12)
        gamma, ecut, delta = coarse
        if edge_minimum(*coarse) > CONTOUR_LEVELS[-1]:
            inside = delta <= CONTOUR_LEVELS[-1]
            log_edges = np.log(ecut[inside])
            elog_pad = max(0.1, (log_edges.max() - log_edges.min()) * 0.25)
            gamma_pad = max(0.01, (gamma[inside].max() - gamma[inside].min()) * 0.25)
            bounds = (
                max(20.0, math.exp(log_edges.min() - elog_pad)),
                min(10000.0, math.exp(log_edges.max() + elog_pad)),
                max(1.3, gamma[inside].min() - gamma_pad),
                min(2.2, gamma[inside].max() + gamma_pad),
            )
        best = json.loads((run / "Fits" / f"fit_{combination}.json").read_text())
        bounds = (
            min(bounds[0], best["ecut"] / 1.2),
            max(bounds[1], best["ecut"] * 1.2),
            min(bounds[2], best["gamma"] - 0.03),
            max(bounds[3], best["gamma"] + 0.03),
        )
        # An open 99% surface is already limited by the available parameter
        # range; a denser broad grid cannot make it a closed measurement.
        # Reserve the finer grid for contours that are demonstrably enclosed.
        final_steps = 18 if edge_minimum(*coarse) <= CONTOUR_LEVELS[-1] else 30
        final = scan_once(run, combination, combination, bounds, final_steps)
        source = combination
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            bounds = (
                *previous["bounds"]["ecut_kev"],
                *previous["bounds"]["gamma"],
            )
        for retry in range(1, 4):
            if edge_minimum(*final) > CONTOUR_LEVELS[-1]:
                break
            wider = expand_bounds(bounds, final)
            if wider == bounds:
                break
            bounds = wider
            source = f"{combination}_expanded{retry}"
            final = scan_once(run, combination, source, bounds, 30)
        publish_grid(run, source, combination)
        open_levels = [
            label for label, level in zip(("68%", "90%", "99%"), CONTOUR_LEVELS)
            if edge_minimum(*final) <= level
        ]
        status = {
            "bounds": {"ecut_kev": bounds[:2], "gamma": bounds[2:]},
            "grid_points": int(final[0].size),
            "edge_min_delta_c": float(edge_minimum(*final)),
            "open_contours": open_levels,
            "backgrounds_included": False,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"Contour {combination}: open at edge = {open_levels}", flush=True)


def refine_xn(run):
    """Publish a resolved X-ray grid only when it reaches the fitted minimum."""
    status_path = run / "PlotData/xn_grid_status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("raw_min_delta_c", math.inf) <= 1.0 and all(
            (run / "PlotData" / f"xn_2d_{field}.txt").is_file()
            for field in ("gamma", "ecut", "delc")
        ):
            print("Refined XMM+NuSTAR contour: already complete", flush=True)
            return
    prefix = "xn_focused"
    bounds = (300.0, 5000.0, 1.68, 1.78)
    gamma, ecut, delta = scan_once(run, "xn", prefix, bounds, (30, 40))
    raw_min = float(np.min(np.loadtxt(run / "PlotData" / f"{prefix}_2d_delc.txt")))
    if raw_min > 1.0:
        raise RuntimeError(f"Refined XMM+NuSTAR scan misses fit minimum: raw delta C={raw_min:.3f}")
    source = prefix
    if edge_minimum(gamma, ecut, delta) <= CONTOUR_LEVELS[-1]:
        bounds = (300.0, 10000.0, 1.68, 1.78)
        source = "xn_focused_wide"
        gamma, ecut, delta = scan_once(run, "xn", source, bounds, (36, 40))
        raw_min = float(np.min(np.loadtxt(run / "PlotData" / f"{source}_2d_delc.txt")))
        if raw_min > 1.0:
            raise RuntimeError(f"Wider XMM+NuSTAR scan misses fit minimum: raw delta C={raw_min:.3f}")
    folder = run / "PlotData"
    for field in ("gamma", "ecut", "delc"):
        current = folder / f"xn_2d_{field}.txt"
        archived = folder / f"xn_coarse_missed_min_2d_{field}.txt"
        if current.exists():
            if archived.exists():
                raise RuntimeError(f"Refusing to overwrite archived scan: {archived}")
            current.rename(archived)
        shutil.copyfile(folder / f"{source}_2d_{field}.txt", current)
    open_levels = [
        label for label, level in zip(("68%", "90%", "99%"), CONTOUR_LEVELS)
        if edge_minimum(gamma, ecut, delta) <= level
    ]
    status = {
        "bounds": {"ecut_kev": bounds[:2], "gamma": bounds[2:]},
        "grid_points": int(gamma.size), "raw_min_delta_c": raw_min,
        "edge_min_delta_c": float(edge_minimum(gamma, ecut, delta)),
        "open_contours": open_levels, "backgrounds_included": False,
    }
    (folder / "xn_grid_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Refined XMM+NuSTAR contour: raw min={raw_min:.3f}; open={open_levels}", flush=True)


def refine_cosi(run):
    """Calculate a very dense, independent COSI grid for publication plotting."""
    if not (run / "Fits/fit_cosi.xcm").is_file():
        raise RuntimeError("Run --stage fit before refining the COSI contour")
    # The bounds retain margin outside the earlier 99% contour while putting
    # substantially more samples across the displayed confidence regions.
    bounds = (720.0, 1450.0, 1.38, 2.04)
    prefix = "cosi_hyperfine"
    gamma, ecut, delta = scan_once(run, "cosi", prefix, bounds, (440, 360))
    raw_min = float(np.min(np.loadtxt(run / f"PlotData/{prefix}_2d_delc.txt")))
    edge = edge_minimum(gamma, ecut, delta)
    if raw_min > 1.0:
        raise RuntimeError(f"Fine COSI scan misses fit minimum: raw delta C={raw_min:.3f}")
    status = {
        "bounds": {"ecut_kev": bounds[:2], "gamma": bounds[2:]},
        "grid_points": int(gamma.size), "raw_min_delta_c": raw_min,
        "edge_min_delta_c": float(edge),
        "open_contours": [
            label for label, level in zip(("68%", "90%", "99%"), CONTOUR_LEVELS)
            if edge <= level
        ],
        "backgrounds_included": False,
    }
    (run / f"PlotData/{prefix}_grid_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Hyper-fine COSI contour: raw min={raw_min:.3f}; open={status['open_contours']}", flush=True)


def export(run):
    sed = run / "PlotData/joint_sed.qdp"
    curve = run / "PlotData/model_xmm_nustar_cosi.qdp"
    rebinned = run / "PlotData/joint_sed_rebinned.qdp"
    light_rebin = run / "PlotData/joint_sed_light_rebin.qdp"
    if sed.exists() and curve.exists():
        if not rebinned.exists():
            commands = [
                f"lmod relxill {RELXILL}", "@Fits/fit_joint.xcm",
                MODEL_GRID, "setplot energy keV", "setplot noadd",
                "setplot rebin 40 20 1", "setplot rebin 30 75 2",
                "setplot rebin 3 2 3",
                "setplot command wdata PlotData/joint_sed_rebinned.qdp",
                "plot eeufspec",
            ]
            xspec(run, "export_joint_sed_rebinned", commands, timeout=600)
        if not rebinned.is_file():
            raise RuntimeError("XSPEC did not produce rebinned joint SED QDP")
        if not light_rebin.exists():
            commands = [
                f"lmod relxill {RELXILL}", "@Fits/fit_joint.xcm",
                MODEL_GRID, "setplot energy keV", "setplot noadd",
                "setplot rebin 10 8 1", "setplot rebin 10 25 2",
                "setplot rebin 3 2 3",
                "setplot command wdata PlotData/joint_sed_light_rebin.qdp",
                "plot eeufspec",
            ]
            xspec(run, "export_joint_sed_light_rebin", commands, timeout=600)
        if not light_rebin.is_file():
            raise RuntimeError("XSPEC did not produce lightly rebinned joint SED QDP")
        return
    if sed.exists() or curve.exists():
        raise RuntimeError("Partial QDP export exists; inspect before resuming")
    commands = [
        f"lmod relxill {RELXILL}", "@Fits/fit_joint.xcm",
        MODEL_GRID, "setplot energy keV", "setplot noadd",
        "setplot command wdata PlotData/joint_sed.qdp",
        "plot eeufspec", "setplot delete 1",
        "data none", "dummyrsp 0.3 10000 3000 log",
        "setplot command wdata PlotData/model_xmm_nustar_cosi.qdp",
        "plot eemodel",
    ]
    xspec(run, "export_joint_sed", commands, timeout=600)
    if not sed.is_file() or not curve.is_file():
        raise RuntimeError("XSPEC did not produce joint SED and model QDP files")
    export(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("truth", "simulate", "fit", "scan", "refine-xn", "refine-cosi", "export", "all"), default="all")
    args = parser.parse_args()
    run = args.simulation_dir.expanduser().resolve()
    ensure_layout(run)
    if args.stage in ("truth", "all"):
        truth(run)
    if args.stage in ("simulate", "all"):
        simulate(run)
    if args.stage in ("fit", "all"):
        fit(run)
    if args.stage in ("scan", "all"):
        scan(run)
    if args.stage == "refine-xn":
        refine_xn(run)
    if args.stage == "refine-cosi":
        refine_cosi(run)
    if args.stage in ("export", "all"):
        export(run)
    print(f"Completed stage {args.stage} in {run}", flush=True)


if __name__ == "__main__":
    main()
