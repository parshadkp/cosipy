"""Portable, static source catalogs: standard astromodels YAML plus FITS models.

Only ``create_dc4_source_catalog`` reads simulation definitions/light curves.
``load_source_catalog`` and the standard COSIpy fitting interfaces need no
simulation library or temporal response recipe.
"""
from array import array
import gzip
import hashlib
from pathlib import Path
import re

import astropy.units as u
from astropy.io import fits
import numpy as np
import yaml

from astromodels import (
    Asymm_Gaussian_on_sphere, ExtendedSource, Gaussian, Gaussian_on_sphere,
    Model, PointSource, Powerlaw, SpectralComponent, LinearPolarization,
)
from astromodels.core.model_parser import ModelParser

from cosipy.threeml.custom_functions import GalpropHealpixModel, SpecFromFits
from .megalib import megalib_map_to_galprop_fits

__all__ = ["load_source_catalog", "write_spectral_template",
           "create_dc4_source_catalog", "time_average_factor",
           ]

CATALOG_FORMAT_VERSION = "1.0"
PHOTON_FLUX_UNIT = u.ph / (u.cm**2 * u.s * u.keV)
ENERGY_RANGE_KEV = (100., 10000.)

# The reference DC4 manifest, excluding GRBs, MGFs and the short magnetar burst.
# State entries are grouped below; they are never folded twice in the catalog.
DC4_SOURCE_DEFINITIONS = {
    'cyg_x1_hard': ('DC3', 'cygX1_hard'),
    'crab': ('DC4', 'Crab'),
    'one_e_1740_compow': ('DC3', '1E1740_compow'),
    'grs_1758_258': ('DC3', 'GRS1758'),
    'cena': ('DC4', 'CenA'),
    'four_c_71p07': ('DC4', '4C71p07'),
    'four_c_21p35_noflare': ('DC3', '4C21p35_noflare'),
    'four_c_21p35_flare': ('DC3', '4C21p35_flare'),
    'maxi_j1820': ('DC3', 'MAXIJ1820'),
    'maxi_j1348': ('DC3', 'MAXIJ1348'),
    'three_c_454p3_low': ('DC4', '3C454p3_low'),
    'three_c_454p3_high': ('DC4', '3C454p3_high'),
    'ngc1068': ('DC4', 'NGC_1068'),
    'ngc4151': ('DC4', 'NGC_4151'),
    'cyg_x3': ('DC4', 'cygX3'),
    'psr_b1259': ('DC3', 'PSRB1259'),
    'one_rxs_j170849': ('DC4', '1RXSJ170849'),
    'generic_magnetar': ('DC4', 'magnetar2'),
    'ls_5039': ('DC3', 'LS5039'),
    'psr_j1846': ('DC3', 'PSRJ1846'),
    'co_nova_continuum': ('DC3', 'nova_co_continuum'),
    'tuc_47': ('DC3', 'Globular_Cluster_Tuc_47'),
    'omega_cen': ('DC3', 'Globular_Cluster_Omega_Cen'),
    'ngc_6397': ('DC3', 'Globular_Cluster_NGC_6397'),
    'ngc_6121': ('DC3', 'Globular_Cluster_NGC_6121'),
    'positrons_26al_line': ('DC3', 'Positrons_from_26Al_line'),
    'positrons_26al_cont': ('DC3', 'Positrons_from_26Al_cont'),
    'positrons_44ti_line': ('DC3', 'Positrons_from_44Ti_line'),
    'positrons_44ti_cont': ('DC3', 'Positrons_from_44Ti_cont'),
    'narrow_bulge_511': ('DC3', 'NB_511'),
    'broad_bulge_511': ('DC3', 'BB_511'),
    'vela_snr_511': ('DC3', 'Vela_SNR_511'),
    'co_nova_511kev': ('DC3', 'nova_co_511keV'),
    'co_nova_478kev': ('DC3', 'nova_co_478keV'),
    'al26_cyg_region': ('DC3', '26Al_Cyg_Region'),
    'al26_ne2001': ('DC3', '26Al_NE2001'),
    'fe60_cyg_region': ('DC3', '60Fe_Cyg_Region'),
    'fe60_ne2001': ('DC3', '60Fe_NE2001'),
    'casa_g16_distribution': ('DC4', 'CasAG16distribution'),
    'g1903': ('DC4', 'G1903'),
    'kepler': ('DC4', 'Kepler'),
    'orion_eridanus': ('DC4', 'OrEr'),
    'positron_annihilation_in_flight': ('DC4', 'Positrons_In_Flight_Annihilation'),
    'positrons_central_source': ('DC4', 'Positrons_Central_Source'),
    'positrons_thin_disk_cont': ('DC4', 'Positrons_Thin_Disk_cont'),
    'positrons_thin_disk_line': ('DC4', 'Positrons_Thin_Disk_line'),
    'sn1987a': ('DC4', 'SN1987A'),
    'tycho': ('DC4', 'Tycho'),
    'upper_scorpius': ('DC4', 'USco'),
    'vela_jr_44ti': ('DC4', 'VelaJr_44Ti'),
    'lmc_511': ('DC3', 'LMC_Gaussian_511'),
    'm31_511': ('DC3', 'M31_Gaussian_511'),
    'virgo_511': ('DC3', 'Virgo_Gaussian_511'),
}
STATE_GROUPS = {
    "four_c_21p35": ("four_c_21p35_noflare", "four_c_21p35_flare"),
    "three_c_454p3": ("three_c_454p3_low", "three_c_454p3_high"),
}


def load_source_catalog(yaml_path):
    """Load ordinary model YAML, resolving FITS paths from its own directory.

    The YAML is neither rewritten nor extended. Returned model file paths are
    absolute in memory, so later changes to the working directory are harmless.
    """
    yaml_path = Path(yaml_path).expanduser().resolve()
    contents = yaml.safe_load(yaml_path.read_text())
    if not isinstance(contents, dict) or not contents:
        raise ValueError("The catalog must be a nonempty astromodels model")

    def resolve(value):
        path = Path(value).expanduser()
        path = path if path.is_absolute() else yaml_path.parent / path
        if not path.is_file():
            raise FileNotFoundError(f"Catalog asset not found: {path}")
        return str(path.resolve())

    def relocate(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "_fitsfile" and value:
                    node[key] = resolve(value)
                elif key == "SpecFromFits":
                    value["fits_file"]["value"] = resolve(value["fits_file"]["value"])
                    relocate(value)
                else:
                    relocate(value)
        elif isinstance(node, list):
            for value in node:
                relocate(value)
    relocate(contents)
    return ModelParser(model_dict=contents).get_model()



def write_spectral_template(path, energy_keV, flux_per_keV, *, overwrite=False):
    """A two-column physical spectrum with only ordinary FITS table headers.

    The interpolation rule belongs to SpecFromFits in the YAML, not this file.
    Reference times, hashes and averaging diagnostics stay in creator output.
    """
    energy, flux = np.asarray(energy_keV, float), np.asarray(flux_per_keV, float)
    if (energy.ndim != 1 or energy.size < 2 or energy.shape != flux.shape
            or np.any(~np.isfinite(energy)) or np.any(energy <= 0)
            or np.any(np.diff(energy) <= 0) or np.any(~np.isfinite(flux))
            or np.any(flux < 0)):
        raise ValueError('Require increasing positive energies and finite nonnegative flux')
    table = fits.BinTableHDU.from_columns([
        fits.Column(name='ENERGY', format='D', unit='keV', array=energy),
        fits.Column(name='FLUX', format='D', unit='ph cm-2 s-1 keV-1', array=flux),
    ], name='SPECTRAL_MODEL')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=overwrite)
    return path


def time_average_factor(times, intensities, start, stop, *, repeating=False):
    """Mean over [start, stop] after normalizing the complete LC to mean one.

    ``Flux`` in a MEGAlib File light curve is already a mean flux over the LC's
    complete span. Normalizing to unit mean respects that convention. A finite
    profile is zero outside its support; a repeating profile includes partial
    periods. No exposure weights or parallel-chunk conventions enter this mean.
    """
    t, y = np.asarray(times, float), np.asarray(intensities, float)
    if (t.ndim != 1 or t.size < 2 or y.shape != t.shape
            or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0)
            or np.any(~np.isfinite(y)) or np.any(y < 0)
            or not np.isfinite(start) or not np.isfinite(stop) or stop <= start):
        raise ValueError("Invalid light curve or reference observation")
    x = t - t[0]
    widths = np.diff(x)
    cumulative = np.r_[0., np.cumsum((y[:-1] + y[1:]) * widths * .5)]
    area, period = cumulative[-1], x[-1]
    if area == 0:
        return 0.

    def primitive(z):
        z = float(np.clip(z, 0., period))
        i = min(np.searchsorted(x, z, side="right") - 1, len(widths) - 1)
        d = z - x[i]
        return cumulative[i] + y[i] * d + .5 * (y[i+1]-y[i]) / widths[i] * d*d

    a, b = start - t[0], stop - t[0]
    if repeating:
        ca, pa = divmod(a, period)
        cb, pb = divmod(b, period)
        integral = (cb - ca) * area + primitive(pb) - primitive(pa)
    else:
        integral = primitive(b) - primitive(a)
    return max(0., integral) / ((stop - start) * area / period)


def _read_dp_table(path, *, spectral=False):
    """Read original numeric DP samples, including gzipped source-library files."""
    opener = gzip.open if Path(path).suffix == ".gz" else open
    x, y = array("d"), array("d")
    interpolation = "LINLIN"
    with opener(path, "rt") as stream:
        for line in stream:
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if fields[0] == "IP":
                interpolation = fields[1].upper()
                if interpolation not in (("LINLIN", "LOGLOG") if spectral else ("LINLIN",)):
                    raise ValueError(f"Unsupported interpolation in {path}: {fields}")
            if fields[0] == "DP":
                x.append(float(fields[1]))
                y.append(float(fields[2]))
    x, y = np.array(x), np.array(y)
    if spectral:
        # MFunction.cxx falls back to LINLIN for the entire table whenever
        # LOGLOG was requested but any intensity is zero (e.g. MAXI cutoffs).
        if interpolation == "LOGLOG" and np.any(y <= 0):
            interpolation = "LINLIN"
        return x, y, interpolation
    return x, y


def _parse_source(path):
    blocks, by_name = [], {}
    for line in path.read_text().splitlines():
        tokens = line.split("#", 1)[0].split()
        if not tokens:
            continue
        if tokens[0] == "DataChallenge.Source":
            block = {"_name": tokens[1]}
            blocks.append(block)
            by_name[tokens[1]] = block
        elif "." in tokens[0]:
            name, field = tokens[0].split(".", 1)
            if name in by_name:
                by_name[name][field] = tokens[1:]
    if not blocks:
        raise ValueError(f"No source blocks in {path}")
    return blocks


def _asset(definition, filename):
    path = definition.parent / filename
    for candidate in (path, Path(str(path) + ".gz")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def _fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_name(value):
    name = re.sub(r"[^a-z0-9_]", "_", value.lower()).strip("_")
    return "s_" + name if name[0].isdigit() else name


def _filename(stem):
    # Keep archive filenames lowercase, recognizable, unique and <=35 characters.
    if len(stem) > 30:
        stem = stem[:25] + "_" + hashlib.sha256(stem.encode()).hexdigest()[:4]
    return stem + ".fits"




def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def create_dc4_source_catalog(source_library, output_directory, start, stop, *,
                              overwrite=False):
    """Create the static 51-entry catalog and all model FITS in one flat folder.

    ``start``/``stop`` are Unix seconds delimiting the reference observation.
    The creator is the only step requiring cosi-sim. Returns (YAML path, rows),
    where rows describe each original component's averaging and input files.
    Existing unrelated files are never deleted. Replacing an existing catalog
    requires ``overwrite=True``.
    """
    from cosipy.response.integrals import get_integral_values, integral_sample_segments
    source_library, output_directory = Path(source_library), Path(output_directory)
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        raise ValueError("Require finite increasing Unix reference times")
    catalog_path = output_directory / "dc4_catalog.yaml"
    if catalog_path.exists() and not overwrite:
        raise FileExistsError(catalog_path)
    definitions, blocks = {}, {}
    for key, (challenge, stem) in DC4_SOURCE_DEFINITIONS.items():
        matches = sorted((source_library / challenge / "sources").rglob(stem + ".source"))
        if len(matches) != 1:
            raise ValueError(f"Expected one {challenge}/{stem}.source; found {matches}")
        definitions[key] = matches[0]
        blocks[key] = _parse_source(matches[0])

    state_members = {member for group in STATE_GROUPS.values() for member in group}
    groups = {k: (k,) for k in definitions if k not in state_members}
    groups.update(STATE_GROUPS)
    output_directory.mkdir(parents=True, exist_ok=True)
    sources, rows = [], []
    profile_cache = {}
    crab_names = {"Crab_Nebula": "nebula", "Crab_P1": "peak1",
                  "Crab_Brg": "bridge", "Crab_P2": "peak2"}

    def temporal_factor(definition, block):
        lc = block.get("Lightcurve")
        if lc is None:
            return 1., None, None
        if len(lc) != 3 or lc[0].lower() != "file" or lc[1].lower() not in ("true", "false"):
            raise ValueError(f"Unsupported light curve {lc} in {definition}")
        path = _asset(definition, lc[2])
        cache_key = (path, lc[1].lower() == "true")
        if cache_key not in profile_cache:
            t, y = _read_dp_table(path)
            factor = time_average_factor(t, y, start, stop, repeating=cache_key[1])
            profile_cache[cache_key] = (factor, path.name, _fingerprint(path))
        return profile_cache[cache_key]

    for key, members in groups.items():
        component_rows = [(definitions[m], b) for m in members for b in blocks[m]]
        first_definition, first = component_rows[0]
        if first["Beam"][0] == "FarFieldNormalizedEnergyBeamFluxFunction":
            if len(component_rows) != 1 or "Lightcurve" in first:
                raise ValueError(f"Unsupported temporal/multiple spatial-energy map: {key}")
            input_map = _asset(first_definition, first["Beam"][1])
            destination = output_directory / _filename(key + "_map")
            megalib_map_to_galprop_fits(input_map, destination, nside=8,
                                      oversample_nside=64, energy_bounds=ENERGY_RANGE_KEV,
                                      overwrite=overwrite)
            spatial = GalpropHealpixModel()
            spatial._fitsfile = str(destination.resolve())
            source = ExtendedSource(key, spatial_shape=spatial)
            source.spectrum.main.Constant.k.unit = 1 / (u.cm**2 * u.s * u.keV)
            sources.append(source)
            rows.append({"source": key, "component": "main", "time factor": 1.,
                         "light curve": "", "model file": destination.name,
                         "source sha256": _fingerprint(first_definition),
                         "light curve sha256": ""})
            continue

        beam = first["Beam"]
        orientation = first["Orientation"]
        if any(b["Beam"] != beam or b["Orientation"] != orientation for _, b in component_rows):
            raise ValueError(f"Multiple positions/morphologies need a grouping rule: {key}")
        latitude, longitude = map(float, orientation[-2:])
        shapes, polarizations, names, tables = [], [], [], []
        for index, (definition, block) in enumerate(component_rows):
            factor, lcname, lchash = temporal_factor(definition, block)
            spec = block["Spectrum"]
            flux = float(block["Flux"][0])
            component = (crab_names[block["_name"]] if key == "crab" else
                         "main" if len(component_rows) == 1 else _safe_name(block["_name"]))
            pol = block.get("Polarization")
            polarization = (LinearPolarization(100. * float(pol[-2]), float(pol[-1]))
                            if pol is not None else None)
            if spec[0].lower() == "file":
                original = _asset(definition, spec[1])
                energy, values, interpolation = _read_dp_table(original, spectral=True)
                if (energy.size < 2 or np.any(np.diff(energy) <= 0)
                        or np.any(~np.isfinite(energy)) or np.any(~np.isfinite(values))
                        or np.any(energy <= 0) or np.any(values < 0)):
                    raise ValueError(f"Invalid spectral samples in {original}")
                # Spectrum File is a probability shape in MEGAlib; .Flux sets
                # its photon integral. Normalize on its complete native support.
                integral = integral_sample_segments(energy, values, interpolation)
                if integral <= 0:
                    raise ValueError(f"Zero spectrum in {original}")
                values = values * (flux * factor / integral)
                # A source outside the reference interval has exactly zero
                # mean emission; zero is representable with linear interpolation.
                if factor == 0.:
                    interpolation = "LINLIN"
                tables.append((energy, values))
                if key in STATE_GROUPS:
                    if interpolation != "LINLIN":
                        raise ValueError("Combined DC4 states require their native LINLIN tables")
                    shape = None  # Written once after summing the two states.
                else:
                    filename = _filename(key + ("_" + component if len(component_rows) > 1 else "") + "_spec")
                    write_spectral_template(output_directory / filename, energy, values,
                                            overwrite=overwrite)
                    shape = SpecFromFits(fits_file=str((output_directory / filename).resolve()),
                                         interpolation=interpolation)
            elif spec[0].lower() == "gaussian":
                shape = Gaussian(F=flux * factor, mu=float(spec[1]), sigma=float(spec[2]))
            elif spec[0].lower() == "powerlaw":
                shape = Powerlaw(K=1., piv=100., index=-float(spec[3]))
                integral = get_integral_values(shape, np.array(spec[1:3], dtype=float))[0]
                shape.K.value *= flux * factor / integral
            else:
                raise ValueError(f"Unsupported retained spectrum in {definition}: {spec}")
            shapes.append(shape)
            polarizations.append(polarization)
            names.append(component)
            rows.append({"source": key, "component": block["_name"],
                         "time factor": factor, "light curve": lcname or "",
                         "model file": "combined spectrum" if key in STATE_GROUPS else
                         filename if spec[0].lower() == "file" else "analytic YAML",
                         "source sha256": _fingerprint(definition),
                         "light curve sha256": lchash or ""})

        if key in STATE_GROUPS:
            if len(tables) != 2 or any(s is not None for s in shapes):
                raise ValueError(f"Expected two tabulated states for {key}")
            signatures = [p.to_dict() if p is not None else None for p in polarizations]
            same_polarization = signatures[1] == signatures[0]
            grid = np.unique(np.concatenate([t[0] for t in tables]))
            # Zero anchors immediately beyond finite support avoid bridging a
            # gap if future input states have different tabulated energy ranges.
            grid = np.unique(np.concatenate(
                [grid] + [np.nextafter(t[0][[0, -1]], [0., np.inf]) for t in tables]
            ))
            combined = sum(np.interp(grid, t[0], t[1], left=0., right=0.) for t in tables)
            filename = _filename(key + "_spec")
            write_spectral_template(output_directory / filename, grid, combined,
                                    overwrite=overwrite)
            shapes = [SpecFromFits(fits_file=str((output_directory / filename).resolve()),
                                   interpolation="LINLIN")]
            # Combining differently polarized states into one intensity table
            # does not define one energy-independent polarization angle/degree.
            # This static product is for continuum spectral analysis.
            names = ["main"]
            polarizations = [polarizations[0] if same_polarization else None]

        if beam[0] == "FarFieldPointSource":
            source = PointSource(key, l=longitude % 360., b=latitude,
                                 components=[SpectralComponent(n, s, p)
                                             for n, s, p in zip(names, shapes, polarizations)])
        else:
            if beam[0] == "FarFieldGaussian":
                spatial = Gaussian_on_sphere(lon0=longitude % 360., lat0=latitude,
                                             sigma=float(beam[-1]))
            elif beam[0] == "FarFieldAssymetricGaussian":
                a, b = sorted(map(float, beam[3:5]), reverse=True)
                spatial = Asymm_Gaussian_on_sphere(lon0=longitude % 360., lat0=latitude,
                                                  a=a, e=np.sqrt(1. - (b/a)**2),
                                                  theta=float(beam[5]))
            else:
                raise ValueError(f"Unsupported beam: {beam}")
            spectrum = shapes[0]
            for shape in shapes[1:]:
                spectrum = spectrum + shape
            source = ExtendedSource(key, spatial_shape=spatial, spectral_shape=spectrum,
                                    polarization=polarizations[0])
        sources.append(source)

    model = Model(*sources)
    for parameter in model.parameters.values():
        parameter.fix = True
    if (len(model.point_sources), len(model.extended_sources)) != (32, 19):
        raise RuntimeError("Unexpected DC4 catalog population")
    contents = _plain(model.to_dict_with_types())

    def relative_paths(node):
        if isinstance(node, dict):
            for name, value in node.items():
                if name == "_fitsfile":
                    node[name] = Path(value).name
                elif name == "SpecFromFits":
                    value["fits_file"]["value"] = Path(value["fits_file"]["value"]).name
                    relative_paths(value)
                else:
                    relative_paths(value)
        elif isinstance(node, list):
            for value in node:
                relative_paths(value)
    relative_paths(contents)
    # No wrapper schema, metadata block or custom YAML comments.
    catalog_path.write_text(yaml.safe_dump(contents, sort_keys=False))
    reloaded = load_source_catalog(catalog_path)
    if len(reloaded.sources) != 51 or reloaded.free_parameters:
        raise RuntimeError("Catalog round trip failed")
    return catalog_path, rows
