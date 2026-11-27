# DC4 source catalog

## Branches

* **`DC4-SourceCat`** contains the COSIpy model/integration changes and independent
  runtime tests. It contains no catalog analysis notebooks.
* **`SourceCatTest`** builds on it and adds both the fitting and catalog creator
  notebooks, notebook tests, simulation comparisons and creation-only code under `dc4_catalog_tools/`. Its `megalib.py`
  converts spatial templates during creation and is never imported during fitting.

The existing `DC4-SourceCat-TEST` branch remains unchanged as a historical reference.
`DC5-SourceCat` is unchanged too.

## Fit with the catalog

Switch to `SourceCatTest` and open `DC4_Static_Source_Catalog_Spectral_Fit.ipynb`.
All analysis functions are
written directly in commented cells: no catalog/fitting helper imports and no
notebook-to-notebook execution. It uses standard COSIpy responses,
`PoissonLikelihood`, `ThreeMLPluginInterface` and threeML Minuit.

Both branches use the same external `DC4_ROOT/dc4_source_catalog/` folder:
ordinary `dc4_catalog.yaml` and 36 physical model FITS files. Its current location is
`/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/COSI/Radio_Quiet_AGN/DC4_Files/dc4_source_catalog`.
The files are no longer tracked or duplicated in either branch. Set `DC4_ROOT`
and `CATALOG_DIRECTORY` in the fitting notebook for another installation.
Also configure the usual observation, instrumental
background, spacecraft history and detector response. Fitting needs no light
curves, `.source`, `.dat`, JSON recipe or `cosi-sim` checkout. The model folder
can be hosted elsewhere later without changing its internal relative references.

The default fit uses **all three months**, retaining existing quality cuts and
orientation livetime, with **no additional pointing or GTI cut**. NGC 4151's CPL
normalization, index (bounds -5 to +5) and cutoff are free, together with one
instrumental-background rate. An optional list frees one shared amplitude per
additional source while retaining component ratios. The independent deterministic
regression uses a small response fixture to recover a known CPL; it is not a
scientific fit to a three-month DC4 observation.

Large extended detector responses and numerical results go in `work/`, outside
the model folder. The first full NSIDE-8 detector response can occupy about 14 GB.
Its cache key includes the actual selected history and response configuration.
A response from an unrelated target GTI is never silently reused.

## Simple spectral FITS

Each spectral file is just a `SPECTRAL_MODEL` binary table with double-precision
`ENERGY` in keV and `FLUX` in ph cm^-2 s^-1 keV^-1. These are physical source
models, not observed counts. Spectral files have no custom time, hash, provenance,
conversion-version or interpolation headers.

FITS necessarily has structural headers describing its layout. Column names and
units make the arrays interpretable; see the
[HEASARC FITS overview](https://heasarc.gsfc.nasa.gov/docs/heasarc/fits_overview.html).
Interpolation (`LINLIN` or `LOGLOG`) is a normal `SpecFromFits` property in the
YAML. `K=1` reproduces the saved spectrum; values outside its support are zero.
The interpolation rule must be retained somewhere because it changes the shape
between samples. No hidden normalization or light curve is applied.

Spatial/energy FITS retain their required HEALPix interpretation: Galactic RING,
NSIDE 8, `SKYMAP` and `ENERGIES`, differential intensity per MeV per sr. NSIDE 64
is used only internally during creation. These COSI models are not XSPEC `atable`
or OGIP PHA products. An eventual HEASARC release can add the keywords required
by its agreed product format.

Production code changes are limited to `SpecFromFits` and selective incident-energy
integration changes in `custom_functions.py`, `functions.py`, `integrals.py` and
`functions_3d.py`. Native sample integration preserves narrow features inside
broad response bins. The likelihood and `BinnedThreeMLModelFolding` are unchanged.

## Static averages and source population

The catalog has **51 frozen entries: 32 point and 19 extended sources**. It removes
18 prompt GRB/MGF/short-burst entries from the earlier 71-entry manifest and combines
the two states of 4C+21.35 and 3C 454.3 into one mean spectrum per AGN. State spectra
are added in physical flux units rather than averaging indices. Crab keeps four
spectral components; CO nova keeps three separate emission entries.

For a MEGAlib file light curve, `.Flux` is a mean over the complete light-curve
span. Creation normalizes that profile to unit mean over its span, integrates over
the reference observation, then applies that average factor exactly once. Finite
profiles are zero outside their support; repeating profiles include partial cycles.
There is no chunk correction. Simulated counts never set model amplitudes.
Reference times, hashes and normalization diagnostics stay in the creator's output,
rather than model FITS headers.

Static averaging discards flux/pointing correlations. The current legacy DC4 paths
are useful diagnostics but do not establish agreement with corrected simulations.
Use the planned steady/average mock for a controlled static-catalog test. Excluded
prompt events must be absent or handled separately in matching observations.

For a flare, select matching event/background times and spacecraft history,
rebuild responses, replace the target's average spectrum and fit its interval
normalization and shape. This separation of mean models and interval analysis is
consistent with the [Fermi catalog](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/8yr_catalog/)
and [LAT light-curve repository](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/LightCurveRepository/about.html).
A likelihood-derived Fermi mean need not equal a simulation-input arithmetic average.

The example is for total-intensity continuum spectroscopy, using unpolarized
copies with the continuum response. It is not a phase/state-dependent polarization
catalog. Broad incident-energy bins preserve line integrals but do not provide
narrow-line spectral resolution. Covariance bands are local Gaussian approximations;
threeML warns if physical bounds truncate their samples.
