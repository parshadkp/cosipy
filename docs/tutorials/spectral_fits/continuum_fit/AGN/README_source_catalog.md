# Full DC4 catalog fit: NGC 4151

Open `NGC4151_DC4_Spectral_Fit_Source_Cat_lightcurve.ipynb` in a fresh kernel
using this cosipy checkout. Configure the data root and `cosi-sim` source-library
root in the first setup cell, then run top to bottom.

## Inputs and provenance

The full-observation catalog and its `.response.json` recipe were copied from
`DC4-analysis` commit `9f930c6d51e99f8704d88c37d80ee968f298e81f`:

- `source_catalog_DC4_all_71files_full_elapsed_lightcurve_response_norm.yaml`
- `source_catalog_DC4_all_71files_full_elapsed_lightcurve_response_norm.response.json`

The models, normalizations, temporal factors, and lightcurve hashes are unchanged.
Absolute input paths are relocated in memory, not rewritten in the supplied files.
Use the same source-library assets and detector response as that DC4 simulation.

`Source_catalog_generator_all_DC4_lightcurve.ipynb` is also included from
`DC4-analysis` for regenerating the catalog. It writes the full-observation
YAML and response JSON used by the fit, plus an NGC 4151 GTI catalog and recipe.
The generator requires the 71 individual-source event FITS files in addition to
the source library, orientation, detector response, and `agn.yaml`. Review its
input and output paths before running it; its save cell overwrites catalog files.
Its light-curve integration uses `cosipy.response.temporal_profile.LightcurveProfile`.

For the two full-observation YAML catalogs, open
`Source_catalog_generator_DC4_yaml_only.ipynb` in the `cosipy312` kernel and
run it top to bottom. Its first code cell contains the editable input and
output paths. It reads the source-library tables, orientation, and existing
`_nside8_perMeV_v3.fits` templates. It does not read event FITS files, fold
responses, produce comparison plots or summaries, or write a response recipe.
It saves separate parallel-chunks and no-chunks catalogs, each with a
`_generated.yaml` suffix to preserve the checked-in catalog. Use each catalog
with a response recipe made using the same normalization method. Set
`OVERWRITE_CATALOGS = True` only when intentionally replacing existing outputs.

To compare fits from those two YAMLs, run
`NGC4151_DC4_Spectral_Fit_Compare_Parallel_NoChunks.ipynb` in the `cosipy312` kernel.
It uses the same NGC 4151 GTI, mock data, background, and detector response for
both fits. It derives the no-chunks lightcurve response recipe in memory and
checks its time factors against the no-chunks YAML before fitting. The notebook
shows a fit-parameter table, fitted spectra with covariance-propagated 68%
uncertainty bands, and observed-count residuals. The
cached NSIDE-8 extended response requires about 13 GiB of RAM.

To test whether nuisance-source amplitudes bias the NGC 4151 CPL fit, run
`NGC4151_DC4_Spectral_Fit_Test_Nuisance_Normalizations.ipynb`. It compares a
fixed-catalog baseline with separate free-scale fits for the 4C 21.35 flare,
Crab, and MAXI J1820, plus a fit freeing all three. Multi-component source
ratios stay tied. Select the parallel-chunks or no-chunks YAML at the top; the
matching temporal response is used in either case. The notebook reports fitted
scales, NGC 4151 parameters, likelihood changes, and count residuals.

Required external inputs are the observation and instrumental background,
spacecraft orientation, detector response, and `cosi-sim` spectra/lightcurves/maps.
Individual-source event files and the catalog-generation notebook are not needed.
The default observation has the updated NGC 4151 CPL+PL injection. The plotted
injected reference is only appropriate to that dataset.

## Selection, model, and caching

- The full catalog has 71 simulation entries: 52 point and 19 extended sources.
- The NGC 4151 GTI uses a 60-degree pointing cut with Earth occultation.
- Default data/background HDF5 inputs already have this cut. Native event FITS
  inputs can instead be selected and binned by setting their `*_already_gti_selected`
  flags to `False`. A projected full-observation histogram cannot be re-cut.
- Only NGC 4151 CPL K, index and cutoff, plus background rate, are fitted.
  All other sources remain fixed. The injected comparison remains CPL+PL.
- Temporal responses use `L(t)/mean_full(L)` inside the GTI; no extra amplitude
  scaling is applied to the YAML. Parallel-chunk conventions and limitations
  are inherited from the response recipe, including the inferred nova convention.
- Missing `_nside8_perMeV_v3.fits` spatial templates are generated in
  `DC4_Sources/DC4_Spatial_Templates` and saved. Existing compatible files are reused.
- The ordinary NSIDE-8 GTI extended response is loaded or generated and saved in
  `Extended_Responses/NGC4151_Cut`. It requires about 13 GiB of memory, plus working
  space. Existing legacy caches have no GTI provenance metadata: use only the
  matching NGC 4151 60-degree cache. A new selection/binned response needs a new path.
- Exposure multiplication scales counts and response exposure, not flux parameters.
  It repeats the selected realization; it does not simulate additional independent events.

No adjacent Python helper files are needed. Reusable functionality lives in:

- `cosipy/response/temporal_profile.py`: piecewise-linear lightcurve integration,
  MEGAlib chunk normalization, and plain/gzipped lightcurve reading.
- `cosipy/response/threeml_temporal_response.py`: zero-exposure and component
  response adapters, and interval-weighted point-response construction.
- `cosipy/util/megalib.py`: coupled spatial/energy map conversion to cached
  GALPROP-compatible HEALPix FITS templates.

The notebook contains the DC4-specific source mapping, JSON recipe interpretation,
path relocation, cache setup, and native-event GTI selection. These definition
cells must be run before the configuration and fit cells. Core also includes the
source-specific point-response option in `cosipy/response/threeml_response.py`
and robust DP-row parsing in `cosipy/threeml/custom_functions.py`.

The notebook shows the spectral fit and a count-space projection; it does not
overwrite catalogs, data, background, or plots. Its spatial-template overwrite
switch defaults to False. Inspect covariance warnings and parameter boundaries
before interpreting uncertainty bands.
