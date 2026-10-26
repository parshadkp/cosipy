# DC4 parallel-chunks source catalog

This directory contains one full-observation pair:

- `source_catalog_DC4_parallel_full.yaml`: 71 frozen injection source models.
- `source_catalog_DC4_parallel_full.response.json`: 74 source/component temporal response recipes, including the Crab's distinct components.

Use both files together. The YAML amplitude for a variable component is its injected amplitude multiplied by its **full-observation elapsed-time mean** `f_full`. For a chosen analysis GTI, the fit notebook integrates that component's light curve over each selected orientation interval and weights the interval livetime by `L_interval / f_full`. This makes the source-specific response depend on when the spacecraft pointed toward the source. A different GTI requires new selected data and responses, but **does not require changing the YAML or JSON**.

## Inputs

Install `cosipy` from this branch and make the `cosi-sim/cosi_sim/Source_Library` and DC4 files available locally. Set `COSI_SIM_SOURCE_LIBRARY` and `DC4_FILES_ROOT` in the environment, or edit the root paths near the top of either notebook. The YAML and JSON use `SOURCE_LIBRARY/`, `DC4_FILES/`, and `DC4_TEMPLATES/` tokens; the fit notebook resolves them in memory. It creates missing NSIDE-8 coupled-map templates in `DC4_FILES_ROOT/DC4_Sources/DC4_Spatial_Templates`.

Run `Source_catalog_generator_DC4_parallel_full.ipynb` to see how the pair is made. It writes `_generated.yaml` and `_generated.response.json` so the checked-in pair can be compared without overwriting it. Generation uses source definitions, spectral tables, light curves, and the original spacecraft history, **not event counts**. The catalog assumes 1000 independent chunks for mission-grid light curves and 500 for the CO nova continuum. These counts come from the earlier DC4 analysis notebooks and have not been verified against the archived per-source simulation configurations. The public pipeline supports a configurable chunk count; its example `inputs.yaml` uses 200.

Run `DC4_Parallel_Source_Catalog_Fit.ipynb` to fit. Its default uses the existing NGC 4151 60-degree, Earth-occluded, three-month preselected data with the updated NGC 4151 CPL+PL injection, then fits a CPL only. For another source or GTI, provide native unbinned, time-tagged source and background FITS from an injection matching the catalog. The available ordinary full DC4 mock FITS has an earlier NGC 4151 injection and is not a matching default for this YAML. The new GTI is applied to both event files and the original orientation before constructing responses. A new extended-response cache is keyed by the GTI and response settings; the known NGC 4151 reference cube can be reused for the default mode. The cube can require about 13 GiB of memory.

The fit notebook reports the fitted parameters, available valid 68% uncertainties, log likelihood, and the ten native measured-energy count bins. It does not perform an independent SED fit or create presentation plots.
