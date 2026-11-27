# SourceCatTest

This branch adds the creation, comparison and self-contained fitting notebooks to
`DC4-SourceCat`. Runtime COSIpy changes and independent unit tests are inherited;
both analysis notebooks and their notebook-specific tests exist only here.

Open `DC4_Static_Source_Catalog_Creator.ipynb` and set `SOURCE_LIBRARY`, `DC4_ROOT`
and `ORIENTATION`. Run creation to write the 51-source YAML and 36 FITS models into
the shared external `DC4_ROOT/dc4_source_catalog/` directory, also used by the fitting
notebook on this branch and the runtime code from `DC4-SourceCat`. The current folder is
`/Users/parshadkp/Library/CloudStorage/OneDrive-ClemsonUniversity/COSI/Radio_Quiet_AGN/DC4_Files/dc4_source_catalog`.
The YAML and FITS are not stored in the repository. Creation requires the `cosi-sim` input
definitions, spectra, maps and light curves. Fitting requires none of these inputs.

The comparison section shows absolute source counts, model/simulation ratios and
response-folded measured-energy spectra in the native ten bins. It starts with
Cen A and NGC 4151. Change `COMPARE_SOURCES` to select others, or
`list(catalog.sources)` for all 51. Extended sources need a verified full-observation
extended detector response; construction is deferred until it is actually needed.

Set `SOURCE_EVENT_ROOT` to your corrected simulations, or override individual
entries in `source_event_files`. The explicit 51-source mapping is in
`dc4_catalog_tools/simulation_files.py`. It sums the two files for each grouped AGN
and uses one file for Crab's combined components. No GRBs/MGFs are included.
Changed event files invalidate their numerical comparison caches automatically.
Caches and CSV/PDF outputs stay in `work/`, outside the model folder.

The saved plots currently use the **legacy** DC4 files: total model/simulation
ratios are 1.0423 for Cen A and 1.0829 for NGC 4151. They are not results from
the future chunk-corrected simulations. Static source averages can also differ
from variable-source events through flux/pointing correlations. No event counts
are used to renormalize the models.

Creation-only code lives in `dc4_catalog_tools/generation.py` and `megalib.py`.
The installed `cosipy/` package has no catalog creation or fitting helper modules.
Spectral FITS files contain only two physical columns, their names/units and FITS
table structure; interpolation is stored as an ordinary model property in YAML.
Reference times, input hashes and averaging diagnostics remain in notebook output.

Run the focused checks with:

```sh
python -m pytest tests/test_source_catalog.py tests/response/test_integrals.py tests/response/test_extended_source_response.py tests/threeml/test_spectral_fitting.py docs/tutorials/spectral_fits/source_catalog/test_catalog_creation.py docs/tutorials/spectral_fits/source_catalog/test_catalog_fitting.py -q
```

Validation: 40 focused checks passed. The creator contains executed Cen A/NGC 4151
comparison plots and tables; the fitting notebook contains the executed independent
CPL recovery check. No new corrected DC4 mock has been substituted yet.
