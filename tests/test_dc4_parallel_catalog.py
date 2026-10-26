"""Portable catalog contract and small source-response regressions."""

import hashlib
import json
import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import yaml
from astromodels import Model, PointSource, Powerlaw, SpectralComponent
from astropy.io import fits
from histpy import Axis, Axes, Histogram

from cosipy.response import (
    BinnedThreeMLModelFolding,
    ComponentPointSourceResponse,
    LightcurveProfile,
)
from cosipy.util.megalib import megalib_map_to_galprop_fits


CATALOG_DIR = (Path(__file__).resolve().parents[1] /
               "docs/tutorials/spectral_fits/continuum_fit/AGN")


def test_portable_pair_has_expected_sources_and_components():
    catalog_path = CATALOG_DIR / "source_catalog_DC4_parallel_full.yaml"
    recipe = json.loads(catalog_path.with_suffix(".response.json").read_text())
    catalog = yaml.safe_load(catalog_path.read_text())
    assert recipe["schema"] == "dc4-component-lightcurve-response-v2"
    assert recipe["selection"] == "full"
    assert recipe["time_basis"] == "elapsed"
    assert recipe["catalog"] == catalog_path.name
    assert "gti_target" not in recipe
    assert len(catalog) == 71
    assert sum(key.endswith("(point source)") for key in catalog) == 52
    assert sum(key.endswith("(extended source)") for key in catalog) == 19
    rows = recipe["components"]
    assert len(rows) == 74
    assert len({(row["source"], row["component"]) for row in rows}) == 74
    assert sum(bool(row["lightcurve"]) for row in rows) == 35
    assert {row["source"] for row in rows} == {
        key.split(" (")[0] for key in catalog
    }
    assert not any("/Users/" in text for text in
                   (catalog_path.read_text(), json.dumps(recipe)))
    assert {row["component"] for row in rows if row["source"] == "crab"} == {
        "Crab_Nebula", "Crab_P1", "Crab_Brg", "Crab_P2"
    }
    nova = next(row for row in rows if row["source"] == "co_nova_continuum")
    assert nova["normalization mode"] == "parallel_megalib"
    assert nova["simulation bins"] == 500


def test_recipe_lightcurve_hashes_when_library_is_available():
    root = Path(os.environ.get("COSI_SIM_SOURCE_LIBRARY", Path.home() /
               "Software/cosi-sim/cosi_sim/Source_Library"))
    if not root.is_dir():
        pytest.skip("cosi-sim source library is not installed")
    recipe = json.loads((CATALOG_DIR /
        "source_catalog_DC4_parallel_full.response.json").read_text())
    for row in recipe["components"]:
        if not row["lightcurve"]:
            continue
        relative = row["lightcurve"].removeprefix("SOURCE_LIBRARY/")
        path = root / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["lightcurve sha256"]


def test_megalib_map_conversion_preserves_per_mev_intensity(tmp_path):
    source = tmp_path / "map.dat"
    source.write_text(
        "PA -180 180\nTA 0 180\nEA 100 1000\n" +
        "".join(f"AP {i} {j} {k} {k+1}\n"
                for i in range(2) for j in range(2) for k in range(2))
    )
    output = tmp_path / "template.fits"
    megalib_map_to_galprop_fits(source, output, nside=8)
    with fits.open(output) as hdul:
        assert hdul["SKYMAP"].header["NSIDE"] == 8
        assert hdul["SKYMAP"].header["MAPVERS"] == "perMeV_v3"
        values = np.column_stack([
            hdul["SKYMAP"].data[name] for name in hdul["SKYMAP"].columns.names
        ])
        assert values.shape == (768, 6)
        np.testing.assert_allclose(values[:, 2:4],
                                   np.tile([1000., 2000.], (768, 1)))
        np.testing.assert_allclose(values[:, [0, 1, 4, 5]], 0.)
        np.testing.assert_allclose(hdul["ENERGIES"].data["ENERGY"][2:4], [.1, 1.])


def test_full_catalog_mean_and_new_gti_keep_same_amplitude():
    # One flare in four hours. A new GTI changes the response weights but not
    # the catalog normalization, which was defined over the full observation.
    profile = LightcurveProfile([0., 1., 2., 3., 4.], [0., 0., 1., 0., 0.],
                                mode="raw")
    full_factor = profile.integral([0.], [4.])[0] / 4.
    catalog_k = 8. * full_factor
    quiet = profile.response_weights([0.], [1.], full_factor)
    flare = profile.response_weights([1.], [3.], full_factor)
    assert catalog_k == 2.
    np.testing.assert_allclose(quiet, [0.])
    np.testing.assert_allclose(flare, [2.])
    assert catalog_k * quiet[0] == 0.
    assert catalog_k * flare[0] == 4.


def test_source_specific_response_overrides_only_named_source():
    axes = Axes([Axis([0., 1., 2.], label="Em")])

    class Response:
        def __init__(self, scale):
            self.axes, self.scale = axes, scale

        def copy(self):
            return Response(self.scale)

        def set_source(self, source):
            self.source = source

        def expectation(self, copy=True):
            k = self.source.spectrum.main.Powerlaw.K.value
            return Histogram(axes, contents=np.ones(2) * self.scale * k)

    model = Model(
        PointSource("steady", l=0., b=0., spectral_shape=Powerlaw(K=1.)),
        PointSource("variable", l=0., b=0., spectral_shape=Powerlaw(K=2.)),
    )
    folding = BinnedThreeMLModelFolding(
        type("Data", (), {"axes": axes})(), Response(1.),
        source_specific_point_source_responses={"variable": Response(3.)},
    )
    folding.set_model(model)
    np.testing.assert_allclose(folding.expectation().contents, [7., 7.])
    model.variable.spectrum.main.Powerlaw.K.value = 4.
    np.testing.assert_allclose(folding.expectation().contents, [13., 13.])


def test_component_response_preserves_distinct_temporal_components():
    axes = Axes([Axis([100., 200.], label="Em", unit=u.keV)])

    class Response:
        def __init__(self, weight):
            self.axes, self.weight = axes, weight

        def copy(self):
            return Response(self.weight)

        def set_source(self, source):
            self.source = source

        def expectation(self, copy=True):
            value = sum(c.shape.K.value for c in self.source.components.values())
            return Histogram(axes, contents=[self.weight * value])

    source = PointSource("crab", l=0., b=0., components=[
        SpectralComponent("nebula", Powerlaw(K=1.)),
        SpectralComponent("peak1", Powerlaw(K=2.)),
    ])
    adapter = ComponentPointSourceResponse({
        "nebula": Response(1.), "peak1": Response(3.),
    })
    adapter.set_source(source)
    np.testing.assert_allclose(adapter.expectation().contents, [7.])
    source.components["peak1"].shape.K.value = 4.
    np.testing.assert_allclose(adapter.expectation().contents, [13.])
