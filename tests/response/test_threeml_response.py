import copy
from types import SimpleNamespace

import numpy as np
from astromodels import Model, PointSource, Powerlaw
from histpy import Histogram

from cosipy.response import BinnedThreeMLModelFolding


class TaggedPointSourceResponse:

    def __init__(self, axes, value):
        self.axes = axes
        self.value = value
        self.source = None

    def copy(self):
        return copy.copy(self)

    def set_source(self, source):
        self.source = source

    def expectation(self):
        return Histogram(self.axes, contents=np.array([self.value]))


def test_source_specific_point_source_response():
    template = Histogram([np.array([0.0, 1.0])])
    data = SimpleNamespace(axes=template.axes)

    default_response = TaggedPointSourceResponse(template.axes, 1.0)
    weighted_response = TaggedPointSourceResponse(template.axes, 10.0)

    folding = BinnedThreeMLModelFolding(
        data=data,
        point_source_response=default_response,
        source_specific_point_source_responses={
            "weighted_source": weighted_response,
        },
    )

    model = Model(
        PointSource(
            "default_source",
            l=0.0,
            b=0.0,
            spectral_shape=Powerlaw(),
        ),
        PointSource(
            "weighted_source",
            l=1.0,
            b=1.0,
            spectral_shape=Powerlaw(),
        ),
    )
    folding.set_model(model)

    expectation = folding.expectation(copy=False)

    assert np.allclose(expectation.contents, [11.0])
    assert folding._source_responses["default_source"].value == 1.0
    assert folding._source_responses["weighted_source"].value == 10.0
