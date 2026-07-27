from cosipy import test_data
from cosipy.threeml.custom_functions import SpecFromDat
import numpy as np


def test_SpecFromDat():

    f = SpecFromDat(dat = test_data.path / "test_SpecFromDat.dat")

    # test_SpecFromDat.dat is a power law with index -2 from 100 keV to 10 MeV
    assert np.all(np.isclose(f.evaluate(np.array([200,2000]), 1),
                             np.array([200.,2000.])**-2 / (1/100 - 1/10000),
                             rtol = .01
                             )
                  )


def test_SpecFromDat_ignores_variable_length_headers(tmp_path):

    spectrum_path = tmp_path / "spectrum.dat"
    spectrum_path.write_text(
        "# Format: DP energy flux\n"
        "# An additional source-specific header line\n"
        "\n"
        "\n"
        "IP LINLIN\n"
        "DP 100.0 1.0e-2\n"
        "DP 200.0 2.5e-3\n"
        "DP 400.0 6.25e-4\n"
        "EN\n"
    )

    spectrum = SpecFromDat(dat=spectrum_path)
    values = spectrum.evaluate(np.array([100.0, 200.0, 400.0]), 1.0)

    assert np.all(np.isfinite(values))
    assert np.all(values > 0)
