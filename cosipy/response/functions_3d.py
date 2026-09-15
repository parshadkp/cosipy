import numpy as np

import astropy.units as u
from astropy.coordinates import Galactic

from histpy import Histogram

import logging
logger = logging.getLogger(__name__)

def get_integrated_extended_model_3d(extendedmodel, image_axis, energy_axis):

    """
    Calculate the integrated flux map for an extended source model.

    Parameters
    ----------
    extendedmodel : astromodels.ExtendedSource
        An astromodels extended source model object. This model represents
        the spatial and spectral distribution of an extended astronomical source.
    image_axis : histpy.HealpixAxis
        Spatial axis for the image.
    energy_axis : histpy.Axis
        Energy axis defining the energy bins.

    Returns
    -------
    flux_map : histpy.Histogram
        2D histogram representing the integrated flux map.
    """

    from cosipy.threeml.custom_functions import GalpropHealpixModel

    if not isinstance(image_axis.coordsys, Galactic):
        raise ValueError

    l, b = image_axis.pix2ang(np.arange(image_axis.npix), lonlat=True)

    if isinstance(extendedmodel.spatial_shape, GalpropHealpixModel):

        # The norm is updated internally by 3ML for each likelihood call
        norm = extendedmodel.spatial_shape.K.value

        # Make sure the dummy spectral parameter is fixed.
        extendedmodel.spectrum.main.Constant.k.free = False

        # Integrate on the union of the response-bin edges and the template's
        # native energy knots. Sampling only at the response-bin edges misses
        # narrow lines that fall inside a broad incident-energy bin.
        energy_edges = energy_axis.edges.to_value(u.MeV)
        cached_edges = getattr(
            extendedmodel.spatial_shape,
            "intg_flux_energy_edges",
            None,
        )
        integrated_flux = getattr(
            extendedmodel.spatial_shape,
            "intg_flux",
            None,
        )
        cache_is_valid = (
            isinstance(integrated_flux, np.ndarray)
            and isinstance(cached_edges, np.ndarray)
            and integrated_flux.shape == (image_axis.npix, energy_axis.nbins)
            and np.array_equal(cached_edges, energy_edges)
        )

        if not cache_is_valid:
            if not extendedmodel.spatial_shape._file_loaded:
                extendedmodel.spatial_shape.load_file(
                    extendedmodel.spatial_shape._fitsfile
                )

            template_energy = (
                extendedmodel.spatial_shape.energy.to_value(u.MeV)
            )
            inside_response = (
                (template_energy > energy_edges[0])
                & (template_energy < energy_edges[-1])
            )
            integration_grid = np.unique(
                np.concatenate(
                    (energy_edges, template_energy[inside_response])
                )
            )
            intensity_unit = (u.MeV * u.s * u.cm**2 * u.sr) ** (-1)
            intensity = extendedmodel.spatial_shape.evaluate(
                l,
                b,
                integration_grid * u.MeV,
                1.0,
            ).to_value(intensity_unit)

            integrated_flux = np.zeros(
                (image_axis.npix, energy_axis.nbins),
                dtype=float,
            )
            logger.info("Integrating intensity over native template energies...")
            for energy_index, (lo_lim, hi_lim) in enumerate(
                zip(energy_edges[:-1], energy_edges[1:])
            ):
                in_bin = (
                    (integration_grid >= lo_lim)
                    & (integration_grid <= hi_lim)
                )
                integrated_flux[:, energy_index] = np.trapezoid(
                    intensity[:, in_bin],
                    integration_grid[in_bin],
                    axis=1,
                )

            extendedmodel.spatial_shape.intg_flux = integrated_flux
            extendedmodel.spatial_shape.intg_flux_energy_edges = (
                energy_edges.copy()
            )

        flux = norm * integrated_flux

        flux_map = Histogram((image_axis, energy_axis), \
                             contents = flux, \
                             unit = (u.s * u.cm**2 * u.sr) ** (-1),
                             copy_contents = False)

    return flux_map
