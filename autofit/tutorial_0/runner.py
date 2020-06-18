import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy import units
from astropy.io import fits

autolens_version = "0.45.0"
#autolens_version = "0.46.2"

config_path = "./config_{}".format(
    autolens_version
)
if os.environ["HOME"].startswith("/cosma"):
    cosma_server = "7"
    output_path = "{}/tutorials/autofit/tutorial_0/output".format(
        os.environ["COSMA{}_DATA_host".format(cosma_server)]
    )
else:
    output_path="./output"

import autofit as af
af.conf.instance = af.conf.Config(
    config_path=config_path,
    output_path=output_path
)
import autolens as al
import autoarray as aa
if not (al.__version__ == autolens_version):
    raise ValueError("...")


from src.dataset.dataset import Dataset, MaskedDataset
from src.fit import fit
from src.phase import phase
from src.plot import fit_plots

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import string_utils as string_utils
import random_utils as random_utils
import casa_utils as casa_utils
import calibration_utils as calibration_utils
# import spectral_utils as spectral_utils
# import plot_utils as plot_utils

import autolens_utils.autolens_plot_utils as autolens_plot_utils
# import autolens_utils.autolens_tracer_utils as autolens_tracer_utils


lens_redshift = 0.5
source_redshift = 2.0

n_pixels = 100
pixel_scale = 0.05

n_channels = 32

frequencies = casa_utils.generate_frequencies(
    central_frequency=260.0 * units.GHz,
    n_channels=n_channels,
    bandwidth=2.0 * units.GHz
)

uv = fits.getdata("./uv.fits")

uv_wavelengths = casa_utils.convert_uv_coords_from_meters_to_wavelengths(
    uv=uv,
    frequencies=frequencies
)

# NOTE: For this tutorial we channel-average the uv_wavelengths.
uv_wavelengths = np.average(
    a=uv_wavelengths,
    axis=0
)

antennas = fits.getdata("./antennas.fits")
if not (uv_wavelengths.shape[0] == antennas.shape[0]):
    raise ValueError("...")

antennas_unique = np.unique(antennas)

if os.path.isfile("./phase_errors.fits"):
    phase_errors = fits.getdata(filename="./phase_errors.fits")
else:
    np.random.seed(
        seed=random_utils.seed_generator()
    )
    phase_errors = np.random.uniform(
        low=-np.pi/2.0, high=np.pi/2.0, size=(antennas_unique.size,)
    )
    fits.writeto("./phase_errors.fits", data=phase_errors)

f = calibration_utils.compute_f_matrix_from_antennas(
    antennas=antennas
)

data_with_phase_errors = False
self_calibration = False


if __name__ == "__main__":

    def corrupt_visibilities_from_f_matrix_and_phase_errors(visibilities, f, phase_errors):

        phases_corrupted = np.add(
            visibilities.phases,
            np.matmul(
                f.T, phase_errors
            )
        )

        return aa.structures.visibilities.Visibilities(
            visibilities_1d=np.stack(
                arrays=(
                    visibilities.amplitudes * np.cos(phases_corrupted),
                    visibilities.amplitudes * np.sin(phases_corrupted)
                ),
                axis=-1
            )
        )


    transformer_class = al.TransformerFINUFFT

    grid = al.Grid.uniform(
        shape_2d=(
            n_pixels,
            n_pixels
        ),
        pixel_scales=(
            pixel_scale,
            pixel_scale
        ),
        sub_size=1
    )


    lens = al.Galaxy(
        redshift=lens_redshift,
        mass=al.mp.EllipticalPowerLaw(
            centre=(0.0, 0.0),
            axis_ratio=0.75,
            phi=45.0,
            einstein_radius=1.0,
            slope=2.0
        ),
    )

    subhalo = al.Galaxy(
        redshift=lens_redshift,
        mass=al.mp.SphericalNFWMCRLudlow(
            centre=(-0.75, 0.15),
            mass_at_200=1e9,
            redshift_object=lens_redshift,
            redshift_source=source_redshift
        ),
    )

    source = al.Galaxy(
        redshift=source_redshift,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.75,
            phi=45.0,
            intensity=0.001,
            effective_radius=0.5,
            sersic_index=1.0,
        ),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[
            lens,
            subhalo,
            source
        ]
    )

    transformer = transformer_class(
        uv_wavelengths=uv_wavelengths,
        grid=grid.in_radians
    )

    visibilities = tracer.profile_visibilities_from_grid_and_transformer(
        grid=grid,
        transformer=transformer
    )

    if data_with_phase_errors:
        visibilities = corrupt_visibilities_from_f_matrix_and_phase_errors(
            visibilities=visibilities,
            f=f,
            phase_errors=phase_errors
        )


    noise_map = np.random.normal(
        loc=0.0, scale=1.0 * 10**-2.0, size=visibilities.shape
    )

    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=np.add(
            visibilities,
            noise_map
        ),
        noise_map=noise_map,
        antennas=antennas
    )

    xy_mask = al.Mask.unmasked(
        shape_2d=grid.shape_2d,
        pixel_scales=grid.pixel_scales,
        sub_size=grid.sub_size,
    )


    def test(dataset, xy_mask, tracer, self_calibration, transformer_class=al.TransformerFINUFFT):

        masked_dataset = MaskedDataset(
            dataset=dataset,
            xy_mask=xy_mask
        )

        grid = masked_dataset.grid

        transformer = transformer_class(
            uv_wavelengths=masked_dataset.uv_wavelengths,
            grid=grid.in_radians
        )

        model_data = tracer.profile_visibilities_from_grid_and_transformer(
            grid=grid,
            transformer=transformer
        )

        fit_plots.residual_map(
            fit=fit.DatasetFit(
                masked_dataset=masked_dataset,
                model_data=model_data
            ),
            transformer=transformer,
            output_format="show"
        )


    test(
        dataset=dataset,
        xy_mask=xy_mask,
        tracer=al.Tracer.from_galaxies(
            galaxies=[
                lens,
                source
            ]
        ),
        self_calibration=self_calibration
    )
