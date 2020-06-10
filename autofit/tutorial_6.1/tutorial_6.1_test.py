import os
import sys
import numpy as np
import matplotlib.pyplot as plt

autolens_version = "0.45.0"
#autolens_version = "0.46.2"

config_path = "./config_{}".format(
    autolens_version
)
if os.environ["HOME"].startswith("/cosma"):
    cosma_server = "7"
    output_path = "{}/tutorials/autofit/tutorial_6/output".format(
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
if not (al.__version__ == autolens_version):
    raise ValueError("...")

from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.region.region import Region
from src.dataset.dataset import Dataset, MaskedDataset
from src.model import profiles, mass_profiles, dark_mass_profiles
from src.fit import fit
from src.phase import phase
from src.plot import fit_plots

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import spectral_utils as spectral_utils
import plot_utils as plot_utils

import interferometry_utils.load_utils as interferometry_load_utils

import autolens_utils.autolens_plot_utils as autolens_plot_utils
import autolens_utils.autolens_tracer_utils as autolens_tracer_utils


lens_redshift = 0.5
source_redshift = 2.0

n_pixels = 100
pixel_scale = 0.05


class Image:
    def __init__(self, array_2d):
        self.array_2d = array_2d

    @property
    def in_1d_binned(self):
        return np.ndarray.flatten(self.array_2d)

    @property
    def in_2d_binned(self):
        return self.array_2d


def reshape_array(array):

    return array.reshape(
        -1,
        array.shape[-1]
    )


def load_uv_wavelengths(filename):

    if not os.path.isfile(filename):
        raise IOError(
            "The file {} does not exist".format(filename)
        )

    u_wavelengths, v_wavelengths = interferometry_load_utils.load_uv_wavelengths_from_fits(
        filename=filename
    )

    uv_wavelengths = np.stack(
        arrays=(u_wavelengths, v_wavelengths),
        axis=-1
    )

    return uv_wavelengths


uv_wavelengths = load_uv_wavelengths(
    filename="{}/uv_wavelengths.fits".format(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    )
)

n_channels = uv_wavelengths.shape[0]
z_step_kms = 50.0 # NOTE: This does not correspond to the actual value of z_step_kms for this uv_wavelengths dataset


# def region(n, n_min, n_max, invert=False):
#
#     mask = np.zeros(
#         shape=int(n), dtype=int
#     )
#
#     if n_min > 0 and n_min < n_max and n_max < n:
#         mask[n_min:n_max] = 1
#     else:
#         raise ValueError("...")
#
#     return mask.astype(bool)


if __name__ == "__main__":

    transformer_class = al.TransformerFINUFFT

    grid_3d = Grid3D(
        grid_2d=al.Grid.uniform(
            shape_2d=(
                n_pixels,
                n_pixels
            ),
            pixel_scales=(
                pixel_scale,
                pixel_scale
            ),
            sub_size=1
        ),
        n_channels=n_channels
    )

    lens_mass_profile = mass_profiles.EllipticalPowerLaw(
        centre=(0.0, 0.0),
        axis_ratio=0.75,
        phi=45.0,
        einstein_radius=1.0,
        slope=2.0
    )

    subhalo_mass_profile = dark_mass_profiles.SphericalNFWMCRLudlow(
        centre=(-0.75, -0.5),
        mass_at_200=1e9,
        redshift_object=lens_redshift,
        redshift_source=source_redshift
    )
    #exit()

    model_1 = profiles.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.75,
        phi=45.0,
        intensity=0.00005,
        effective_radius=0.5,
        sersic_index=1.0,
    )

    # ================= #
    """
    # NOTE: THIS SHOULD BE A TEST SOMEWHERE
    tracer = al.Tracer.from_galaxies(
        galaxies=[
            al.Galaxy(
                redshift=lens_redshift,
                mass=lens_mass_profile,
            ),
            al.Galaxy(
                redshift=source_redshift,
                light=model_1
            )
        ]
    )

    lensed_image = tracer.profile_image_from_grid(
        grid=grid_3d.grid_2d
    )
    print(lensed_image.in_2d.shape)

    image = model_1.profile_image_from_grid(
        grid=grid_3d.grid_2d,
        grid_radial_minimum=None
    )

    lens_image_approx = autolens_tracer_utils.lensed_image_from_tracer(
        tracer=tracer,
        grid=grid_3d.grid_2d,
        image=image.in_2d,
    )

    figure, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].imshow(lensed_image.in_2d)
    axes[1].imshow(lens_image_approx)
    axes[2].imshow(lensed_image.in_2d - lens_image_approx)
    plt.show()
    exit()
    """
    # ================= #

    model_2 = profiles.Kinematical(
        centre=(0.0, 0.0),
        z_centre=16.0,
        intensity=5.0,
        effective_radius=0.5,
        inclination=30.0,
        phi=50.0,
        turnover_radius=0.05,
        maximum_velocity=200.0,
        velocity_dispersion=50.0
    )

    cube_model_1 = model_1.profile_cube_from_grid(
        grid=grid_3d.grid_2d,
        shape_3d=grid_3d.shape_3d,
        z_step_kms=z_step_kms
    )
    cube_model_2 = model_2.profile_cube_from_grid(
        grid=grid_3d.grid_2d,
        shape_3d=grid_3d.shape_3d,
        z_step_kms=z_step_kms
    )
    cube = cube_model_1 #+ cube_model_2
    # plot_utils.plot_cube(
    #     cube=cube,
    #     ncols=8
    # )
    # exit()

    lensed_cube = autolens_tracer_utils.lensed_cube_from_tracer(
        tracer=al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=lens_redshift,
                    mass=lens_mass_profile,
                ),
                al.Galaxy(
                    redshift=lens_redshift,
                    mass=subhalo_mass_profile,
                ),
                al.Galaxy(
                    redshift=source_redshift,
                    light=al.lp.LightProfile()
                )
            ]
        ),
        grid=grid_3d.grid_2d,
        cube=cube
    )
    # plot_utils.plot_cube(
    #     cube=lensed_cube,
    #     ncols=8,
    #     extent=grid_3d.extent,
    #     points=[
    #         plot_utils.Point(
    #             y=subhalo_mass_profile.centre[0],
    #             x=subhalo_mass_profile.centre[1],
    #             color="black"
    #         ),
    #         plot_utils.Point(
    #             y=lens_mass_profile.centre[0],
    #             x=lens_mass_profile.centre[1],
    #             color="white"
    #         ),
    #
    #     ]
    # )
    # exit()

    emission_line_region = Region.from_limits(
        n=n_channels,
        n_min=10,
        n_max=22
    )

    # # NOTE: TEST ...
    # #cube_region = cube[emission_line_region]
    # cube[emission_line_region] = 0.0
    # plot_utils.plot_cube(
    #     cube=cube,
    #     ncols=8
    # )
    # exit()

    # # NOTE: Plotting
    # spectrum_model_1 = spectral_utils.get_spectrum_from_cube(
    #     cube=cube_model_1
    # )
    # spectrum_model_2 = spectral_utils.get_spectrum_from_cube(
    #     cube=cube_model_2
    # )
    # plt.figure(
    #     figsize=(10, 5)
    # )
    # plt.plot(spectrum_model_1, color="r")
    # plt.plot(spectrum_model_2, color="b")
    # plt.show()
    # exit()


    transformers = []
    for i in range(uv_wavelengths.shape[0]):
        transformer = transformer_class(
            uv_wavelengths=uv_wavelengths[i],
            grid=grid_3d.grid_2d.in_radians
        )
        transformers.append(transformer)

    visibilities = np.zeros(
        shape=uv_wavelengths.shape
    )
    for i in range(visibilities.shape[0]):
        visibilities[i] = transformers[i].visibilities_from_image(
                image=Image(
                    array_2d=lensed_cube[i]
                )
            )

    print(visibilities.shape)

    uv_wavelengths_reshaped = reshape_array(array=uv_wavelengths)
    visibilities_reshaped = reshape_array(array=visibilities)

    transformer = transformer_class(
        uv_wavelengths=uv_wavelengths_reshaped,
        grid=grid_3d.grid_2d.in_radians
    )
    visibilities_reshaped_temp = transformer.visibilities_from_image(
            image=Image(
                array_2d=lensed_cube[0]
            )
        )

    print(
        np.sum(visibilities_reshaped - visibilities_reshaped_temp)
    )
