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
import string_utils as string_utils
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

velocities = np.linspace(
    -n_channels * z_step_kms / 2.0,
    +n_channels * z_step_kms / 2.0,
    n_channels
)


if __name__ == "__main__":

    def test(
        dataset,
        xy_mask,
        profiles,
        lens_redshift,
        source_redshift,
        transformer_class=al.TransformerFINUFFT
    ):

        def src_model_from_profiles(profiles, masked_dataset):

            return sum(
                [
                    profile.profile_cube_from_grid(
                        grid=masked_dataset.grid_3d.grid_2d,
                        shape_3d=masked_dataset.grid_3d.shape_3d,
                        z_step_kms=masked_dataset.z_step_kms
                    )
                    for profile in profiles
                ]
            )

        masked_dataset = MaskedDataset(
            dataset=dataset,
            xy_mask=xy_mask
        )

        transformers = []
        for i in range(masked_dataset.uv_wavelengths.shape[0]):
            transformers.append(
                transformer_class(
                    uv_wavelengths=masked_dataset.uv_wavelengths[i],
                    grid=masked_dataset.grid_3d.grid_2d.in_radians
                )
            )

        len_profiles = []
        src_profiles = []
        for profile in profiles:
            if isinstance(profile, al.mp.MassProfile):
                len_profiles.append(profile)
            else:
                src_profiles.append(profile)

        galaxies = []
        for profile in len_profiles:
            galaxies.append(
                al.Galaxy(
                    redshift=lens_redshift,
                    mass=profile,
                )
            )

        galaxies.append(
            al.Galaxy(
                redshift=source_redshift,
                light=al.lp.LightProfile()
            )
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=galaxies
        )

        cube = src_model_from_profiles(
            profiles=src_profiles,
            masked_dataset=masked_dataset
        )

        lensed_cube = autolens_tracer_utils.lensed_cube_from_tracer(
            tracer=tracer,
            grid=masked_dataset.grid_3d.grid_2d,
            cube=cube
        )

        model_data = np.zeros(
            shape=masked_dataset.data.shape
        )
        for i in range(model_data.shape[0]):
            model_data[i] = transformers[i].visibilities_from_image(
                    image=Image(
                        array_2d=lensed_cube[i]
                    )
                )

        # dirty_cube = autolens_plot_utils.dirty_cube_from_visibilities(
        #     visibilities=masked_dataset.data,
        #     transformers=transformers,
        #     shape=masked_dataset.grid_shape_3d
        # )
        #
        # dirty_model_cube = autolens_plot_utils.dirty_cube_from_visibilities(
        #     visibilities=model_data,
        #     transformers=transformers,
        #     shape=masked_dataset.grid_shape_3d
        # )
        #
        #
        # velocities = np.linspace(
        #     -n_channels * z_step_kms / 2.0,
        #     +n_channels * z_step_kms / 2.0,
        #     n_channels
        # )
        # dirty_moment_0 = spectral_utils.moment_0(
        #     cube=dirty_cube,
        #     velocities=velocities
        # )
        # dirty_model_moment_0 = spectral_utils.moment_0(
        #     cube=dirty_model_cube,
        #     velocities=velocities
        # )
        # plt.figure()
        # plt.imshow(dirty_moment_0 - dirty_model_moment_0, cmap="jet")
        # plt.show()
        # exit()

        fit_plots.residual_map(
            fit=fit.DatasetFit(
                masked_dataset=masked_dataset,
                model_data=model_data
            ),
            transformers=transformers
        )


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

    source_light_profile = profiles.Kinematical(
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

    cube = source_light_profile.profile_cube_from_grid(
        grid=grid_3d.grid_2d,
        shape_3d=grid_3d.shape_3d,
        z_step_kms=z_step_kms
    )

    # al.Galaxy(
    #     redshift=lens_redshift,
    #     mass=subhalo_mass_profile,
    # ),
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

    noise_map = np.random.normal(
        loc=0.0, scale=5.0 * 10**-1.0, size=visibilities.shape
    )
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=np.add(
            visibilities,
            noise_map
        ),
        noise_map=noise_map,
        z_step_kms=z_step_kms
    )
    # plot_utils.plot_cube(
    #     cube=autolens_plot_utils.dirty_cube_from_visibilities(
    #         visibilities=dataset.visibilities,
    #         transformers=transformers,
    #         shape=cube.shape
    #     ),
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
    #     ],
    #     #cube_contours=lensed_cube,
    # )
    # exit()

    xy_mask = Mask3D.unmasked(
        shape_3d=grid_3d.shape_3d,
        pixel_scales=grid_3d.pixel_scales,
        sub_size=grid_3d.sub_size,
    )

    # # NOTE: ...
    # fit_plots.residual_map(
    #     fit=fit.DatasetFit(
    #         masked_dataset=MaskedDataset(
    #             dataset=dataset,
    #             xy_mask=xy_mask
    #         ),
    #         model_data=visibilities
    #     ),
    #     transformers=transformers
    # )
    # exit()


    test(
        dataset=dataset,
        xy_mask=xy_mask,
        profiles=[
            lens_mass_profile,
            source_light_profile
        ],
        lens_redshift=lens_redshift,
        source_redshift=source_redshift,
        transformer_class=transformer_class,
    )
