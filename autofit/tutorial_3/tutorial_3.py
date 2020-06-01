import os
import sys

autolens_version = "0.45.0"

import autofit as af
af.conf.instance = af.conf.Config(
    config_path="./config_{}".format(autolens_version),
    output_path="./output"
)
import autolens as al

from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.dataset.dataset import Dataset, MaskedDataset
from src.model import profiles
from src.fit import (
    fit
)
from src.phase import (
    phase as ph,
)

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import plot_utils as plot_utils

import interferometry_utils.load_utils as interferometry_load_utils


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


def dirty_cube_from_visibilities(visibilities, transformers, shape, invert=True):

    if len(transformers) != visibilities.shape[0]:
        raise ValueError("...")

    dirty_cube = np.zeros(
        shape=shape
    )
    for i in range(visibilities.shape[0]):
        dirty_cube[i] = transformers[i].image_from_visibilities(
            visibilities=visibilities[i]
        )

    if invert:
        return dirty_cube[:, ::-1, :]
    return dirty_cube



uv_wavelengths = load_uv_wavelengths(
    filename="{}/uv_wavelengths.fits".format(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    )
)

n_channels = uv_wavelengths.shape[0]
z_step_kms = 50.0 # NOTE: This does not correspond to the actual value of z_step_kms for this uv_wavelengths dataset

#transformer_class =

if __name__ == "__main__":

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

    transformers = []
    for i in range(uv_wavelengths.shape[0]):
        transformer = al.TransformerFINUFFT(
            uv_wavelengths=uv_wavelengths[i],
            grid=grid_3d.grid_2d.in_radians
        )
        transformers.append(transformer)

    xy_mask = Mask3D.unmasked(
        shape_3d=grid_3d.shape_3d,
        pixel_scales=grid_3d.pixel_scales,
        sub_size=grid_3d.sub_size,
    )

    model = profiles.Kinematical(
        centre=(0.0, 0.0),
        z_centre=16.0,
        intensity=1.0,
        effective_radius=0.5,
        inclination=30.0,
        phi=50.0,
        turnover_radius=0.05,
        maximum_velocity=200.0,
        velocity_dispersion=50.0
    )

    cube = model.profile_cube_from_grid(
        grid=grid_3d.grid_2d,
        shape_3d=grid_3d.shape_3d,
        z_step_kms=z_step_kms
    )
    # plot_utils.plot_cube(
    #     cube=cube,
    #     ncols=8
    # )

    visibilities = np.zeros(
        shape=uv_wavelengths.shape
    )
    for i in range(visibilities.shape[0]):
        visibilities[i] = transformers[i].visibilities_from_image(
                image=Image(
                    array_2d=cube[i]
                )
            )
    # plot_utils.plot_cube(
    #     cube=dirty_cube_from_visibilities(
    #         visibilities=visibilities,
    #         transformers=transformers,
    #         shape=cube.shape
    #     ),
    #     ncols=8
    # )
    # exit()


    noise_map = np.random.normal(
        loc=0.0, scale=10**-1.0, size=visibilities.shape
    )
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=np.add(
            visibilities, noise_map
        ),
        noise_map=noise_map,
        z_step_kms=z_step_kms
    )
    # for i in range(dataset.data.shape[0]):
    #     plt.figure()
    #     plt.plot(
    #         dataset.data[i, :, 0],
    #         dataset.data[i, :, 1],
    #         linestyle="None",
    #         marker="o",
    #         color="black"
    #     )
    #     plt.show()
    # exit()
    # plot_utils.plot_cube(
    #     cube=dirty_cube_from_visibilities(
    #         visibilities=dataset.visibilities,
    #         transformers=transformers,
    #         shape=cube.shape,
    #         invert=True
    #     ),
    #     ncols=8,
    #     cube_contours=cube,
    #
    # )
    # exit()

    masked_dataset = MaskedDataset(
        dataset=dataset,
        xy_mask=xy_mask
    )

    # residual_map = fit.residual_map_from_data_model_data_and_mask(
    #     data=masked_dataset.data,
    #     mask=masked_dataset.uv_mask,
    #     model_data=visibilities
    # )
    # # plot_utils.plot_cube(
    # #     cube=dirty_cube_from_visibilities(
    # #         visibilities=residual_map,
    # #         transformers=transformers,
    # #         shape=cube.shape
    # #     ),
    # #     ncols=8
    # # )
    # # exit()
    # likelihood = fit.likelihood_from_chi_squared_and_noise_normalization(
    #     chi_squared=fit.chi_squared_from_chi_squared_map_and_mask(
    #         chi_squared_map=fit.chi_squared_map_from_residual_map_noise_map_and_mask(
    #             residual_map=residual_map,
    #             noise_map=masked_dataset.noise_map,
    #             mask=masked_dataset.uv_mask
    #         ),
    #         mask=masked_dataset.uv_mask
    #     ),
    #     noise_normalization=fit.noise_normalization_from_noise_map_and_mask(
    #         noise_map=masked_dataset.noise_map_real_and_imag_averaged,
    #         mask=masked_dataset.uv_mask_real_and_imag_averaged
    #     )
    # )
    # print(likelihood)
    # exit()

    model_1 = profiles.Kinematical

    # model_1.z_centre = af.GaussianPrior(
    #     mean=grid_3d.n_channels / 2.0,
    #     sigma=2.0
    # )

    phase_name = "phase_tutorial_3"
    os.system(
        "rm -r output/{}*".format(phase_name)
    )
    phase = ph.Phase(
        phase_name=phase_name,
        profiles=af.CollectionPriorModel(
            model_1=model_1
        )
    )

    phase.optimizer.constant_efficiency = True
    phase.optimizer.n_live_points = 100
    phase.optimizer.sampling_efficiency = 0.5
    phase.optimizer.evidence_tolerance = 100.0

    phase.run(
        dataset=dataset,
        xy_mask=xy_mask
    )
