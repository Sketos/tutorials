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



n_pixels = 100
pixel_scale = 0.05
n_channels = 32
z_step_kms = 50.0


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



    # mask_3d = Mask3D(
    #     mask_2d=al.Mask.unmasked(
    #         shape_2d=grid_3d.shape_2d,
    #         pixel_scales=grid_3d.pixel_scales,
    #         sub_size=grid_3d.sub_size,
    #     ),
    #     z_mask=np.full(
    #         shape=(grid_3d.n_channels, ),
    #         fill_value=False
    #     )
    # )

    mask_3d = Mask3D.unmasked(
        shape_3d=grid_3d.shape_3d,
        pixel_scales=grid_3d.pixel_scales,
        sub_size=grid_3d.sub_size,
    )
    #print(mask_3d)
    # print(
    #     mask_3d.pixel_scales,
    #     mask_3d.sub_size,
    #     mask_3d.shape,
    #     mask_3d.n_channels,
    #     mask_3d.shape_2d
    # )
    # exit()

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


    noise_map = np.random.normal(
        loc=0.0, scale=10**-5.0, size=grid_3d.shape_3d
    )
    dataset = Dataset(
        data=np.add(
            cube,
            noise_map
        ),
        noise_map=noise_map,
        z_step_kms=z_step_kms
    )
    # plot_utils.plot_cube(
    #     cube=dataset.data,
    #     ncols=8
    # )
    # exit()


    masked_dataset = MaskedDataset(
        dataset=dataset,
        mask=mask_3d
    )



    # residual_map = fit.residual_map_from_data_model_data_and_mask(
    #     data=masked_dataset.data,
    #     mask=masked_dataset.mask,
    #     model_data=cube
    # )
    # # plot_utils.plot_cube(
    # #     cube=residual_map,
    # #     ncols=8
    # # )
    # # exit()
    #
    # likelihood = fit.likelihood_from_chi_squared_and_noise_normalization(
    #     chi_squared=fit.chi_squared_from_chi_squared_map_and_mask(
    #         chi_squared_map=fit.chi_squared_map_from_residual_map_noise_map_and_mask(
    #             residual_map=residual_map,
    #             noise_map=masked_dataset.noise_map,
    #             mask=masked_dataset.mask
    #         ),
    #         mask=masked_dataset.mask
    #     ),
    #     noise_normalization=fit.noise_normalization_from_noise_map_and_mask(
    #         noise_map=masked_dataset.noise_map,
    #         mask=masked_dataset.mask
    #     )
    # )
    # print(likelihood)
    # exit()

    model_1 = profiles.Kinematical

    # model_1.z_centre = af.GaussianPrior(
    #     mean=grid_3d.n_channels / 2.0,
    #     sigma=2.0
    # )

    phase_name = "phase_tutorial_2"
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
        mask=mask_3d
    )
