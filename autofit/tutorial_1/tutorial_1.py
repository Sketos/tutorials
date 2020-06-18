import os
import sys

autolens_version = "0.45.0"

import autofit as af
af.conf.instance = af.conf.Config(
    config_path="./config_{}".format(autolens_version),
    output_path="./output"
)
import autolens as al

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
# import autolens_init_utils as autolens_init_utils
#
#
# workspace_paths = autolens_init_utils.get_workspace_paths(
#     cosma_server="7"
# )




n_pixels = 100
pixel_scale = 0.05

#profile = "Gaussian"

# class Mask:
#     def __init__(self, grid):
#
#         self.grid = grid
#
#         return 0.0



if __name__ == "__main__":

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

    mask = al.Mask.unmasked(
        shape_2d=grid.shape_2d,
        pixel_scales=grid.pixel_scales,
        sub_size=grid.sub_size,
    )
    # print(mask.mask_sub_1)
    # exit()

    # model = profiles.EllipticalGaussian(
    #     centre=(0.0, 0.5),
    #     axis_ratio=0.75,
    #     phi=45.0,
    #     intensity=0.1,
    #     sigma=0.5
    # )
    model = profiles.EllipticalSersic(
        centre=(0.0, 0.5),
        axis_ratio=0.75,
        phi=45.0,
        intensity=0.1,
        effective_radius=0.5,
        sersic_index=1.0,
    )

    image = model.profile_image_from_grid(
        grid=grid
    )
    # plt.figure()
    # plt.imshow(image.in_2d)
    # plt.show()
    # exit()

    noise_map = np.random.normal(
        loc=0.0, scale=10**-2.0, size=grid.shape_2d
    )
    dataset = Dataset(
        data=np.add(
            image.in_2d,
            noise_map
        ),
        noise_map=noise_map
    )
    # plt.figure()
    # plt.imshow(dataset.data)
    # plt.show()
    # exit()

    # masked_dataset = MaskedDataset(
    #     dataset=dataset,
    #     mask=mask
    # )

    # likelihood = fit.likelihood_from_chi_squared_and_noise_normalization(
    #     chi_squared=fit.chi_squared_from_chi_squared_map_and_mask(
    #         chi_squared_map=fit.chi_squared_map_from_residual_map_noise_map_and_mask(
    #             residual_map=fit.residual_map_from_data_model_data_and_mask(
    #                 data=dataset.data,
    #                 mask=mask,
    #                 model_data=image.in_2d
    #             ),
    #             noise_map=noise_map,
    #             mask=mask
    #         ),
    #         mask=mask
    #     ),
    #     noise_normalization=fit.noise_normalization_from_noise_map_and_mask(
    #         noise_map=noise_map,
    #         mask=mask
    #     )
    # )
    # print(likelihood)
    # exit()

    model = af.PriorModel(profiles.EllipticalSersic)

    model.centre_0 = af.GaussianPrior(
        mean=0.0, sigma=0.05
    )
    model.centre_1 = af.GaussianPrior(
        mean=0.5, sigma=0.05
    )
    model.axis_ratio = af.GaussianPrior(
        mean=0.75, sigma=0.05
    )
    model.phi = af.GaussianPrior(
        mean=45.0, sigma=5.0
    )
    # model.intensity = af.GaussianPrior(
    #     mean=0.1, sigma=0.1
    # )
    model.effective_radius = af.GaussianPrior(
        mean=0.5, sigma=0.25
    )
    model.sersic_index = af.GaussianPrior(
        mean=1.0, sigma=0.2
    )

    phase_name = "phase_tutorial_1"
    os.system(
        "rm -r output/{}".format(phase_name)
    )
    phase = ph.Phase(
        phase_name=phase_name,
        profiles=af.CollectionPriorModel(
            model=model
        ),
    )
    #non_linear_class=af.Emcee

    phase.optimizer.const_efficiency_mode = True
    phase.optimizer.n_live_points = 100
    phase.optimizer.sampling_efficiency = 0.5
    phase.optimizer.evidence_tolerance = 100.0

    phase.run(
        dataset=dataset,
        mask=mask
    )
