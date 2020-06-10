import autolens as al

import os
import sys

sys.path.append(
    "{}/tutorials/autofit/tutorial_2".format(
        os.environ["GitHub"]
    )
)

from src.grid.grid import Grid3D



class Dataset:
    def __init__(self, data, noise_map, z_step_kms):

        self.data = data
        self.noise_map = noise_map

        self.z_step_kms = z_step_kms


class MaskedDataset:
    def __init__(self, dataset, mask):

        self.dataset = dataset
        # TODO: Check that dataset has attribute data.

        # TODO: Check that mask and dataset.data have the same shape.
        self.mask = mask

        #if mask.pixel_scales is not None:

        self.grid_3d = Grid3D(
            grid_2d=al.Grid.uniform(
                shape_2d=mask.shape_2d,
                pixel_scales=mask.pixel_scales,
                sub_size=mask.sub_size
            ),
            n_channels=mask.n_channels
        )

        self.data = dataset.data
        self.noise_map = dataset.noise_map

        self.z_step_kms = dataset.z_step_kms
