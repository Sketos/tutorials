import autolens as al

import os
import sys
import numpy as np

sys.path.append(
    "{}/tutorials/autofit/tutorial_6".format(
        os.environ["GitHub"]
    )
)

from src.grid.grid import Grid3D



class Dataset:
    def __init__(self, uv_wavelengths, visibilities, noise_map, z_step_kms):

        self.uv_wavelengths = uv_wavelengths
        self.visibilities = visibilities
        self.noise_map = noise_map

        self.z_step_kms = z_step_kms


class MaskedDataset:
    def __init__(self, dataset, xy_mask, uv_mask=None):

        self.dataset = dataset

        self.xy_mask = xy_mask

        self.grid_3d = Grid3D(
            grid_2d=al.Grid.uniform(
                shape_2d=xy_mask.shape_2d,
                pixel_scales=xy_mask.pixel_scales,
                sub_size=xy_mask.sub_size
            ),
            n_channels=xy_mask.n_channels
        )

        self.uv_wavelengths = dataset.uv_wavelengths
        self.visibilities = dataset.visibilities

        if uv_mask is None:
            self.uv_mask = np.full(
                shape=self.visibilities.shape,
                fill_value=False
            )
        else:
            self.uv_mask = uv_mask

        self.uv_mask_real_and_imag_averaged = np.full(
            shape=self.uv_mask.shape[:-1],
            fill_value=False
        )

        self.noise_map = dataset.noise_map

        self.noise_map_real_and_imag_averaged = np.average(
            a=self.noise_map, axis=-1
        )

        self.z_step_kms = dataset.z_step_kms


    @property
    def data(self):
        return self.visibilities
