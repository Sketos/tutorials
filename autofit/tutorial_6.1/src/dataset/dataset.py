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


# NOTE: rename uv_mask to visibilities_mask and xy_mask to real_space_mask
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
        self.noise_map = dataset.noise_map

        if uv_mask is None:
            self.uv_mask = np.full(
                shape=self.visibilities.shape,
                fill_value=False
            )
        else:
            if uv_mask.shape == self.visibilities.shape:
                self.uv_mask = uv_mask
            else:
                raise ValueError(
                    "The shape of the uv_mask does not match the shape of the visibilities"
                )

        self.uv_mask_real_and_imag_averaged = np.full(
            shape=self.uv_mask.shape[:-1],
            fill_value=False
        )
        for i in range(self.uv_mask.shape[0]):
            for j in range(self.uv_mask.shape[1]):
                if self.uv_mask[i, j, 0] == self.uv_mask[i, j, 1] == True:
                    self.uv_mask_real_and_imag_averaged[i, j] = True

        self.noise_map_real_and_imag_averaged = np.average(
            a=self.noise_map, axis=-1
        )

        self.z_step_kms = dataset.z_step_kms


    @property
    def data(self):
        return self.visibilities

    @property
    def mask(self):
        return self.uv_mask

    # @property
    # def grid_3d(self):
    #     return self.grid_3d
    #
    # @property
    # def grid_2d(self):
    #     return self.grid_3d.grid_2d

    @property
    def grid_shape_3d(self):
        return self.grid_3d.shape_3d


class MaskedDatasetLite:
    def __init__(
        self,
        uv_wavelengths,
        visibilities,
        noise_map,
        noise_map_real_and_imag_averaged,
        uv_mask,
        uv_mask_real_and_imag_averaged
    ):

        self.uv_wavelengths = uv_wavelengths
        self.visibilities = visibilities

        self.noise_map = noise_map
        self.noise_map_real_and_imag_averaged = noise_map_real_and_imag_averaged

        self.uv_mask = uv_mask
        self.uv_mask_real_and_imag_averaged = uv_mask_real_and_imag_averaged

    @property
    def data(self):
        return self.visibilities

    @property
    def mask(self):
        return self.uv_mask
