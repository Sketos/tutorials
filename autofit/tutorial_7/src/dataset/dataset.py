import autolens as al
import autoarray as aa

import os
import sys
import numpy as np

sys.path.append(
    "{}/tutorials/autofit/tutorial_7".format(
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


# class CubeDataset:
#     def __init__(self, uv_wavelengths, visibilities, noise_map, z_step_kms):
#
#         self.shape = visibilities.shape
#         if len(self.shape) == 3:
#             self.datasets = []
#             for i in range(self.shape[0]):
#                 self.datasets.append(
#                     Dataset(
#                         uv_wavelengths=uv_wavelengths[i],
#                         visibilities=visibilities[i],
#                         noise_map=noise_map[i]
#                     )
#                 )
#         else:
#             raise ValueError("Not a CubeDataset")
#
#         self.z_step_kms = z_step_kms


class MaskedDatasetLite:
    def __init__(
        self,
        visibilities,
        noise_map,
        noise_map_real_and_imag_averaged,
        uv_mask,
        uv_mask_real_and_imag_averaged
    ):

        self.visibilities = visibilities

        self.noise_map = noise_map
        self.noise_map_real_and_imag_averaged = noise_map_real_and_imag_averaged

        self.uv_mask = uv_mask
        self.uv_mask_real_and_imag_averaged = uv_mask_real_and_imag_averaged

    @property
    def data(self):
        return self.visibilities


class MaskedDataset:
    def __init__(self, dataset, xy_mask, uv_mask=None, region=None):

        self.dataset = dataset

        self.xy_mask = xy_mask

        # NOTE: This should change and instead be initialized from mask
        if xy_mask.pixel_scales is not None:

            self.grid_3d = Grid3D(
                grid_2d=aa.structures.grids.MaskedGrid.from_mask(
                    mask=xy_mask.mask_2d
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

        # NOTE: We cant have real and imag being different, either both are True of both are False.
        # TODO: If both real and imag are True then this is True, but if one is False use that for the averaging.
        self.uv_mask_real_and_imag_averaged = np.full(
            shape=self.uv_mask.shape[:-1],
            fill_value=False
        )

        self.noise_map = dataset.noise_map

        self.noise_map_real_and_imag_averaged = np.average(
            a=self.noise_map, axis=-1
        )

        self.z_step_kms = dataset.z_step_kms

        # # NOTE: Can this be made more elegantly?
        # if region is not None:
        #     self.uv_wavelengths = self.uv_wavelengths[region]
        #     self.visibilities = self.visibilities[region]
        #     self.noise_map = self.noise_map[region]
        #     self.noise_map_real_and_imag_averaged = self.noise_map_real_and_imag_averaged[region]
        #     self.uv_mask = self.uv_mask[region]
        #     self.uv_mask_real_and_imag_averaged = self.uv_mask_real_and_imag_averaged[region]
        #
        #
        # # print(self.uv_wavelengths.shape)
        # # print(self.visibilities.shape)
        # # print(self.noise_map.shape)
        # # print(self.noise_map_real_and_imag_averaged.shape)
        # # print(self.uv_mask_real_and_imag_averaged.shape)
        # # print("------")
        # # #exit()

    @property
    def data(self):
        return self.visibilities
