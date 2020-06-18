import autoarray as aa
import autolens as al

import os
import sys
import numpy as np

sys.path.append(
    "{}/tutorials/autofit/tutorial_0".format(
        os.environ["GitHub"]
    )
)

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import calibration_utils as calibration_utils

class Dataset:
    def __init__(self, uv_wavelengths, visibilities, noise_map, antennas, time=None):

        self.uv_wavelengths = uv_wavelengths
        self.visibilities = visibilities
        self.noise_map = noise_map

        if (self.uv_wavelengths.shape == antennas.shape):
            self.antennas = antennas
        else:
            raise ValueError()

        self.time = time

        # NOTE: In the case where the real an imag part of the noise is the
        # same, which in real data is not.
        # C = np.diag(
        #     np.power(
        #         np.divide(
        #             np.average(a=noise_map, axis=-1),
        #             self.amplitudes,
        #         ),
        #         -2
        #     )
        # )

        self.C = np.diag(
            np.power(
                np.divide(
                    np.sqrt(
                        np.add(
                            self.visibilities[:, 0]**2.0 * self.noise_map[:, 0]**2.0,
                            self.visibilities[:, 1]**2.0 * self.noise_map[:, 1]**2.0
                        )
                    ),
                    self.amplitudes**2.0,
                ),
                -2
            )
        )

        self.f = calibration_utils.compute_f_matrix_from_antennas(
            antennas=self.antennas
        )

        self.A = calibration_utils.compute_A_matrix_from_f_and_C_matrices(
            f=self.f, C=self.C
        )
        self.B = calibration_utils.compute_B_matrix_from_f_and_C_matrices(
            f=self.f, C=self.C
        )

    # @property
    # def amplitudes(self):
    #     return np.hypot(
    #         self.visibilities[:, 0],
    #         self.visibilities[:, 1]
    #     )
    #
    # @property
    # def phases(self):
    #     return np.arctan2(
    #         self.visibilities[:, 1],
    #         self.visibilities[:, 0]
    #     )

    @property
    def data(self):
        return self.visibilities

    @property
    def amplitudes(self):
        return self.visibilities.amplitudes

    @property
    def phases(self):
        return self.visibilities.phases


# NOTE: rename uv_mask to visibilities_mask and xy_mask to real_space_mask
class MaskedDataset:
    def __init__(self, dataset, xy_mask, uv_mask=None):

        self.dataset = dataset

        self.xy_mask = xy_mask

        self.grid = al.Grid.uniform(
            shape_2d=xy_mask.shape_2d,
            pixel_scales=xy_mask.pixel_scales,
            sub_size=xy_mask.sub_size
        )

        self.uv_wavelengths = dataset.uv_wavelengths
        self.visibilities = dataset.visibilities
        self.noise_map = dataset.noise_map
        self.antennas = dataset.antennas

        self.f = dataset.f
        self.A = dataset.A
        self.B = dataset.B

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
                    "The shape of the \"uv_mask\" does not match the shape of the visibilities"
                )

        self.uv_mask_real_and_imag_averaged = np.full(
            shape=self.uv_mask.shape[:-1],
            fill_value=False
        )
        for i in range(self.uv_mask.shape[0]):
                if self.uv_mask[i, 0] == self.uv_mask[i, 1] == True:
                    self.uv_mask_real_and_imag_averaged[i] = True

        self.noise_map_real_and_imag_averaged = np.average(
            a=self.noise_map, axis=-1
        )


    @property
    def data(self):
        return self.visibilities

    @property
    def sigma(self):
        return self.noise_map

    @property
    def mask(self):
        return self.uv_mask

    @property
    def phases(self):
        return self.visibilities.phases




# class MaskedDatasetLite:
#     def __init__(
#         self,
#         uv_wavelengths,
#         visibilities,
#         noise_map,
#         noise_map_real_and_imag_averaged,
#         uv_mask,
#         uv_mask_real_and_imag_averaged
#     ):
#
#         self.uv_wavelengths = uv_wavelengths
#         self.visibilities = visibilities
#
#         self.noise_map = noise_map
#         self.noise_map_real_and_imag_averaged = noise_map_real_and_imag_averaged
#
#         self.uv_mask = uv_mask
#         self.uv_mask_real_and_imag_averaged = uv_mask_real_and_imag_averaged
#
#     @property
#     def data(self):
#         return self.visibilities
#
#     @property
#     def mask(self):
#         return self.uv_mask
