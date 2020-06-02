import autolens as al
import autoarray as aa

import os
import sys
import numpy as np

sys.path.append(
    "{}/tutorials/autofit/tutorial_8".format(
        os.environ["GitHub"]
    )
)

from src.grid.grid import Grid3D


def reshape_array(array):

    return array.reshape(
        -1,
        array.shape[-1]
    )


class Dataset:
    def __init__(self, uv_wavelengths, visibilities, noise_map, z_step_kms):

        self.uv_wavelengths = uv_wavelengths
        self.visibilities = visibilities
        self.noise_map = noise_map

        self.z_step_kms = z_step_kms


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

        self.xy_mask = xy_mask # NOTE: rename to real_space_mask

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
        self.noise_map = dataset.noise_map

        if uv_mask is None:
            self.uv_mask = np.full(
                shape=self.visibilities.shape,
                fill_value=False
            )
        else:
            self.uv_mask = uv_mask

        # TODO: Turn this into a function
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

        # TODO: If all elements are False or True raise ValueError. Actually if they are all the same raise an error.
        # if region is None:
        #     raise ValueError("...")
        self.region = region

        # # NOTE: Can this be made more elegantly?
        # if region is not None:
        #     self.uv_wavelengths = self.uv_wavelengths[region]
        #     self.visibilities = self.visibilities[region]
        #     self.noise_map = self.noise_map[region]
        #     self.noise_map_real_and_imag_averaged = self.noise_map_real_and_imag_averaged[region]
        #     self.uv_mask = self.uv_mask[region]
        #     self.uv_mask_real_and_imag_averaged = self.uv_mask_real_and_imag_averaged[region]


        # print(self.uv_wavelengths.shape)
        # print(self.visibilities.shape)
        # print(self.noise_map.shape)
        # print(self.noise_map_real_and_imag_averaged.shape)
        # print(self.uv_mask_real_and_imag_averaged.shape)
        # print("------")
        # #exit()

    @property
    def data(self):
        return self.visibilities

    @property
    def uv_mask_inside_region(self):
        if self.region is not None:
            return self.uv_mask[self.region]
        return self.uv_mask

    @property
    def uv_mask_outside_region(self):
        if self.region is not None:
            return self.uv_mask[~self.region]
        return self.uv_mask

    @property
    def visibilities_inside_region(self):
        if self.region is not None:
            return self.visibilities[self.region]
        return self.visibilities

    @property
    def visibilities_outside_region(self):
        if self.region is not None:
            return self.visibilities[~self.region]
        return self.visibilities

    @property
    def uv_wavelengths_inside_region(self):
        if self.region is not None:
            return self.uv_wavelengths[self.region]
        return self.uv_wavelengths

    @property
    def uv_wavelengths_outside_region(self):
        if self.region is not None:
            return self.uv_wavelengths[~self.region]
        return self.uv_wavelengths

    @property
    def noise_map_inside_region(self):
        if self.region is not None:
            return self.noise_map[self.region]
        return self.noise_map

    @property
    def noise_map_outside_region(self):
        if self.region is not None:
            return self.noise_map[~self.region]
        return self.noise_map

    @property
    def dataset_outside_region(self):
        return Dataset(
            uv_wavelengths=self.uv_wavelengths_outside_region,
            visibilities=self.visibilities_outside_region,
            noise_map=self.noise_map_outside_region,
            z_step_kms=self.z_step_kms
        )

    @property
    def dataset_inside_region(self):
        return Dataset(
            uv_wavelengths=self.uv_wavelengths_inside_region,
            visibilities=self.visibilities_inside_region,
            noise_map=self.noise_map_inside_region,
            z_step_kms=self.z_step_kms
        )


# NOTE: rename this cause this is something else in autolens ...
class RegionMaskedDataset:
    def __init__(self, dataset, continuum=False, uv_mask=None):

        self.dataset = dataset

        if isinstance(continuum, bool):
            self.continuum = continuum
        else:
            raise ValueError(
                "must be a boolean"
            )

        for name in [
            "visibilities",
            "uv_wavelengths",
            "noise_map"
        ]:

            array = getattr(self.dataset, name)
            if self.continuum:
                array = reshape_array(array=array)
            setattr(
                self,
                name,
                array
            )

        if uv_mask is None:
            self.uv_mask = np.full(
                shape=self.visibilities.shape,
                fill_value=False
            )
        else:
            if self.continuum:
                self.uv_mask = reshape_array(
                    array=uv_mask
                )
            else:
                self.uv_mask = uv_mask

        # TODO: ...
        self.uv_mask_real_and_imag_averaged = np.full(
            shape=self.uv_mask.shape[:-1],
            fill_value=False
        )

        self.noise_map_real_and_imag_averaged = np.average(
            a=self.noise_map, axis=-1
        )

        self.z_step_kms = dataset.z_step_kms


    @property
    def data(self):
        return self.visibilities
