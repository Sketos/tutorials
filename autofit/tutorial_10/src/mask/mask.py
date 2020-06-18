import autolens as al

import os
import sys
import numpy as np


sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import plot_utils as plot_utils


class Mask3D(np.ndarray):

    # noinspection PyUnusedLocal
    def __new__(
        cls, mask_2d, z_mask, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)
    ):

        mask_3d = np.tile(
            A=mask_2d, reps=(z_mask.shape[0], 1, 1)
        )
        # print(mask_3d)
        # plot_utils.plot_cube(cube=mask_3d.astype("int"), ncols=8)
        # #mask_3d[z_mask] = True
        # exit()
        mask_3d = mask_3d.astype("bool")
        obj = mask_3d.view(cls)
        obj.mask_2d = mask_2d
        obj.z_mask = z_mask
        obj.sub_size = sub_size
        obj.pixel_scales = pixel_scales
        obj.origin = origin
        return obj


    @property
    def n_channels(self):
        return self.shape[0]

    @property
    def shape_2d(self):
        return self.shape[1:]

    @classmethod
    def manual(
        cls, mask_2d, z_mask, invert=False
    ):

        # TODO: Check that mask_2d is an autolens mask

        if type(z_mask) is list:
            z_mask = np.asarray(z_mask).astype("bool")

        # TODO: There should be an invert_mask_2d and invert_z_mask
        # if invert:
        #     mask_3d = np.invert(mask_3d)

        return Mask3D(
            mask_2d=mask_2d,
            z_mask=z_mask,
            pixel_scales=mask_2d.pixel_scales,
            sub_size=mask_2d.sub_size,
            origin=mask_2d.origin
        )

    @classmethod
    def unmasked(
        cls, shape_3d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), invert=False
    ):

        return cls.manual(
            mask_2d=al.Mask.unmasked(
                shape_2d=shape_3d[1:],
                pixel_scales=pixel_scales,
                sub_size=sub_size,
                origin=origin
            ),
            z_mask=np.full(
                shape=shape_3d[0],
                fill_value=False
            ),
            invert=invert,
        )
