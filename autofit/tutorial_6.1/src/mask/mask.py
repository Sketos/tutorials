import numpy as np


class Mask3D(np.ndarray):

    # noinspection PyUnusedLocal
    def __new__(
        cls, mask_3d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)
    ):

        mask_3d = mask_3d.astype("bool")
        obj = mask_3d.view(cls)
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
        cls, mask_3d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), invert=False
    ):

        if type(mask_3d) is list:
            mask_3d = np.asarray(mask_3d).astype("bool")

        if invert:
            mask_3d = np.invert(mask_3d)

        if type(pixel_scales) is float:
            pixel_scales = (
                pixel_scales,
                pixel_scales
            )

        return Mask3D(
            mask_3d=mask_3d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin
        )


    @classmethod
    def unmasked(
        cls, shape_3d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), invert=False
    ):

        return cls.manual(
            mask_3d=np.full(
                shape=shape_3d,
                fill_value=False
            ),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            invert=invert,
        )


    @classmethod
    def from_mask_2d_and_z_mask(
        cls, mask_2d, z_mask, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), invert=False
    ):
        pass
