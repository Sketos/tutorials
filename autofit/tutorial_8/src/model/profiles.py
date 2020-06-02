import autofit as af
import autolens as al

from autoastro import dimensions as dim

import numpy as np


def cube_from_image(image, shape_3d):

    # NOTE:
    if image.in_2d.shape != shape_3d[1:]:
        raise ValueError("...")

    return np.tile(
        A=image.in_2d, reps=(shape_3d[0], 1, 1)
    )


class EllipticalSersic(al.lp.EllipticalSersic):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 4.0,
    ):

        super(EllipticalSersic, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=phi,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )


    def profile_cube_from_grid(self, grid, shape_3d, z_step_kms, grid_radial_minimum=None):

        return cube_from_image(
            image=self.profile_image_from_grid(
                grid=grid,
                grid_radial_minimum=grid_radial_minimum
            ),
            shape_3d=shape_3d
        )


class Kinematical(al.lp.Kinematical):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        z_centre: float = 0.0,
        intensity: float = 0.1,
        effective_radius: float = 1.0,
        inclination: float = 0.0,
        phi: float = 50.0,
        turnover_radius: float = 0.0,
        maximum_velocity: float = 200.0,
        velocity_dispersion: float = 50.0,
    ):
        super(Kinematical, self).__init__(
            centre=centre,
            z_centre=z_centre,
            intensity=intensity,
            effective_radius=effective_radius,
            inclination=inclination,
            phi=phi,
            turnover_radius=turnover_radius,
            maximum_velocity=maximum_velocity,
            velocity_dispersion=velocity_dispersion,
        )
