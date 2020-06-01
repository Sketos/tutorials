import autofit as af
import autolens as al

from autoastro import dimensions as dim


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


    
