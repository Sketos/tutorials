import autofit as af
import autolens as al

from autoastro import dimensions as dim

class EllipticalGaussian(al.lp.EllipticalGaussian):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        intensity: dim.Luminosity = 0.1,
        sigma: dim.Length = 0.01,
    ):

        super(EllipticalGaussian, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=phi,
            intensity=intensity,
            sigma=sigma
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
