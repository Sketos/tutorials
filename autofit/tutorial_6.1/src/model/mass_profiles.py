import autofit as af
import autolens as al

if al.__version__ in [
    "0.45.0"
]:
    from autoastro import dimensions as dim
if al.__version__ in [
    "0.46.0",
    "0.46.2"
]:
    from autogalaxy import dimensions as dim

import numpy as np


class EllipticalPowerLaw(al.mp.EllipticalPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        einstein_radius: dim.Length = 1.0,
        slope: float = 2.0,
    ):
        super(EllipticalPowerLaw, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=phi,
            einstein_radius=einstein_radius,
            slope=slope,
        )
