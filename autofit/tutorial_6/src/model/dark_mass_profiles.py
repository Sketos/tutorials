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

class SphericalNFWMCRLudlow(al.mp.SphericalNFWMCRLudlow):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):

        super(SphericalNFWMCRLudlow, self).__init__(
            centre=centre,
            mass_at_200=mass_at_200,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )
