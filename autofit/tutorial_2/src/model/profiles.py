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


# class Kinematical(al.lp.LightProfile):
#     @af.map_types
#     def __init__(
#         self,
#         centre: dim.Position = (0.0, 0.0),
#         z_centre: float = 0.0,
#         intensity: float = 0.1,
#         effective_radius: float = 1.0,
#         inclination: float = 0.0,
#         phi: float = 50.0,
#         turnover_radius: float = 0.0,
#         maximum_velocity: float = 200.0,
#         velocity_dispersion: float = 50.0,
#     ):
#
#         super(Kinematical, self).__init__()
#
#         self.centre = centre
#         self.z_centre = z_centre
#         self.intensity = intensity
#         self.effective_radius = effective_radius
#         self.inclination = inclination
#         self.phi = phi
#         self.turnover_radius = turnover_radius
#         self.maximum_velocity = maximum_velocity
#         self.velocity_dispersion = velocity_dispersion
#
#
#
#
#     def convert_centre_from_arcsec_to_pixels(self, value, pixel_scale, n_pixels):
#
#         return value / pixel_scale + n_pixels / 2.0
#
#
#     def convert_radius_from_arcsec_to_pixels(self, value, pixel_scale):
#
#         return value / pixel_scale
#
#
#     # NOTE: grid should be replaced with grid_3d (working on it)
#     # NOTE: Maybe not the best way to do this ... (It is very fast the way it is)
#     def convert_parameters(self, grid):
#         #start = time.time()
#
#         #galpak = (x, y)
#         #autolens_centre = (y, x)
#
#         # NOTE: "converted_parameters" can also be a numpy array.
#         converted_parameters = []
#         for i, (name, value) in enumerate(self.__dict__.items()):
#             if name not in ["id", "_assertions", "cls"]:
#                 #print(name, value)
#                 if name == "centre":
#                     for (i, sign) in zip([1, 0], [1.0, -1.0]):
#                         converted_parameters.append(
#                             self.convert_centre_from_arcsec_to_pixels(
#                                 value=sign * value[i],
#                                 pixel_scale=grid.pixel_scale,
#                                 n_pixels=grid.shape_2d[i]
#                             )
#                         )
#
#                 elif name.endswith("radius"):
#                     converted_parameters.append(
#                         self.convert_radius_from_arcsec_to_pixels(
#                             value=value,
#                             pixel_scale=grid.pixel_scale
#                         )
#                     )
#                 else:
#                     converted_parameters.append(value)
#         # end = time.time()
#         # print(
#         #     "It took t={} to convert parameters".format(end - start)
#         # )
#
#         #print("parameters (converted):", converted_parameters)
#
#         return converted_parameters
#
#
#     # NOTE: grid should be replaced with grid_3d, where in this case shape_3d will be deprecated.
#     def profile_cube_from_grid(self, grid, shape_3d, z_step_kms):
#
#         # NOTE: I dont want to initialize this every time.
#         model = galpak.DiskModel(
#             flux_profile='exponential',
#             thickness_profile="gaussian",
#             rotation_curve='isothermal',
#             dispersion_profile="thick"
#         )
#
#         cube, _, _, _ = model._create_cube(
#             galaxy=galpak.GalaxyParameters.from_ndarray(
#                 a=self.convert_parameters(
#                     grid=grid
#                 )
#             ),
#             shape=shape_3d,
#             z_step_kms=z_step_kms,
#             zo=self.z_centre
#         )
#
#         # NOTE: The returned object should be
#         return cube.data
