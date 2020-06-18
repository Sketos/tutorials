import os
import sys
import numpy as np
import matplotlib.pyplot as plt

autolens_version = "0.45.0"
#autolens_version = "0.46.2"

config_path = "./config_{}".format(
    autolens_version
)
if os.environ["HOME"].startswith("/cosma"):
    cosma_server = "7"
    output_path = "{}/tutorials/autofit/tutorial_6/output".format(
        os.environ["COSMA{}_DATA_host".format(cosma_server)]
    )
else:
    output_path="./output"

import autofit as af
af.conf.instance = af.conf.Config(
    config_path=config_path,
    output_path=output_path
)
import autolens as al
if not (al.__version__ == autolens_version):
    raise ValueError("...")

from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.region.region import Region
from src.dataset.dataset import Dataset, MaskedDataset
from src.model import profiles, mass_profiles, dark_mass_profiles
from src.fit import fit
from src.phase import phase
from src.plot import fit_plots

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import string_utils as string_utils
import spectral_utils as spectral_utils
import plot_utils as plot_utils

import interferometry_utils.load_utils as interferometry_load_utils

import autolens_utils.autolens_plot_utils as autolens_plot_utils
import autolens_utils.autolens_tracer_utils as autolens_tracer_utils


lens_redshift = 0.5
source_redshift = 2.0

n_pixels = 100
pixel_scale = 0.05


class Image:
    def __init__(self, array_2d):
        self.array_2d = array_2d

    @property
    def in_1d_binned(self):
        return np.ndarray.flatten(self.array_2d)

    @property
    def in_2d_binned(self):
        return self.array_2d


def load_uv_wavelengths(filename):

    if not os.path.isfile(filename):
        raise IOError(
            "The file {} does not exist".format(filename)
        )

    u_wavelengths, v_wavelengths = interferometry_load_utils.load_uv_wavelengths_from_fits(
        filename=filename
    )

    uv_wavelengths = np.stack(
        arrays=(u_wavelengths, v_wavelengths),
        axis=-1
    )

    return uv_wavelengths


uv_wavelengths = load_uv_wavelengths(
    filename="{}/uv_wavelengths.fits".format(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    )
)

n_channels = uv_wavelengths.shape[0]
z_step_kms = 50.0 # NOTE: This does not correspond to the actual value of z_step_kms for this uv_wavelengths dataset



if __name__ == "__main__":

    transformer_class = al.TransformerFINUFFT

    grid_3d = Grid3D(
        grid_2d=al.Grid.uniform(
            shape_2d=(
                n_pixels,
                n_pixels
            ),
            pixel_scales=(
                pixel_scale,
                pixel_scale
            ),
            sub_size=1
        ),
        n_channels=n_channels
    )

    lens_mass_profile = mass_profiles.EllipticalPowerLaw(
        centre=(0.0, 0.0),
        axis_ratio=0.75,
        phi=45.0,
        einstein_radius=1.0,
        slope=2.0
    )

    subhalo_mass_profile = dark_mass_profiles.SphericalNFWMCRLudlow(
        centre=(-0.75, -0.5),
        mass_at_200=1e9,
        redshift_object=lens_redshift,
        redshift_source=source_redshift
    )

    source_model_1 = profiles.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.75,
        phi=45.0,
        intensity=0.00005,
        effective_radius=0.5,
        sersic_index=1.0,
    )

    source_model_2 = profiles.Kinematical(
        centre=(0.0, 0.0),
        z_centre=16.0,
        intensity=5.0,
        effective_radius=0.5,
        inclination=30.0,
        phi=50.0,
        turnover_radius=0.05,
        maximum_velocity=200.0,
        velocity_dispersion=50.0
    )

    cube_source_model_1 = source_model_1.profile_cube_from_grid(
        grid=grid_3d.grid_2d,
        shape_3d=grid_3d.shape_3d,
        z_step_kms=z_step_kms
    )
    cube_source_model_2 = source_model_2.profile_cube_from_grid(
        grid=grid_3d.grid_2d,
        shape_3d=grid_3d.shape_3d,
        z_step_kms=z_step_kms
    )

    # # NOTE: Plotting
    # spectrum_model_1 = spectral_utils.get_spectrum_from_cube(
    #     cube=cube_source_model_1
    # )
    # spectrum_model_2 = spectral_utils.get_spectrum_from_cube(
    #     cube=cube_source_model_2
    # )
    # plt.figure(
    #     figsize=(10, 5)
    # )
    # plt.plot(spectrum_model_1, color="r")
    # plt.plot(spectrum_model_2, color="b")
    # plt.show()
    # exit()

    cube = cube_source_model_1 + cube_source_model_2


    lensed_cube = autolens_tracer_utils.lensed_cube_from_tracer(
        tracer=al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=lens_redshift,
                    mass=lens_mass_profile,
                ),
                al.Galaxy(
                    redshift=lens_redshift,
                    mass=subhalo_mass_profile,
                ),
                al.Galaxy(
                    redshift=source_redshift,
                    light=al.lp.LightProfile()
                )
            ]
        ),
        grid=grid_3d.grid_2d,
        cube=cube
    )

    transformers = []
    for i in range(uv_wavelengths.shape[0]):
        transformer = transformer_class(
            uv_wavelengths=uv_wavelengths[i],
            grid=grid_3d.grid_2d.in_radians
        )
        transformers.append(transformer)

    visibilities = np.zeros(
        shape=uv_wavelengths.shape
    )
    for i in range(visibilities.shape[0]):
        visibilities[i] = transformers[i].visibilities_from_image(
                image=Image(
                    array_2d=lensed_cube[i]
                )
            )


    noise_map = np.random.normal(
        loc=0.0, scale=5.0 * 10**-1.0, size=visibilities.shape
    )
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=np.add(
            visibilities, noise_map
        ),
        noise_map=noise_map,
        z_step_kms=z_step_kms
    )
    # plot_utils.plot_cube(
    #     cube=autolens_plot_utils.dirty_cube_from_visibilities(
    #         visibilities=dataset.visibilities,
    #         transformers=transformers,
    #         shape=lensed_cube.shape
    #     ),
    #     ncols=8,
    #     cube_contours=lensed_cube,
    # )
    # exit()

    lens_model = af.PriorModel(mass_profiles.EllipticalPowerLaw)
    lens_model.centre_0 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    lens_model.centre_1 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    lens_model.einstein_radius = af.UniformPrior(
        lower_limit=0.85,
        upper_limit=1.25
    )
    lens_model.slope = 2.0

    source_model_1 = af.PriorModel(profiles.EllipticalSersic)
    source_model_1.centre_0 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    source_model_1.centre_1 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    source_model_1.intensity = af.LogUniformPrior(
        lower_limit=5.0 * 10**-6.0,
        upper_limit=5.0 * 10**-4.0
    )

    source_model_2 = af.PriorModel(profiles.Kinematical)
    source_model_2.centre_0 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    source_model_2.centre_1 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    source_model_2.z_centre = af.GaussianPrior(
        mean=16.0,
        sigma=2.0
    )
    source_model_2.intensity = af.LogUniformPrior(
        lower_limit=10**-2.0,
        upper_limit=10**+2.0
    )
    source_model_2.maximum_velocity = af.UniformPrior(
        lower_limit=25.0,
        upper_limit=400.0
    )
    source_model_2.velocity_dispersion = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=100.0
    )


    phase_folders = [
        string_utils.remove_substring_from_end_of_string(
            string=os.path.basename(__file__),
            substring=".py"
        )
    ]

    phase_1 = phase.Phase(
        phase_name="phase_1__version_{}".format(
            autolens_version
        ),
        phase_folders=phase_folders,
        profiles=af.CollectionPriorModel(
            lens=lens_model,
            source_1=source_model_1,
            source_2=source_model_2,
        ),
        lens_redshift=lens_redshift,
        source_redshift=source_redshift,
    )

    phase_1.optimizer.const_efficiency_mode = True
    phase_1.optimizer.n_live_points = 100
    phase_1.optimizer.sampling_efficiency = 0.2
    phase_1.optimizer.evidence_tolerance = 0.5

    xy_mask = Mask3D.unmasked(
        shape_3d=grid_3d.shape_3d,
        pixel_scales=grid_3d.pixel_scales,
        sub_size=grid_3d.sub_size,
    )

    phase_1.run(
        dataset=dataset,
        xy_mask=xy_mask
    )
