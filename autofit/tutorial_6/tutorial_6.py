import os
import sys

autolens_version = "0.45.0"

import autofit as af
af.conf.instance = af.conf.Config(
    config_path="./config_{}".format(
        autolens_version
    ),
    output_path="./output"
)
import autolens as al

from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
#from src.region.region import region
from src.dataset.dataset import Dataset, MaskedDataset
from src.model import profiles, mass_profiles
from src.fit import (
    fit
)
from src.phase import (
    phase as ph,
)

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import spectral_utils as spectral_utils
import plot_utils as plot_utils

import interferometry_utils.load_utils as interferometry_load_utils

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


def dirty_cube_from_visibilities(visibilities, transformers, shape, invert=True):

    if len(transformers) != visibilities.shape[0]:
        raise ValueError("...")

    dirty_cube = np.zeros(
        shape=shape
    )
    for i in range(visibilities.shape[0]):
        dirty_cube[i] = transformers[i].image_from_visibilities(
            visibilities=visibilities[i]
        )

    if invert:
        return dirty_cube[:, ::-1, :]
    return dirty_cube



uv_wavelengths = load_uv_wavelengths(
    filename="{}/uv_wavelengths.fits".format(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    )
)

n_channels = uv_wavelengths.shape[0]
z_step_kms = 50.0 # NOTE: This does not correspond to the actual value of z_step_kms for this uv_wavelengths dataset


def region(n, n_min, n_max, invert=False):

    mask = np.zeros(
        shape=int(n), dtype=int
    )

    if n_min > 0 and n_min < n_max and n_max < n:
        mask[n_min:n_max] = 1
    else:
        raise ValueError("...")

    return mask.astype(bool)


if __name__ == "__main__":

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

    transformers = []
    for i in range(uv_wavelengths.shape[0]):
        transformer = al.TransformerFINUFFT(
            uv_wavelengths=uv_wavelengths[i],
            grid=grid_3d.grid_2d.in_radians
        )
        transformers.append(transformer)

    xy_mask = Mask3D.unmasked(
        shape_3d=grid_3d.shape_3d,
        pixel_scales=grid_3d.pixel_scales,
        sub_size=grid_3d.sub_size,
    )

    lens_model = mass_profiles.EllipticalPowerLaw(
        centre=(0.0, 0.0),
        axis_ratio=0.75,
        phi=45.0,
        einstein_radius=1.0,
        slope=2.0
    )

    model_1 = profiles.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.75,
        phi=45.0,
        intensity=0.00005,
        effective_radius=0.5,
        sersic_index=1.0,
    )
    model_2 = profiles.Kinematical(
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

    cube_model_1 = model_1.profile_cube_from_grid(
        grid=grid_3d.grid_2d,
        shape_3d=grid_3d.shape_3d,
        z_step_kms=z_step_kms
    )
    cube_model_2 = model_2.profile_cube_from_grid(
        grid=grid_3d.grid_2d,
        shape_3d=grid_3d.shape_3d,
        z_step_kms=z_step_kms
    )
    cube = cube_model_1 + cube_model_2
    # plot_utils.plot_cube(
    #     cube=cube,
    #     ncols=8
    # )
    # exit()

    emission_line_region = region(
        n=n_channels,
        n_min=10,
        n_max=22
    )

    # # NOTE: TEST ...
    # cube_region = cube[emission_line_region]
    # plot_utils.plot_cube(
    #     cube=cube_region,
    #     ncols=8
    # )
    # exit()

    # # NOTE: Plotting
    # spectrum_model_1 = spectral_utils.get_spectrum_from_cube(
    #     cube=cube_model_1
    # )
    # spectrum_model_2 = spectral_utils.get_spectrum_from_cube(
    #     cube=cube_model_2
    # )
    # plt.figure(
    #     figsize=(10, 5)
    # )
    # plt.plot(spectrum_model_1, color="r")
    # plt.plot(spectrum_model_2, color="b")
    # plt.show()
    # exit()

    visibilities = np.zeros(
        shape=uv_wavelengths.shape
    )
    for i in range(visibilities.shape[0]):
        visibilities[i] = transformers[i].visibilities_from_image(
                image=Image(
                    array_2d=cube[i]
                )
            )
    # plot_utils.plot_cube(
    #     cube=dirty_cube_from_visibilities(
    #         visibilities=visibilities,
    #         transformers=transformers,
    #         shape=cube.shape
    #     ),
    #     ncols=8,
    #     cube_contours=cube,
    # )
    # exit()

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
    #     cube=dirty_cube_from_visibilities(
    #         visibilities=dataset.visibilities,
    #         transformers=transformers,
    #         shape=cube.shape,
    #         invert=True
    #     ),
    #     ncols=8,
    #     cube_contours=cube,
    #
    # )
    # exit()

    # NOTE: Plotting visibilities
    # for i in range(dataset.visibilities.shape[0]):
    #     plt.figure()
    #     plt.plot(
    #         dataset.visibilities[i, :, 0],
    #         dataset.visibilities[i, :, 1],
    #         linestyle="None",
    #         marker="o",
    #         color="black"
    #     )
    #     plt.show()
    # exit()

    masked_dataset = MaskedDataset(
        dataset=dataset,
        xy_mask=xy_mask
    )

    # fit_temp = fit.DatasetFit(
    #     masked_dataset=masked_dataset,
    #     model_data=visibilities
    # )
    # # plot_utils.plot_cube(
    # #     cube=dirty_cube_from_visibilities(
    # #         visibilities=fit_temp.residual_map,
    # #         transformers=transformers,
    # #         shape=cube.shape
    # #     ),
    # #     ncols=8
    # # )
    # # print("likelihood = ", fit_temp.likelihood)
    # # exit()

    lens_model_1 = af.PriorModel(mass_profiles.EllipticalPowerLaw)
    model_1 = af.PriorModel(profiles.EllipticalSersic)
    model_1.centre_0 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    model_1.centre_1 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    model_1.intensity = af.LogUniformPrior(
        lower_limit=5.0 * 10**-6.0,
        upper_limit=5.0 * 10**-4.0
    )
    model_2 = af.PriorModel(profiles.Kinematical)
    model_2.z_centre = af.GaussianPrior(
        mean=16.0,
        sigma=2.0
    )
    model_2.intensity = af.LogUniformPrior(
        lower_limit=10**-2.0,
        upper_limit=10**+2.0
    )

    phase_name = "phase_tutorial_6"
    os.system(
        "rm -r output/{}*".format(phase_name)
    )
    phase = ph.Phase(
        phase_name=phase_name,
        profiles=af.CollectionPriorModel(
            lens=lens_model_1,
            src_model_1=model_1,
            src_model_2=model_2
        ),
        lens_redshift=lens_redshift,
        source_redshift=source_redshift,
    )

    phase.optimizer.constant_efficiency = True
    phase.optimizer.n_live_points = 100
    phase.optimizer.sampling_efficiency = 0.5
    phase.optimizer.evidence_tolerance = 100.0

    phase.run(
        dataset=dataset,
        xy_mask=xy_mask
    )
