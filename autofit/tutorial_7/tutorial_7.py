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

from autoarray.operators.inversion import inversions as inv

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

from scipy.interpolate import griddata
from astropy.io import fits

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import spectral_utils as spectral_utils
import plot_utils as plot_utils

import interferometry_utils.load_utils as interferometry_load_utils

import autolens_utils.autolens_tracer_utils as autolens_tracer_utils
import autolens_utils.autolens_plot_utils as autolens_plot_utils


lens_redshift = 0.5
source_redshift = 2.0

n_pixels = 100
pixel_scale = 0.05


def reshape_array(array):

    return array.reshape(
        -1,
        array.shape[-1]
    )


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

# TODO: Make this a class
def region(n, n_min, n_max, invert=False):

    mask = np.zeros(
        shape=int(n), dtype=int
    )

    if n_min > 0 and n_min < n_max and n_max < n:
        mask[n_min:n_max] = 1
    else:
        raise ValueError("...")

    return mask.astype(bool)



# a = np.array([[0,2], [1,1], [2,3], [3,1]])
# mask = np.full(
#     shape=a.shape,
#     fill_value=False
# )
# mask[0,1] = True
# print(a.shape)
# print(a)
# print(mask)
# #print(np.mean(a), np.average(a))
# exit()

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

    transformers = []
    for i in range(uv_wavelengths.shape[0]):
        transformer = transformer_class(
            uv_wavelengths=uv_wavelengths[i],
            grid=grid_3d.grid_2d.in_radians
        )
        transformers.append(transformer)

    xy_mask = Mask3D.unmasked(
        shape_3d=grid_3d.shape_3d,
        pixel_scales=grid_3d.pixel_scales,
        sub_size=grid_3d.sub_size,
    )
    # xy_mask = Mask3D.manual(
    #     mask_2d=al.Mask.circular(
    #         shape_2d=grid_3d.shape_2d,
    #         pixel_scales=grid_3d.pixel_scales,
    #         sub_size=grid_3d.sub_size,
    #         radius=2.5,
    #         centre=(0.0, 0.0)
    #     ),
    #     z_mask=np.full(
    #         shape=grid_3d.shape_3d[0],
    #         fill_value=False
    #     ),
    # )
    #exit()

    lens_mass_profile = mass_profiles.EllipticalPowerLaw(
        centre=(0.0, 0.0),
        axis_ratio=0.75,
        phi=45.0,
        einstein_radius=1.0,
        slope=2.0
    )

    src_model = profiles.Kinematical(
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
    cube = src_model.profile_cube_from_grid(
        grid=grid_3d.grid_2d,
        shape_3d=grid_3d.shape_3d,
        z_step_kms=z_step_kms
    )
    # plot_utils.plot_cube(
    #     cube=cube,
    #     ncols=8
    # )
    # exit()

    lensed_cube = autolens_tracer_utils.lensed_cube_from_tracer(
        tracer=al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=lens_redshift,
                    mass=lens_mass_profile,
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
    # plot_utils.plot_cube(
    #     cube=lensed_cube,
    #     ncols=8
    # )
    # exit()

    visibilities = np.zeros(
        shape=uv_wavelengths.shape
    )
    for i in range(visibilities.shape[0]):
        visibilities[i] = transformers[i].visibilities_from_image(
                image=Image(
                    array_2d=lensed_cube[i]
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
        loc=0.0, scale=2.0 * 10**-1.0, size=visibilities.shape
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
    #     cube_contours=lensed_cube,
    #
    # )
    # exit()

    line_emission_region = region(
        n=dataset.uv_wavelengths.shape[0],
        n_min=11,
        n_max=23,
        invert=False
    )

    masked_dataset = MaskedDataset(
        dataset=dataset,
        xy_mask=xy_mask,
    )
    #exit()



    pixelization_shape_0 = 20
    pixelization_shape_1 = 20
    regularization_coefficient = 10.0

    source_galaxy = al.Galaxy(
        redshift=source_redshift,
        pixelization=al.pix.VoronoiMagnification(
            shape=(pixelization_shape_0, pixelization_shape_1)
        ),
        regularization=al.reg.Constant(
            coefficient=regularization_coefficient
        ),
    )

    tracer_with_inversion = al.Tracer.from_galaxies(
        galaxies=[
            al.Galaxy(
                redshift=lens_redshift,
                mass=lens_mass_profile,
            ),
            source_galaxy
        ]
    )

    mappers_of_planes = tracer_with_inversion.mappers_of_planes_from_grid(
        grid=masked_dataset.grid_3d.grid_2d,
        inversion_uses_border=False,
        preload_sparse_grids_of_planes=None
    )
    mapper = mappers_of_planes[-1]

    n = 16


    prior_values = griddata(
        points=masked_dataset.grid_3d.grid_2d,
        values=np.ndarray.flatten(cube[n]),
        xi=mapper.pixelization_grid,
        method="linear",
    )
    #print(prior_values)
    # plt.figure()
    # plt.plot(mapper.pixelization_grid[:, 0], mapper.pixelization_grid[:, 1], linestyle="None", marker="o")
    # plt.show()
    # exit()

    # plt.figure()
    # autolens_plot_utils.draw_voronoi_pixels(
    #     mapper=mapper,
    #     values=prior_values,
    # )
    # plt.show()

    # plt.figure()
    # plt.imshow(cube[n])
    # plt.show()
    #exit()

    inversion = inv.InversionInterferometer.from_data_mapper_and_regularization(
        visibilities=dataset.visibilities[n],
        noise_map=dataset.noise_map[n],
        transformer=transformers[n],
        mapper=mapper,
        regularization=source_galaxy.regularization,
        prior_values=None
    )

    # #print(inversion.reconstruction)
    # #print("prior values:", prior_values)
    # plt.figure()
    # autolens_plot_utils.draw_voronoi_pixels(
    #     mapper=inversion.mapper,
    #     values=inversion.reconstruction - prior_values
    # )
    # plt.show()
    # exit()

    inversion_with_prior_values = inv.InversionInterferometer.from_data_mapper_and_regularization(
        visibilities=dataset.visibilities[n],
        noise_map=dataset.noise_map[n],
        transformer=transformers[n],
        mapper=mapper,
        regularization=source_galaxy.regularization,
        prior_values=prior_values
    )

    print(
        np.subtract(inversion.reconstruction, inversion_with_prior_values.reconstruction)
    )
    #exit()

    nrows = 1
    ncols = 5
    plt.figure()

    plt.subplot(
        nrows,
        ncols,
        1
    )
    autolens_plot_utils.draw_voronoi_pixels(
        mapper=inversion.mapper,
        values=inversion.reconstruction
    )
    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(
        nrows,
        ncols,
        2
    )
    autolens_plot_utils.draw_voronoi_pixels(
        mapper=inversion_with_prior_values.mapper,
        values=inversion_with_prior_values.reconstruction
    )
    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(
        nrows,
        ncols,
        3
    )
    autolens_plot_utils.draw_voronoi_pixels(
        mapper=inversion.mapper,
        values=inversion.reconstruction - inversion_with_prior_values.reconstruction
    )
    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(
        nrows,
        ncols,
        4
    )
    autolens_plot_utils.draw_voronoi_pixels(
        mapper=mapper,
        values=prior_values
    )
    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(
        nrows,
        ncols,
        5
    )
    plt.imshow(cube[n], cmap="jet", aspect="auto")
    plt.xticks([])
    plt.yticks([])

    plt.show()
    exit()

    lens = al.GalaxyModel(
        redshift=lens_redshift,
        mass=al.mp.EllipticalIsothermal,
    )

    source = al.GalaxyModel(
        redshift=source_redshift,
        pixelization=al.pix.VoronoiMagnification,
        regularization=al.reg.Constant,
    )

    source.pixelization.shape.shape_0 = af.UniformPrior(
        lower_limit=5, upper_limit=50
    )
    source.pixelization.shape.shape_1 = af.UniformPrior(
        lower_limit=5, upper_limit=50
    )

    #source.pixelization.shape = (15, 15)
    #source.regularization.coefficient = 1.0

    lens.mass.centre_0 = 0.0
    lens.mass.centre_1 = 0.0
    lens.mass.axis_ratio = 0.75
    lens.mass.phi = 45.0
    lens.mass.einstein_radius = 1.0
    lens.mass.slope = 2.0


    phase_name = "phase_tutorial_7"

    data_directory = "./data/{}".format(phase_name)
    if not os.path.isdir(data_directory):
        os.system(
            "mkdir {}".format(data_directory)
        )
    os.system(
        "rm ./data/{}/*.fits".format(phase_name)
    )
    for _ in ["uv_wavelengths", "visibilities", "noise_map"]:
        fits.writeto(
            "{}/{}.fits".format(
                data_directory, _
            ),
            data=getattr(dataset, _)
        )

    os.system(
        "rm -r ./output/{}*".format(phase_name)
    )
    phase = ph.Phase(
        phase_name=phase_name,
        galaxies=dict(
            lens=lens,
            source=source,
        ),
        transformer_class=transformer_class,
        region=line_emission_region
    )

    phase.optimizer.constant_efficiency = True
    phase.optimizer.n_live_points = 100
    phase.optimizer.sampling_efficiency = 0.5
    phase.optimizer.evidence_tolerance = 100.0

    phase.run(
        dataset=dataset,
        xy_mask=xy_mask
    )
