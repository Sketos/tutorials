import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy import units
from astropy.io import fits

autolens_version = "0.45.0"
#autolens_version = "0.46.2"

config_path = "./config_{}".format(
    autolens_version
)
if os.environ["HOME"].startswith("/cosma"):
    cosma_server = "7"
    output_path = "{}/tutorials/autofit/tutorial_0/output".format(
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
import autoarray as aa
if not (al.__version__ == autolens_version):
    raise ValueError("...")
import autolens.plot as aplt

from src.dataset.dataset import Dataset, MaskedDataset
from src.fit import fit
from src.phase import phase
from src.plot import fit_plots

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import string_utils as string_utils
import random_utils as random_utils
import casa_utils as casa_utils
import calibration_utils as calibration_utils
# import spectral_utils as spectral_utils
# import plot_utils as plot_utils

import autolens_utils.autolens_plot_utils as autolens_plot_utils
# import autolens_utils.autolens_tracer_utils as autolens_tracer_utils













lens_redshift = 0.5
source_redshift = 2.0

n_pixels = 100
pixel_scale = 0.05


n_channels = 32

frequencies = casa_utils.generate_frequencies(
    central_frequency=260.0 * units.GHz,
    n_channels=n_channels,
    bandwidth=2.0 * units.GHz
)

uv = fits.getdata("./uv.fits")

uv_wavelengths = casa_utils.convert_uv_coords_from_meters_to_wavelengths(
    uv=uv,
    frequencies=frequencies
)

uv_wavelengths = np.average(
    a=uv_wavelengths,
    axis=0
)

antennas = fits.getdata("./antennas.fits")
if not (uv_wavelengths.shape[0] == antennas.shape[0]):
    raise ValueError("...")

antennas_unique = np.unique(antennas)

if os.path.isfile("./phase_errors.fits"):
    phase_errors = fits.getdata(filename="./phase_errors.fits")
else:
    np.random.seed(
        seed=random_utils.seed_generator()
    )
    phase_errors = np.random.uniform(
        low=-np.pi/2.0, high=np.pi/2.0, size=(antennas_unique.size,)
    )
    fits.writeto("./phase_errors.fits", data=phase_errors)

f = calibration_utils.compute_f_matrix_from_antennas(
    antennas=antennas
)


# NOTE: DEVELOPING ...
# ========================================= #
antenna_distances = fits.getdata("antenna_distances.fits")

baselines = np.matmul(
    f.T, antenna_distances
)
# plt.show()
# plt.plot(baselines)
# plt.show()
baselines_reshaped = baselines.reshape(
    int(
        len(baselines) / (len(antenna_distances) * (len(antenna_distances) - 1))
    ),
    (len(antenna_distances) * (len(antenna_distances) - 1))
)
# plt.figure()
# plt.imshow(baselines_reshaped, aspect="auto")
# plt.show()

phase_difference = np.matmul(
    f.T, phase_errors
)
phase_difference_reshaped = baselines.reshape(
    int(
        len(phase_difference) / (len(phase_errors) * (len(phase_errors) - 1))
    ),
    (len(phase_errors) * (len(phase_errors) - 1))
)
# plt.figure()
# plt.imshow(phase_difference_reshaped, aspect="auto")
# plt.show()
# ========================================= #

data_with_phase_errors = True
self_calibration = True

if __name__ == "__main__":

    transformer_class = al.TransformerFINUFFT

    grid = al.Grid.uniform(
        shape_2d=(
            n_pixels,
            n_pixels
        ),
        pixel_scales=(
            pixel_scale,
            pixel_scale
        ),
        sub_size=1
    )


    lens = al.Galaxy(
        redshift=lens_redshift,
        mass=al.mp.EllipticalPowerLaw(
            centre=(-0.0, 0.0),
            axis_ratio=0.75,
            phi=45.0,
            einstein_radius=1.0,
            slope=2.0
        ),
    )

    subhalo = al.Galaxy(
        redshift=lens_redshift,
        mass=al.mp.SphericalNFWMCRLudlow(
            centre=(-0.75, 0.15),
            mass_at_200=1e9,
            redshift_object=lens_redshift,
            redshift_source=source_redshift
        ),
    )

    source = al.Galaxy(
        redshift=source_redshift,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.75,
            phi=45.0,
            intensity=0.001,
            effective_radius=0.5,
            sersic_index=1.0,
        ),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[
            lens,
            source
        ]
    )

    # aplt.Tracer.profile_image(
    #     tracer=tracer,
    #     grid=grid
    # )
    # exit()

    # lensed_image = tracer.profile_image_from_grid(
    #     grid=grid
    # )
    # plt.figure()
    # plt.imshow(lensed_image.in_2d)
    # plt.show()
    # exit()



    transformer = transformer_class(
        uv_wavelengths=uv_wavelengths,
        grid=grid.in_radians
    )

    visibilities = tracer.profile_visibilities_from_grid_and_transformer(
        grid=grid,
        transformer=transformer
    )

    def corrupt_visibilities_from_f_matrix_and_phase_errors(visibilities, f, phase_errors):

        phases_corrupted = np.add(
            visibilities.phases,
            np.matmul(
                f.T, phase_errors
            )
        )

        return aa.structures.visibilities.Visibilities(
            visibilities_1d=np.stack(
                arrays=(
                    visibilities.amplitudes * np.cos(phases_corrupted),
                    visibilities.amplitudes * np.sin(phases_corrupted)
                ),
                axis=-1
            )
        )

    if data_with_phase_errors:
        visibilities = corrupt_visibilities_from_f_matrix_and_phase_errors(
            visibilities=visibilities,
            f=f,
            phase_errors=phase_errors
        )


    noise_map = np.random.normal(
        loc=0.0, scale=1.0 * 10**-2.0, size=visibilities.shape
    )

    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=np.add(
            visibilities,
            noise_map
        ),
        noise_map=noise_map,
        antennas=antennas
    )
    # plt.figure()
    # plt.plot(
    #     dataset.visibilities[:, 0],
    #     dataset.visibilities[:, 1],
    #     linestyle="None",
    #     marker="o",
    #     markersize=10,
    #     color="black"
    # )
    # plt.show()
    # exit()

    # NOTE: DELETE ...
    # phase_difference = dataset.visibilities.phases - visibilities.phases
    # phase_difference_reshaped = baselines.reshape(
    #     int(
    #         len(phase_difference) / (len(phase_errors) * (len(phase_errors) - 1))
    #     ),
    #     (len(phase_errors) * (len(phase_errors) - 1))
    # )
    # # plt.figure()
    # # plt.imshow(phase_difference_reshaped, aspect="auto", cmap="jet")
    # # plt.show()
    # plt.plot(phase_difference_reshaped[:, 0])
    # plt.show()
    # exit()

    # NOTE: Plot
    # autolens_plot_utils.plot_dirty_image_from_visibilities_and_transformer(
    #     visibilities=dataset.visibilities,
    #     transformer=transformer
    # )
    # exit()

    xy_mask = al.Mask.unmasked(
        shape_2d=grid.shape_2d,
        pixel_scales=grid.pixel_scales,
        sub_size=grid.sub_size,
    )

    def test(dataset, xy_mask, tracer, self_calibration, transformer_class=al.TransformerFINUFFT):

        masked_dataset = MaskedDataset(
            dataset=dataset,
            xy_mask=xy_mask
        )

        grid = masked_dataset.grid

        transformer = transformer_class(
            uv_wavelengths=masked_dataset.uv_wavelengths,
            grid=grid.in_radians
        )

        model_data = tracer.profile_visibilities_from_grid_and_transformer(
            grid=grid,
            transformer=transformer
        )

        if self_calibration:
            phase_errors = calibration_utils.phase_errors_from_A_and_B_matrices(
                phases=masked_dataset.phases,
                model_phases=model_data.phases,
                A=masked_dataset.A,
                B=masked_dataset.B
            )

            model_phases_corrected = np.add(
                model_data.phases,
                np.matmul(
                    masked_dataset.f.T,
                    phase_errors
                )
            )

            model_data = aa.structures.visibilities.Visibilities(
                visibilities_1d=np.stack(
                    arrays=(
                        model_data.amplitudes * np.cos(model_phases_corrected),
                        model_data.amplitudes * np.sin(model_phases_corrected)
                    ),
                    axis=-1
                )
            )

        fit_temp = fit.DatasetFit(
            masked_dataset=masked_dataset,
            model_data=model_data
        )

        # print("likelihood = ", fit_temp.likelihood)
        #
        # fit_plots.residual_map(
        #     fit=fit_temp,
        #     transformer=transformer,
        #     output_filename=None,
        #     output_path=None,
        #     output_format="show",
        # )

        return fit_temp.likelihood

    axis_ratio_min = 0.7475
    axis_ratio_max = 0.7525
    axis_ratio_arr = np.linspace(axis_ratio_min, axis_ratio_max, 100)

    likelihoods = []
    for axis_ratio in axis_ratio_arr:
        print(axis_ratio)
        likelihood = test(
            dataset=dataset,
            xy_mask=xy_mask,
            tracer=al.Tracer.from_galaxies(
                galaxies=[
                    al.Galaxy(
                        redshift=lens_redshift,
                        mass=al.mp.EllipticalPowerLaw(
                            centre=(-0.0, 0.0),
                            axis_ratio=axis_ratio,
                            phi=45.0,
                            einstein_radius=1.0,
                            slope=2.0
                        ),
                    ),
                    source
                ]
            ),
            self_calibration=self_calibration
        )
        likelihoods.append(likelihood)
    plt.figure()
    plt.plot(axis_ratio_arr, likelihoods, color="black")
    plt.axvline(lens.mass.axis_ratio, linestyle="--", color="r")
    plt.axvline(axis_ratio_arr[np.argmax(likelihoods)], linestyle="--", color="b")

    plt.show()
    exit()

    """
    test(
        dataset=dataset,
        xy_mask=xy_mask,
        tracer=al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=lens_redshift,
                    mass=al.mp.EllipticalPowerLaw(
                        centre=(-0.0, 0.0),
                        axis_ratio=0.741,
                        phi=45.0,
                        einstein_radius=1.0,
                        slope=2.0
                    ),
                ),
                source
            ]
        ),
        self_calibration=self_calibration
    )
    """
    exit()



    """
    model_visibilities = tracer.profile_visibilities_from_grid_and_transformer(
        grid=grid,
        transformer=transformer
    )
    model_phases = model_visibilities.phases
    print(model_phases)

    phase_errors_derived = calibration_utils.phase_errors_from_A_and_B_matrices(
        phases=dataset.phases,
        model_phases=model_visibilities.phases,
        A=dataset.A,
        B=dataset.B
    )

    offset = phase_errors[0] - phase_errors_derived[0]

    plt.figure()
    plt.plot(phase_errors, color="black", linewidth=4, alpha=0.75, label="input")
    plt.plot(phase_errors_derived + offset, color="r", linewidth=1, label="recovered")
    #plt.plot(phase_errors - (phase_errors_solution + offset), color="b", linewidth=1, label="difference")
    plt.xlabel("# of antenna", fontsize=15)
    plt.ylabel("Phase (rad)", fontsize=15)
    plt.legend()
    plt.show()
    exit()
    """

    phases = dataset.phases
    model_data = tracer.profile_visibilities_from_grid_and_transformer(
        grid=grid,
        transformer=transformer
    )
    model_phases = model_data.phases

    # NOTE: TEST: ... delete afterwards
    # phase_errors_temp = np.linalg.solve(
    #     dataset.A,
    #     np.matmul(
    #         dataset.B,
    #         calibration_utils.wrap(
    #             a=np.subtract(
    #                 phases,
    #                 model_phases
    #             )
    #         )
    #     )
    # )
    # plt.plot(phase_errors, color="black", linewidth=4, alpha=0.75, label="input")
    # plt.plot(phase_errors_temp, color="r", linewidth=2, alpha=0.75, label="input")
    # plt.show()
    # exit()

    dt = 120 # sec

    time = fits.getdata(filename="./time.fits")
    time -= time[0]



    n_segments = np.zeros(shape=time.shape)

    time_i = 0.0
    n = 0
    while True:
        time_f = time_i + dt

        idx = np.logical_and(
            time >= time_i,
            time < time_f,
        )

        time_i = time_f

        if not any(idx):
            break
        else:
            n_segments[idx] = n
            n += 1




    # def block_diag_inverse(A, M):
    #     return np.array([
    #         A[i*M:(i+1)*M, i*M:(i+1)*M]
    #         for i in range(int(A.shape[0] / M))
    #     ])
    # print(dataset.C.shape, len(np.unique(n_segments)));exit()
    # C_blocks = block_diag_inverse(A=dataset.C, M=dataset.C.shape[0] / len(np.unique(n_segments)))
    # print(C_blocks.shape)

    def split_block_diagonal_matrix(A, n):

        N = int(A.shape[0] / n)
        print(n, N)
        return [
            A[i*N:(i+1)*N, i*N:(i+1)*N]
            for i in range(n)
        ]




    # print(dataset.f.shape)
    # print(dataset.A.shape)
    # print(dataset.B.shape)
    # exit()
    f_blocks = []
    A_blocks = []
    B_blocks = []
    phase_errors_blocks = []
    for n, C_block in enumerate(
        split_block_diagonal_matrix(
            A=dataset.C, n=len(np.unique(n_segments))
        )
    ):
        print(n)

        idx = np.squeeze(
            np.where(n_segments==n)
        )

        arr = dataset.visibilities[n_segments==n]
        #print(arr.shape)

        # f = calibration_utils.compute_f_matrix_from_antennas(
        #     antennas=self.antennas[idx]
        # )
        f_blocks.append(dataset.f[:, idx])

        A_block = calibration_utils.compute_A_matrix_from_f_and_C_matrices(
            f=f[:, idx], C=C_block
        )
        #print(A.shape)
        B_block = calibration_utils.compute_B_matrix_from_f_and_C_matrices(
            f=f[:, idx], C=C_block
        )
        #print(B.shape)

        phase_difference = calibration_utils.wrap(
            a=np.subtract(
                phases[idx],
                model_phases[idx]
            )
        )
        #print(B.shape, phase_difference.shape)



        B_block = np.matmul(
            B_block,
            phase_difference
        )

        A_blocks.append(A_block)
        B_blocks.append(B_block)

        print(A_block.shape, B_block.shape)
        phase_errors_block = np.linalg.solve(A_block, B_block)
        phase_errors_blocks.append(phase_errors_block)


    # plt.plot(phase_errors, color="black", linewidth=4, alpha=0.75, label="input")
    # for phase_errors_block in phase_errors_blocks:
    #     offset = phase_errors[0] - phase_errors_block[0]
    #     plt.plot(phase_errors_block + offset, linewidth=2, alpha=0.75)
    # plt.show()
    # exit()


    from scipy.linalg import block_diag
    A = block_diag((*A_blocks))
    #B = block_diag((*B_blocks))
    #B = B_blocks.flatten()

    B = []
    for block in B_blocks:
        B.extend(block)
    B = np.asarray(B)
    print(B)
    print(phases.shape, model_phases.shape, A.shape, B.shape)


    phase_errors_block = np.linalg.solve(A,B)


    # exit()
    # phase_errors_block=calibration_utils.phase_errors_from_A_and_B_matrices(
    #     phases=phases,
    #     model_phases=model_phases,
    #     A=A,
    #     B=B
    # )
    # #print(phase_errors_block.shape)
    # #print(phase_errors_block.reshape(43, int(phase_errors_block.shape[0] / 43)))




    phase_errors_block_reshaped = phase_errors_block.reshape(
        len(np.unique(n_segments)),
        int(phase_errors_block.shape[0] / len(np.unique(n_segments)))
    )
    plt.figure()
    plt.plot(phase_errors, color="black", linewidth=4, alpha=0.75, label="input")
    for i in range(phase_errors_block_reshaped.shape[0]):
        offset = phase_errors[0] - phase_errors_block_reshaped[i, 0]
        plt.plot(phase_errors_block_reshaped[i, :] + offset)


    plt.show()

    exit()


    lens = al.GalaxyModel(
        redshift=lens_redshift,
        mass=al.mp.EllipticalPowerLaw,
    )
    lens.mass.centre_0 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    lens.mass.centre_1 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    lens.mass.einstein_radius = af.UniformPrior(
        lower_limit=0.85,
        upper_limit=1.25
    )
    lens.mass.slope = 2.0

    source = al.GalaxyModel(
        redshift=source_redshift,
        light=al.lp.EllipticalSersic,
    )
    source.light.centre_0 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    source.light.centre_1 = af.GaussianPrior(
        mean=0.0,
        sigma=0.25
    )
    source.light.intensity = af.LogUniformPrior(
        lower_limit=1.0 * 10**-4.0,
        upper_limit=1.0 * 10**-2.0
    )

    lens.mass.centre_0 = 0.0
    lens.mass.centre_1 = 0.0
    #lens.mass.axis_ratio = 0.75
    lens.mass.phi = 45.0
    lens.mass.einstein_radius = 1.0
    lens.mass.slope = 2.0

    source.light.centre_0 = 0.0
    source.light.centre_1 = 0.0
    source.light.axis_ratio = 0.75
    source.light.phi = 45.0
    source.light.intensity = 0.001
    source.light.effective_radius = 0.5
    source.light.sersic_index = 1.0

    phase_folders = []

    phase_folders.append(
        "data_{}_phase_errors".format(
            "with" if data_with_phase_errors else "without"
        )
    )
    phase_folders.append(
        "model_{}_self_calibration".format(
            "with" if self_calibration else "without"
        )
    )

    #evidence_tolerance = 0.5
    #evidence_tolerance = 0.8
    evidence_tolerance = 100.0
    phase_folders.append(
        "evidence_tolerance__{}".format(evidence_tolerance)
    )

    phase_1_name = "phase_tutorial_0__version_{}".format(autolens_version)
    # os.system(
    #     "rm -r output/{}".format(phase_1_name)
    # )
    phase_1 = phase.Phase(
        phase_name=phase_1_name,
        phase_folders=phase_folders,
        galaxies=dict(
            lens=lens,
            source=source,
        ),
        self_calibration=self_calibration,
    )

    phase_1.optimizer.const_efficiency_mode = True
    phase_1.optimizer.n_live_points = 100
    phase_1.optimizer.sampling_efficiency = 0.2
    phase_1.optimizer.evidence_tolerance = evidence_tolerance


    phase_1.run(
        dataset=dataset,
        xy_mask=xy_mask
    )
