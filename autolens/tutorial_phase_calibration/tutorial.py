import os
import sys
import time

autolens_version = "0.45.0"

config_path = "./config_{}".format(
    autolens_version
)
if os.environ["HOME"].startswith("/cosma"):
    cosma_server = "7"
    output_path = "{}/tutorials/autofit/tutorial_8.1/output".format(
        os.environ["COSMA{}_DATA_host".format(cosma_server)]
    )
else:
    output_path="./output"

import autofit as af
af.conf.instance = af.conf.Config(
    config_path=config_path,
    output_path=output_path
)
import autoarray as aa
import autolens as al
if not (al.__version__ == autolens_version):
    raise ValueError("...")


import numpy as np
import matplotlib.pyplot as plt
from astropy import units, constants
from astropy.io import fits

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import random_utils as random_utils
import array_utils as array_utils
import casa_utils as casa_utils
import spectral_utils as spectral_utils

n_channels = 32

frequencies = casa_utils.generate_frequencies(
    central_frequency=260.0 * units.GHz,
    n_channels=n_channels,
    bandwidth=2.0 * units.GHz
)



def convert_uv_coords_from_meters_to_wavelengths(uv, frequencies):

    def convert_array_to_wavelengths(array, frequency):

        array_converted = (
            (array * units.m) * (frequency * units.Hz) / constants.c
        ).decompose()

        return array_converted.value

    if np.shape(frequencies):

        u_wavelengths, v_wavelengths = np.zeros(
            shape=(
                2,
                len(frequencies),
                uv.shape[1]
            )
        )

        for i in range(len(frequencies)):
            u_wavelengths[i, :] = convert_array_to_wavelengths(array=uv[0, :], frequency=frequencies[i])
            v_wavelengths[i, :] = convert_array_to_wavelengths(array=uv[1, :], frequency=frequencies[i])

    else:

        u_wavelengths = convert_array_to_wavelengths(array=uv[0, :], frequency=frequencies)
        v_wavelengths = convert_array_to_wavelengths(array=uv[1, :], frequency=frequencies)

    return np.stack(
        arrays=(u_wavelengths, v_wavelengths),
        axis=-1
    )

transformer_class = al.TransformerFINUFFT

lens_redshift = 0.5
source_redshift = 2.0

n_pixels = 100
pixel_scale = 0.05

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

# np.tile(
#     A=image.in_2d, reps=(shape_3d[0], 1, 1)
# )

if __name__ == "__main__":

    uv = fits.getdata("./uv.fits")

    # plt.figure()
    # plt.plot(uv[0], uv[1], linestyle='None', marker="o")
    # plt.show()
    # exit()

    uv_wavelengths = convert_uv_coords_from_meters_to_wavelengths(
        uv=uv,
        frequencies=frequencies
    )



    uv_wavelengths = uv_wavelengths[0]

    transformer = transformer_class(
        uv_wavelengths=uv_wavelengths,
        grid=grid.in_radians
    )

    # transformer = transformer_class(
    #     uv_wavelengths=array_utils.reshape_array(
    #         array=uv_wavelengths
    #     ),
    #     grid=grid.in_radians
    # )



    lens = al.Galaxy(
        redshift=lens_redshift,
        mass=al.mp.EllipticalPowerLaw(
            centre=(0.0, 0.5),
            axis_ratio=0.75,
            phi=45.0,
            einstein_radius=1.0,
            slope=2.0
        )
    )

    source = al.Galaxy(
        redshift=source_redshift,
        light=al.lp.EllipticalSersic(
            centre=(0.25, -0.25),
            axis_ratio=0.75,
            phi=45.0,
            intensity=0.001,
            effective_radius=0.5,
            sersic_index=1.0,
        )
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[
            lens,
            source
        ]
    )

    # lensed_image = tracer.profile_image_from_grid(
    #     grid=grid
    # )
    # plt.figure()
    # plt.imshow(lensed_image.in_2d)
    # plt.show()
    # exit()

    visibilities = tracer.profile_visibilities_from_grid_and_transformer(
        grid=grid,
        transformer=transformer
    )

    # plt.figure()
    # plt.plot(visibilities[:, 0], visibilities[:, 1], linestyle="None", marker="o", color="black")
    # plt.show()
    # exit()

    # NOTE: ...
    # dirty_lensed_image = transformer.image_from_visibilities(
    #     visibilities=visibilities
    # )
    # plt.figure()
    # plt.imshow(dirty_lensed_image)
    # plt.show()
    # exit()

    amplitude = np.hypot(visibilities[:, 0], visibilities[:, 1])
    phase = np.arctan2(visibilities[:, 1], visibilities[:, 0])

    antennas = fits.getdata("./antennas.fits")
    #print(uv_wavelengths.shape)
    #print(antennas.shape)

    # antennas = array_utils.reshape_array(
    #     array=np.tile(
    #         A=antennas, reps=(len(frequencies), 1, 1)
    #     )
    # )
    # #exit()

    #print(antennas[:, 0])
    #print(antennas[:, 1])

    antennas_unique = np.unique(antennas)

    np.random.seed(
        seed=random_utils.seed_generator()
    )
    phase_errors = np.random.uniform(low=-np.pi/4.0, high=np.pi/4.0, size=(antennas_unique.size,))
    # print(phase_errors)
    # exit()



    # figure, axes = plt.subplots(nrows=1, ncols=2)
    # axes[0].plot(np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1]) * 10**3.0, amplitude, linestyle="None", marker="o", markersize=10, color="black")
    # axes[1].plot(np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1]) * 10**3.0, phase, linestyle="None", marker="o", markersize=10, color="black")
    # axes[0].set_xscale("log")
    # axes[1].set_xscale("log")
    # plt.show()
    # exit()

    # dPhi_dphi = np.zeros((len(antennas_unique) - 1, antennas.shape[0]))
    # for j in range(1, len(antennas_unique)):
    #     dPhi_dphi[j-1,:] = (antennas[:, 0]==antennas_unique[j])-1*(antennas[:, 1]==antennas_unique[j])

    dPhi_dphi = np.zeros((len(antennas_unique), antennas.shape[0]))
    for j in range(0, len(antennas_unique)):
        dPhi_dphi[j, :] = (antennas[:, 0]==antennas_unique[j])-1*(antennas[:, 1]==antennas_unique[j])

    # plt.figure()
    # plt.imshow(dPhi_dphi, aspect="auto")
    # plt.show()

    # phases_corrupted = np.add(
    #     visibilities.phases,
    #     np.matmul(
    #         dPhi_dphi.T, phase_errors
    #     )
    # )
    #
    # visibilities_corrupted = aa.structures.visibilities.Visibilities(
    #     visibilities_1d=np.stack(
    #         arrays=(
    #             visibilities.amplitudes * np.cos(phases_corrupted),
    #             visibilities.amplitudes * np.sin(phases_corrupted)
    #         ),
    #         axis=-1
    #     )
    # )

    visibilities_corrupted_temp = np.array(amplitude * np.exp(1j * (phase + np.matmul(dPhi_dphi.T, phase_errors))))

    visibilities_corrupted = np.stack(
        arrays=(visibilities_corrupted_temp.real, visibilities_corrupted_temp.imag),
        axis=-1
    )

    # amplitude_corrupted = np.hypot(visibilities_corrupted[:, 0], visibilities_corrupted[:, 1])
    # phase_corrupted = np.arctan2(visibilities_corrupted[:, 1], visibilities_corrupted[:, 0])
    #
    # plt.figure()
    # plt.plot(np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1]) * 10**3.0, amplitude_corrupted - amplitude, linestyle="None", marker="o", markersize=10, color="black")
    # plt.xscale("log")
    # plt.show()
    # exit()

    sigma = np.random.normal(
        loc=0.0, scale=1.0 * 10**-3.0, size=visibilities.shape
    )
    visibilities_corrupted = np.add(
        visibilities_corrupted,
        sigma
    )
    # plt.figure()
    # plt.plot(
    #     visibilities_corrupted[:, 0],
    #     visibilities_corrupted[:, 1],
    #     linestyle="None",
    #     marker="o",
    #     markersize=10,
    #     color="black"
    # )
    # plt.plot(
    #     visibilities[:, 0],
    #     visibilities[:, 1],
    #     linestyle="None",
    #     marker="o",
    #     markersize=5,
    #     color="r"
    # )
    # plt.show()
    # exit()

    amplitude_corrupted = np.hypot(visibilities_corrupted[:, 0], visibilities_corrupted[:, 1])
    phase_corrupted = np.arctan2(visibilities_corrupted[:, 1], visibilities_corrupted[:, 0])

    def corrupt_visibilities_with_phase_errors(visibilities, phase_errors):
        pass


    # phase_corrupted_temp = phase + np.dot(dPhi_dphi.T, dphi)
    # phase_corrupted_temp = (phase_corrupted + np.pi) % (2 * np.pi) - np.pi
    #
    # plt.plot(np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1]) * 10**3.0, phase, linestyle="None", marker="o", markersize=10, color="black")
    # plt.plot(np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1]) * 10**3.0, phase_corrupted, linestyle="None", marker="o", markersize=10, color="r", alpha=0.5)
    # plt.plot(np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1]) * 10**3.0, phase_corrupted - phase, linestyle="None", marker="o", markersize=10, color="b", alpha=0.5)
    # plt.xscale("log")
    # plt.show()
    # exit()

    # idx = np.logical_and(antennas[:, 0]==10, antennas[:, 1]==11)
    # plt.plot(np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1])[idx] * 10**3.0, phase_corrupted[idx] - phase[idx], linestyle="None", marker="o", markersize=10, color="r", alpha=0.5)
    # plt.xscale("log")
    # plt.show()
    # exit()

    # import scipy.sparse
    #
    # C = scipy.sparse.diags((sigma / amplitude_corrupted)**-2.,0)
    # #C = scipy.sparse.diags((np.average(sigma, axis=-1) / amplitude_corrupted)**-2.,0)
    # #C = scipy.sparse.diags(np.ones(shape=amplitude_corrupted.shape), 0)
    # C = C.toarray()

    #C = np.diag((sigma / amplitude_corrupted)**-2.)

    # dPhi_dphi = np.zeros((len(antennas_unique) - 1, antennas.shape[0]))
    # for j in range(0, len(antennas_unique)):
    #     dPhi_dphi[j - 1, :] = (antennas[:, 0]==antennas_unique[j])-1*(antennas[:, 1]==antennas_unique[j])

    # C = np.diag(
    #     np.power(
    #         np.divide(
    #             np.average(a=sigma, axis=-1),
    #             amplitude_corrupted,
    #         ),
    #         -2
    #     )
    # )

    C = np.diag(
        np.power(
            np.divide(
                np.sqrt(
                    np.add(
                        visibilities_corrupted[:, 0]**2.0 * sigma[:, 0]**2.0,
                        visibilities_corrupted[:, 1]**2.0 * sigma[:, 1]**2.0
                    )
                ),
                amplitude_corrupted**2.0,
            ),
            -2
        )
    )

    F = np.matmul(
        dPhi_dphi,
        np.matmul(C, dPhi_dphi.T)
    )
    Finv = np.linalg.inv(F)

    FdPC = np.matmul(
        -Finv,
        np.matmul(dPhi_dphi, C)
    )

    phase_difference = phase_corrupted - visibilities.phases
    phase_difference = (phase_difference + np.pi) % (2 * np.pi) - np.pi

    #phase_errors_temp = np.dot(FdPC, phase_difference)

    #phase_corrupted_corrected += np.dot(dPhi_dphi.T,phase_errors_temp)

    # B = np.matmul(
    #     np.matmul(dPhi_dphi, C),
    #     phase_difference
    # )
    # exit()

    phase_errors_solution = np.linalg.solve(
        F,
        np.matmul(
            np.matmul(dPhi_dphi, C),
            phase_difference
        )
    )

    # offset = phase_errors[0] - phase_errors_solution[0]
    #
    # plt.figure()
    # plt.plot(phase_errors, color="black", linewidth=4, alpha=0.75, label="input")
    # plt.plot(phase_errors_solution + offset, color="r", linewidth=1, label="recovered")
    # plt.plot(phase_errors - (phase_errors_solution + offset), color="b", linewidth=1, label="difference")
    # plt.xlabel("# of antenna", fontsize=15)
    # plt.ylabel("Phase (rad)", fontsize=15)
    # plt.legend()
    # plt.show()



    # phase_errors = phase_errors[1:]
    # offset = phase_errors[0] - phase_errors_solution[0]
    #
    # plt.figure()
    # plt.plot(phase_errors, color="black", linewidth=4, alpha=0.75, label="input")
    # plt.plot(phase_errors_solution + offset, color="r", linewidth=1, label="recovered")
    # plt.xlabel("# of antenna", fontsize=15)
    # plt.ylabel("Phase (rad)", fontsize=15)
    # plt.legend()
    # plt.show()


    phases_corrected = np.add(
        visibilities.phases,
        np.matmul(
            dPhi_dphi.T, phase_errors_solution
        )
    )

    visibilities_corrected = aa.structures.visibilities.Visibilities(
        visibilities_1d=np.stack(
            arrays=(
                visibilities.amplitudes * np.cos(phases_corrected),
                visibilities.amplitudes * np.sin(phases_corrected)
            ),
            axis=-1
        )
    )

    plt.figure()
    plt.plot(
        visibilities_corrupted[:, 0],
        visibilities_corrupted[:, 1],
        linestyle="None",
        marker="o",
        markersize=10,
        color="black"
    )
    plt.plot(
        visibilities[:, 0],
        visibilities[:, 1],
        linestyle="None",
        marker="o",
        markersize=5,
        color="r"
    )
    plt.plot(
        visibilities_corrected[:, 0],
        visibilities_corrected[:, 1],
        linestyle="None",
        marker="o",
        markersize=5,
        color="b"
    )
    plt.show()
    exit()

    #plt.plot(np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1]) * 10**3.0, phase, linestyle="None", marker="o", markersize=10, color="black")
    #plt.plot(np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1]) * 10**3.0, phase_corrupted_corrected, linestyle="None", marker="o", markersize=10, color="r", alpha=0.5)
