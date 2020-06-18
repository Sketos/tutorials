# NOTE : Run this script a few times ... The agreement of the recovered phase offsets varies ... WEIRD!!!
# NOTE : The equations that this script is based on can be found in the appendix of Hezaveh et al. (2013; https://iopscience.iop.org/article/10.1088/0004-637X/767/2/132/pdf)


import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units, constants
from astropy.io import fits


def perturbative_self_calibration(
    visibilities,
    model_visibilities,
    noise_map,
    antennas,
    dPhi_dphi=None,
    C=None
):

    if visibilities.shape == model_visibilities.shape:
        amplitude_visibilities = np.sqrt(visibilities[:, 0]**2.0 + visibilities[:, 1]**2.0)
        phase_visibilities = np.arctan2(visibilities[:, 1], visibilities[:, 0])
        #amplitude_model_visibilities = np.sqrt(model_visibilities[:, 0]**2.0 + model_visibilities[:, 1]**2.0)
        phase_model_visibilities = np.arctan2(model_visibilities[:, 1], model_visibilities[:, 0])
    else:
        raise ValueError

    if dPhi_dphi is None:
        antenna1 = antennas[:, 0]
        antenna2 = antennas[:, 1]
        antenna_unique=np.unique(
            np.asarray([
                antenna1,
                antenna2
            ]
        ).flatten())

        dPhi_dphi = np.zeros(
            shape=(
                antenna_unique.size,
                visibilities.shape[0]
            ),
            dtype=np.float64
        )
        for j in range(0, antenna_unique.size):
            dPhi_dphi[j, :] = (antenna1 == antenna_unique[j]) - 1.0 * (antenna2 == antenna_unique[j])


    # If the phase errors are not provided, calculate them ...
    if C is None:
        C = np.diag(np.ones(shape=visibilities.shape[0]))

    # This is just to avoid inverting this matrix for now, because the inverse of the identity matrix is an identity matrix ...
    if all(C_i == 1.0 for C_i in C.diagonal()):
        Cinv = C
    else:
        Cinv = np.linalg.inv(C)

    F = np.dot(dPhi_dphi, np.dot(Cinv, dPhi_dphi.T))
    Finv = np.linalg.inv(F)
    FdPC = np.dot(-Finv, np.dot(dPhi_dphi, Cinv))
    deltaphi = phase_visibilities - phase_model_visibilities
    # NOTE : wrap to +/- pi
    deltaphi = (deltaphi + np.pi) % (2.0 * np.pi) - np.pi
    dphi = np.dot(FdPC, deltaphi)
    # NOTE : There must be something wrong going on with this matrix (FdPC) ... dont know what yet ...
    # plt.figure()
    # plt.imshow(
    #     FdPC,
    #     cmap="jet",
    #     aspect="auto")
    # plt.colorbar()
    # plt.show()
    # exit()

    phase_offsets = np.dot(
        dPhi_dphi.T,
        dphi
    )
    visibilities_calibrated_temp = amplitude_visibilities * np.exp(1j * (phase_visibilities + phase_offsets))
    visibilities_calibrated = np.array([
        visibilities_calibrated_temp.real,
        visibilities_calibrated_temp.imag]
    ).T

    return visibilities_calibrated, dphi

if __name__=="__main__":

    fits_path = "."

    hdu = fits.open(fits_path + "/uv_wavelengths.fits")
    uv_wavelengths_hdu = 0
    uv_wavelengths = hdu[uv_wavelengths_hdu].data

    hdu = fits.open(fits_path + "/antennas.fits")
    antennas_hdu = 0
    antennas = hdu[antennas_hdu].data
    antenna_unique = np.unique(np.asarray([
        antennas[:, 0],
        antennas[:, 1]]).flatten())

    hdu = fits.open(fits_path + "/time.fits")
    time_hdu = 0
    time = hdu[time_hdu].data
    time_unique = np.unique(time)

    hdu = fits.open(fits_path + "/noise_map.fits")
    noise_map_hdu = 0
    noise_map = hdu[noise_map_hdu].data

    hdu = fits.open(fits_path + "/visibilities.fits")
    visibilities_hdu = 0
    visibilities = hdu[visibilities_hdu].data

    amplitude = np.sqrt(visibilities[:, 0]**2.0 + visibilities[:, 1]**2.0)
    phase = np.arctan2(visibilities[:, 1], visibilities[:, 0])

    # plt.plot(np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1]), phase, linestyle="None", marker="o")
    # plt.show()
    # exit()

    phase_offsets_std = 10.0
    phase_offsets_temp = np.random.normal(
        0.0,
        phase_offsets_std,
        len(antenna_unique))
    phase_offsets_true = phase_offsets_temp
    if np.any(phase_offsets_temp < -180.0) or np.any(phase_offsets_temp > 180.0):
        print("Need to wrap the phase offsets around -180 (-pi) and 180 (pi) degrees (radians)")
    dPhi_dphi = np.zeros((
        antenna_unique.size,
        visibilities.shape[0]))
    for j in range(0, antenna_unique.size):
        dPhi_dphi[j, :] = (antennas[:, 0] == antenna_unique[j]) - 1 * (antennas[:, 1] == antenna_unique[j])
    phase_offsets = np.dot(
        dPhi_dphi.T,
        phase_offsets_temp
    )

    phase_corrupted = phase + (phase_offsets * units.deg.to(units.rad))
    # NOTE : wrap to +/- pi
    phase_corrupted = (phase_corrupted + np.pi) % (2.0 * np.pi) - np.pi
    visibilities_corrupted = amplitude * np.exp(1j * phase_corrupted)
    visibilities_corrupted = np.array([
        visibilities_corrupted.real,
        visibilities_corrupted.imag]).T

    amplitude_corrupted = np.sqrt(visibilities_corrupted[:, 0]**2.0 + visibilities_corrupted[:, 1]**2.0)
    phase_corrupted = np.arctan2(visibilities_corrupted[:, 1], visibilities_corrupted[:, 0])

    # # TEST 1: check that the phase offsets have been implemented correctly.
    # antenna_1_selected = 0
    # antenna_2_selected = 1
    # idx = np.logical_and(
    #     antennas[:, 0]==antenna_1_selected,
    #     antennas[:, 1]==antenna_2_selected
    # )
    # if np.sum(idx) == len(time_unique):
    #     phase_diff_for_antenna_pair_vs_time = phase_corrupted[idx] - phase[idx]
    #     print(-(phase_offsets_true[antenna_2_selected] - phase_offsets_true[antenna_1_selected]) * units.deg.to(units.rad))
    #     print(phase_diff_for_antenna_pair_vs_time)
    # exit()
    # # TEST 2: check that the phase offsets have been implemented correctly.
    # for i in range(len(antenna_unique)):
    #     for j in range(len(antenna_unique)):
    #         if i<j:
    #             antenna_1_selected = i
    #             antenna_2_selected = j
    #             diff = phase_offsets_true[antenna_2_selected] - phase_offsets_true[antenna_1_selected]
    #             diff *= -1
    #
    #             idx = np.logical_and(
    #                 antennas[:, 0]==antenna_1_selected,
    #                 antennas[:, 1]==antenna_2_selected
    #             )
    #             if np.sum(idx) == len(time_unique):
    #                 phase_diff_for_antenna_pair_vs_time = phase_corrupted[idx] - phase[idx]
    #                 if np.all(array_element == diff for array_element in phase_diff_for_antenna_pair_vs_time):
    #                     print(i,j,)
    #                 else:
    #                     print(i,j,"WTF???")
    # exit()

    C = np.diag(np.ones(shape=visibilities_corrupted.shape[0]))
    # NOTE : In practice the diagonal elements of the C array should reflect the error in the phase which is given by the expression below. However, for this set of simulations, the visibilities do not have any noise (here noise does NOT refer to phase offsets).
    #phase_error = np.sqrt((noise_map[:, 0]**2.0 * visibilities_corrupted[:, 0]**2.0 + noise_map[:, 1]**2.0 * visibilities_corrupted[:, 1]**2.0) / amplitude_corrupted**4.0)

    visibilities_calibrated, phase_offsets_recovered_from_calibration = perturbative_self_calibration(
        visibilities=visibilities_corrupted,
        model_visibilities=visibilities,
        noise_map=noise_map,
        antennas=antennas,
        dPhi_dphi=dPhi_dphi,
        C=C
    )

    # # TRY THIS : What if I use only visibilities corresponding to t=t1. For some reason the recovered visibilities seem to get better ... This is counter-intuitive because we have more equations for the same number of unknowns ... That would imply that the numerical error carries on and it's additive rather than random ...
    # idx = np.where(time == time_unique[0])
    # visibilities_corrupted_idx = visibilities_corrupted[idx[0], :]
    # visibilities_idx = visibilities[idx[0], :]
    # noise_map_idx = noise_map[idx[0], :]
    # antennas_idx = antennas[idx[0], :]
    # dPhi_dphi_idx = np.zeros((
    #     antenna_unique.size,
    #     visibilities_idx.shape[0]))
    # for j in range(0, antenna_unique.size):
    #     dPhi_dphi_idx[j, :] = (antennas_idx[:, 0] == antenna_unique[j]) - 1 * (antennas_idx[:, 1] == antenna_unique[j])
    # C_idx = np.diag(np.ones(shape=visibilities_corrupted_idx.shape[0]))
    # visibilities_calibrated, phase_offsets_recovered_from_calibration = perturbative_self_calibration(
    #     visibilities=visibilities_corrupted_idx,
    #     model_visibilities=visibilities_idx,
    #     noise_map=noise_map_idx,
    #     antennas=antennas_idx,
    #     dPhi_dphi=dPhi_dphi_idx,
    #     C=C_idx
    # )

    # NOTE : The corrupted visibilities are invariant to adding a constant value to all "phase_offsets_true" because:
    #
    # V_{12,corrupted} = amplitude_{12} * exp(1j * (phase_{12} + (phase_offset_1 - phase_offset_2)))
    #
    # So, if
    #
    #   phase_offset_1_new = phase_offset_1 + constant
    #   phase_offset_2_new = phase_offset_2 + constant
    #
    # then V_{12,corrupted} will be the same. In this case, choose a reference antenna "antenna_ref" (It can be any antenna).

    antenna_ref = 0
    offset = (phase_offsets_true[antenna_ref] * units.deg.to(units.rad)) + phase_offsets_recovered_from_calibration[antenna_ref]

    # TODO : In practice we dont need to solve for N number of antennas since we can choose a reference antenna and solve for N-1, where N is the number of antennas. I haven't implemented this yet.

    plt.figure()
    plt.plot(
        phase_offsets_true * units.deg.to(units.rad),
        color="b",
        label="true"
    )
    plt.plot(
        -phase_offsets_recovered_from_calibration + offset,
        color="r",
        label="recovered"
    )
    diff = ((phase_offsets_true * units.deg.to(units.rad)) + phase_offsets_recovered_from_calibration - offset)
    plt.plot(
        diff,
        color="black",
        label="true - recovered"
    )
    plt.axhline(
        0.0,
        linestyle="--",
        color="black"
    )
    plt.xlabel(
        "# of antennas",
        fontsize=15
    )
    plt.ylabel(
        "phase offset (rad)",
        fontsize=15
    )
    plt.legend()
    plt.show()
