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
if not (al.__version__ == autolens_version):
    raise ValueError("...")

import numpy as np
import matplotlib.pyplot as plt
from astropy import units
from astropy.io import fits

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import casa_utils as casa_utils
import spectral_utils as spectral_utils

n_channels = 32

frequencies = casa_utils.generate_frequencies(
    central_frequency=260.0 * units.GHz,
    n_channels=n_channels,
    bandwidth=2.0 * units.GHz
)


if __name__ == "__main__":

    uv = fits.getdata("./uv.fits")
    # print(uv.shape)
    # plt.figure()
    # plt.plot(uv[0], uv[1], linestyle='None', marker="o")
    # plt.show()
    # exit()

    antennas = fits.getdata("./antennas.fits")
    #print(antennas.shape)

    #print(antennas[:, 0])
    #print(antennas[:, 1])

    antennas_unique = np.unique(
        np.asarray([antennas[:, 0], antennas[:, 1]]).flatten()
    )


    # dPhi_dphi = np.zeros((len(antennas_unique) - 1, antennas.shape[0]))
    # for j in range(1, len(antennas_unique)):
    #     dPhi_dphi[j-1,:] = (antennas[:, 0]==antennas_unique[j])-1*(antennas[:, 1]==antennas_unique[j])

    # dPhi_dphi = np.zeros((len(antennas_unique), antennas.shape[0]))
    # for j in range(0, len(antennas_unique)):
    #     dPhi_dphi[j, :] = (antennas[:, 0]==antennas_unique[j])-1*(antennas[:, 1]==antennas_unique[j])
    #
    # plt.figure()
    # plt.imshow(dPhi_dphi, aspect="auto")
    # plt.show()
