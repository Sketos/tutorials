import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import getdist_utils as getdist_utils

from getdist import mcsamples, plots
import corner

import matplotlib
matplotlib.use('QT5Agg')

if __name__ == "__main__":





    runner = "runner__lens__powerlaw__source__kinematics__data__lens__powerlaw__source__kinematics"
    truths = [0.75, 45.0, 0.0, 0.0, 1.0, 2.0, 0.5, 30.0, 50.0, 0.05, 0.0, 0.0, 16.0, 5.0, 200.0, 50.0]
    labels = ["q", r"$\theta$", "y", "x", r"$\theta_E$", r"$\alpha$"]
    #runner = "runner__lens__sie__source__ellipticalsersic_and_kinematics__data__lens__sie__source__ellipticalsersic_and_kinematics"
    #truths = [0.75, 45.0, 0.0, 0.0, 1.0, 0.75, 45.0, 0.5, 1.0, 0.0, 0.0, 0.00005, 0.5, 30.0, 50.0, 0.05, 0.0, 0.0, 16.0, 5.0, 200.0, 50.0]


    optimizer_directory = "./output_cosma/{}/phase_1__version_0.45.0/optimizer_backup".format(runner)


    # # NOTE:
    #
    # samples = np.loadtxt(
    #     "{}/{}".format(optimizer_directory, "multinest.txt")
    # )
    # print(samples.shape)
    # exit()



    samples = mcsamples.loadMCSamples(
        "{}/multinest".format(optimizer_directory)
    )

    def get_params(samples):

        params = [
            name for name in samples.paramNames.names
        ]

        return params
        # if str(name).startswith(
        #     "galaxies_{}".format(galaxy)
        # )

    params = get_params(samples=samples)


    plotter = plots.get_single_plotter(width_inch=16)
    plotter.triangle_plot(
        samples,
        filled=True,
    )
    plt.show()

    # #print(samples.samples.shape);exit()
    # fig = corner.corner(
    #     xs=samples.samples[:, 0:6],
    #     weights=samples.weights,
    #     bins=20,
    #     colors="b",
    #     truths=truths[0:6],
    #     labels=labels,
    #     smooth=1.0,
    #     plot_datapoints=False
    # )
    # #quantiles=[0.16, 0.5, 0.84],
    # plt.show()
