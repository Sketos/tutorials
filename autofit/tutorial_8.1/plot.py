import os
import sys
import matplotlib.pyplot as plt

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import getdist_utils as getdist_utils

from getdist import mcsamples, plots

import matplotlib
matplotlib.use('QT5Agg')

if __name__ == "__main__":


    optimizer_directory = "./output/tutorial_8.1/phase_tutorial_8.1/optimizer_backup"

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
        params=params,
        filled=True,
    )
    plt.show()
