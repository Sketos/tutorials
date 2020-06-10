import os
import sys

import matplotlib.pyplot as plt

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import plot_utils as plot_utils


def cube_plotter(
    cube,
    output_path=None,
    output_filename=None,
    output_format="show",
):

    plot_utils.plot_cube(
        cube=cube,
        ncols=8,
        show=False
    )

    if "show" in output_format:
        plt.show()
    elif "png" in output_format:
        plt.savefig(output_path + output_filename + ".png")
    plt.clf()


# def data(fit, output_path=None, output_filename=None, output_format="show"):
#     """Plot the data values of a Fit.
#
#     Parameters
#     -----------
#     Fit : fit.Fit
#         The observed Fit dataset whose data is plotted.
#     """
#     array_plotter(
#         array=fit.data,
#         output_path=output_path,
#         output_filename=output_filename,
#         output_format=output_format,
#     )
#
#
# def model_data(fit, output_path=None, output_filename=None, output_format="show"):
#     """Plot the model data of a Fit.
#
#     Parameters
#     -----------
#     Fit : fit.Fit
#         The Fit model data.
#     """
#     array_plotter(
#         array=fit.model_data,
#         output_path=output_path,
#         output_filename=output_filename,
#         output_format=output_format,
#     )

def residual_map(fit, output_path=None, output_filename=None, output_format="show"):

    cube_plotter(
        cube=fit.residual_map,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
