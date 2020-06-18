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
