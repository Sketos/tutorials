import os
import sys
import matplotlib.pyplot as plt


def image_plotter(
    image,
    cmap="jet",
    output_path=None,
    output_filename=None,
    output_format="show",
):

    plt.figure()
    plt.imshow(
        image,
        cmap=cmap
    )
    plt.colorbar()

    if "show" in output_format:
        plt.show()
    elif "png" in output_format:
        plt.savefig(output_path + output_filename + ".png")
    plt.clf()
