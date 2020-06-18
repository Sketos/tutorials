import os
import sys

sys.path.append(
    "{}/tutorials/autofit/tutorial_6".format(
        os.environ["GitHub"]
    )
)
from src.plot import plotter

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import autolens_utils.autolens_plot_utils as autolens_plot_utils


def data(masked_dataset, transformers, output_path=None, output_filename=None, output_format="show"):

    cube = autolens_plot_utils.dirty_cube_from_visibilities(
        visibilities=masked_dataset.visibilities,
        transformers=transformers,
        shape=masked_dataset.grid_shape_3d
    )

    plotter.cube_plotter(
        cube=cube,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
