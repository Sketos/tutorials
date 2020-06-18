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


def model_data(fit, transformers, output_path=None, output_filename=None, output_format="show"):

    cube = autolens_plot_utils.dirty_cube_from_visibilities(
        visibilities=fit.model_data,
        transformers=transformers,
        shape=fit.grid_shape_3d
    )

    plotter.cube_plotter(
        cube=cube,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def residual_map(fit, transformers, output_path=None, output_filename=None, output_format="show"):

    cube = autolens_plot_utils.dirty_cube_from_visibilities(
        visibilities=fit.residual_map,
        transformers=transformers,
        shape=fit.grid_shape_3d
    )

    plotter.cube_plotter(
        cube=cube,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
