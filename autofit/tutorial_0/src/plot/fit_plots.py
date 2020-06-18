import os
import sys

sys.path.append(
    "{}/tutorials/autofit/tutorial_0".format(
        os.environ["GitHub"]
    )
)
from src.plot import plotter

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import autolens_utils.autolens_plot_utils as autolens_plot_utils


def model_data(fit, transformer, output_path=None, output_filename=None, output_format="show"):

    dirty_image = autolens_plot_utils.dirty_image_from_visibilities_and_transformer(
        visibilities=fit.model_data,
        transformer=transformer
    )

    plotter.image_plotter(
        image=dirty_image,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def model_data_bettername(model_data, transformer, output_path=None, output_filename=None, output_format="show"):

    dirty_image = autolens_plot_utils.dirty_image_from_visibilities_and_transformer(
        visibilities=model_data,
        transformer=transformer
    )

    plotter.image_plotter(
        image=dirty_image,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def residual_map(fit, transformer, output_path=None, output_filename=None, output_format="show"):

    dirty_image = autolens_plot_utils.dirty_image_from_visibilities_and_transformer(
        visibilities=fit.residual_map,
        transformer=transformer
    )

    plotter.image_plotter(
        image=dirty_image,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
