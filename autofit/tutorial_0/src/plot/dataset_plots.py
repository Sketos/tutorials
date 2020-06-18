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


def data(
    masked_dataset,
    transformer,
    output_path=None,
    output_filename=None,
    output_format="show"
):

    dirty_image = autolens_plot_utils.dirty_image_from_visibilities_and_transformer(
        visibilities=masked_dataset.data,
        transformer=transformer
    )

    plotter.image_plotter(
        image=dirty_image,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def sigma(
    masked_dataset,
    transformer,
    output_path=None,
    output_filename=None,
    output_format="show"
):

    dirty_image = autolens_plot_utils.dirty_image_from_visibilities_and_transformer(
        visibilities=masked_dataset.sigma,
        transformer=transformer
    )

    plotter.image_plotter(
        image=dirty_image,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
