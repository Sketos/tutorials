import os
import sys

sys.path.append(
    "{}/tutorials/autofit/tutorial_6".format(
        os.environ["GitHub"]
    )
)

from src.plot import (
    fit_plots
)

class AbstractVisualizer:
    def __init__(self, image_path):

        self.image_path = image_path


class Visualizer(AbstractVisualizer):
    def __init__(self, masked_dataset, transformers, image_path):

        super().__init__(image_path)

        self.masked_dataset = masked_dataset

        self.transformers = transformers

        # dataset_plots.data(
        #     dataset=masked_dataset,
        #     output_filename="dataset_data",
        #     output_path=self.image_path,
        #     output_format="png",
        # )
        # dataset_plots.noise_map(
        #     dataset=masked_dataset,
        #     output_filename="dataset_noise_map",
        #     output_path=self.image_path,
        #     output_format="png",
        # )

    def visualize_fit(self, fit, during_analysis):

        # fit_plots.data(
        #     fit=fit,
        #     output_filename="fit_data",
        #     output_path=self.image_path,
        #     output_format="png",
        # )
        # fit_plots.noise_map(
        #     fit=fit,
        #     output_filename="fit_noise_map",
        #     output_path=self.image_path,
        #     output_format="png",
        # )

        # fit_plots.model_data(
        #     fit=fit,
        #     transformers=self.transformers,
        #     output_filename="fit_model_data",
        #     output_path=self.image_path,
        #     output_format="png",
        # )

        fit_plots.residual_map(
            fit=fit,
            transformers=self.transformers,
            output_filename="fit_residual_map",
            output_path=self.image_path,
            output_format="png",
        )

        # fit_plots.chi_squared_map(
        #     fit=fit,
        #     output_filename="fit_chi_squared_map",
        #     output_path=self.image_path,
        #     output_format="png",
        # )
        #
        # if not during_analysis:
        #
        #     fit_plots.normalized_residual_map(
        #         fit=fit,
        #         output_filename="fit_normalized_residual_map",
        #         output_path=self.image_path,
        #         output_format="png",
        #     )
