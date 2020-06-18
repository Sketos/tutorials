import os
import sys

sys.path.append(
    "{}/tutorials/autofit/tutorial_0".format(
        os.environ["GitHub"]
    )
)

from src.plot import (
    dataset_plots, fit_plots
)

class AbstractVisualizer:
    def __init__(self, image_path):

        self.image_path = image_path


class Visualizer(AbstractVisualizer):
    def __init__(self, masked_dataset, transformer, image_path):

        super().__init__(image_path)

        self.masked_dataset = masked_dataset

        self.transformer = transformer


        dataset_plots.data(
            masked_dataset=self.masked_dataset,
            transformer=self.transformer,
            output_filename="data",
            output_path=self.image_path,
            output_format="png",
        )

        dataset_plots.sigma(
            masked_dataset=self.masked_dataset,
            transformer=self.transformer,
            output_filename="sigma",
            output_path=self.image_path,
            output_format="png",
        )

    def visualize_model_data(self, model_data, phase_calibrated, during_analysis):

        fit_plots.model_data_bettername(
            model_data=model_data,
            transformer=self.transformer,
            output_filename="fit_model_data" if not phase_calibrated else "fit_model_data_with_phase_errors",
            output_path=self.image_path,
            output_format="png",
        )

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
        #     transformer=self.transformer,
        #     output_filename="fit_model_data",
        #     output_path=self.image_path,
        #     output_format="png",
        # )

        fit_plots.residual_map(
            fit=fit,
            transformer=self.transformer,
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
