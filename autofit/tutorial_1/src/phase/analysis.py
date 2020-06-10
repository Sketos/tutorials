import os
import sys

import autofit as af

sys.path.append(
    "{}/tutorials/autofit/tutorial_1".format(
        os.environ["GitHub"]
    )
)

from src.fit import fit as f
from src.phase import (
    visualizer,
)


class Analysis(af.Analysis):
    def __init__(self, masked_dataset, image_path=None):

        self.masked_dataset = masked_dataset

        self.visualizer = visualizer.Visualizer(
            masked_dataset=self.masked_dataset, image_path=image_path
        )

    def fit(self, instance):

        model_data = self.model_data_from_instance(
            instance=instance
        )

        fit = self.fit_from_model_data(model_data=model_data)
        print("likelihood = ", fit.likelihood)

        return fit.likelihood

    def model_data_from_instance(self, instance):

        return sum(
            [
                profile.profile_image_from_grid(grid=self.masked_dataset.grid).in_2d
                for profile in instance.profiles
            ]
        )

    def fit_from_model_data(self, model_data):

        return f.DatasetFit(
            masked_dataset=self.masked_dataset,
            model_data=model_data
        )

    def visualize(self, instance, during_analysis):

        model_data = self.model_data_from_instance(
            instance=instance
        )

        fit = self.fit_from_model_data(model_data=model_data)

        self.visualizer.visualize_fit(
            fit=fit,
            during_analysis=during_analysis
        )
