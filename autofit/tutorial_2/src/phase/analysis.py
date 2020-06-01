import autofit as af

import os
import sys
import time

sys.path.append(
    "{}/tutorials/autofit/tutorial_2".format(
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

        start = time.time()
        model_data = self.model_data_from_instance(
            instance=instance
        )
        end = time.time()
        print(
            "It tool t={} to compute the model".format(end - start)
        )

        fit = self.fit_from_model_data(model_data=model_data)
        print(fit.likelihood)
        return fit.likelihood

    def model_data_from_instance(self, instance):

        return sum(
            [
                profile.profile_cube_from_grid(
                    grid=self.masked_dataset.grid_3d.grid_2d,
                    shape_3d=self.masked_dataset.grid_3d.shape_3d,
                    z_step_kms=self.masked_dataset.z_step_kms
                )
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
