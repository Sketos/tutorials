import autofit as af
import autoarray as aa
import autolens as al

import os
import sys
import time
import numpy as np

sys.path.append(
    "{}/tutorials/autofit/tutorial_0".format(
        os.environ["GitHub"]
    )
)

from src.fit import fit as f
from src.phase import (
    visualizer,
)

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import calibration_utils as calibration_utils
import plot_utils as plot_utils

import autolens_utils.autolens_tracer_utils as autolens_tracer_utils


def is_light_profile(obj):
    return isinstance(obj, al.lp.LightProfile)


def is_mass_profile(obj):
    return isinstance(obj, al.mp.MassProfile)



class Image:
    def __init__(self, array_2d):
        self.array_2d = array_2d

    @property
    def in_2d_binned(self):
        return self.array_2d


class Analysis(af.Analysis):
    def __init__(
        self,
        masked_dataset,
        transformer,
        self_calibration,
        image_path=None
    ):

        self.masked_dataset = masked_dataset

        self.transformer = transformer

        self.self_calibration = self_calibration

        self.visualizer = visualizer.Visualizer(
            masked_dataset=self.masked_dataset,
            transformer=self.transformer,
            image_path=image_path
        )

        self.n_tot = 0

    def fit(self, instance):

        start = time.time()
        model_data = self.model_data_from_instance(
            instance=instance
        )
        end = time.time()

        if self.self_calibration:
            print(
                "performing \"self-calibration\""
            )

            model_data = self.model_data_with_self_calibration(
                model_data=model_data
            )

        #exit()
        fit = self.fit_from_model_data(model_data=model_data)

        self.n_tot += 1
        print(
            "n = {}".format(self.n_tot)
        )

        # print(
        #     "chi_squared = {}".format(fit.chi_squared)
        # )
        # exit()
        print(
            "likelihood = {}".format(fit.likelihood)
        )
        #exit()

        return fit.likelihood


    def phase_errors_from_model_data(self, model_data):

        return calibration_utils.phase_errors_from_A_and_B_matrices(
            phases=self.masked_dataset.phases,
            model_phases=model_data.phases,
            A=self.masked_dataset.A,
            B=self.masked_dataset.B
        )

    def model_data_with_self_calibration(self, model_data):

        phase_errors = self.phase_errors_from_model_data(
            model_data=model_data
        )

        model_phases_corrected = np.add(
            model_data.phases,
            np.matmul(
                self.masked_dataset.f.T,
                phase_errors
            )
        )

        model_data = aa.structures.visibilities.Visibilities(
            visibilities_1d=np.stack(
                arrays=(
                    model_data.amplitudes * np.cos(model_phases_corrected),
                    model_data.amplitudes * np.sin(model_phases_corrected)
                ),
                axis=-1
            )
        )

        return model_data

    def model_data_from_instance(self, instance):

        tracer = al.Tracer.from_galaxies(
            galaxies=instance.galaxies
        )
        print(
            "axis_ratio = {}".format(tracer.mass_profiles[0].axis_ratio)
        )

        model_data = tracer.profile_visibilities_from_grid_and_transformer(
            grid=self.masked_dataset.grid,
            transformer=self.transformer
        )

        return model_data

    def fit_from_model_data(self, model_data):

        return f.DatasetFit(
            masked_dataset=self.masked_dataset,
            model_data=model_data
        )

    def visualize(self, instance, during_analysis):
        # NOTE: Does the visualizer calling the same functions as "fit" does?

        model_data = self.model_data_from_instance(
            instance=instance
        )

        self.visualizer.visualize_model_data(
            model_data=model_data,
            phase_calibrated=False,
            during_analysis=during_analysis,
        )

        if self.self_calibration:
            model_data = self.model_data_with_self_calibration(
                model_data=model_data
            )

            self.visualizer.visualize_model_data(
                model_data=model_data,
                phase_calibrated=True,
                during_analysis=during_analysis,
            )

        fit = self.fit_from_model_data(model_data=model_data)


        self.visualizer.visualize_fit(
            fit=fit,
            during_analysis=during_analysis
        )
