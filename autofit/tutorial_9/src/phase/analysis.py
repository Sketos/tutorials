import autofit as af
import autolens as al

import os
import sys
import time
import numpy as np

sys.path.append(
    "{}/tutorials/autofit/tutorial_6".format(
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
    def __init__(self, masked_dataset, transformers, lens_redshift, source_redshift, image_path=None):

        self.masked_dataset = masked_dataset

        self.transformers = transformers

        self.lens_redshift = lens_redshift
        self.source_redshift = source_redshift

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

    def src_model_from_profiles(self, profiles):

        return sum(
            [
                profile.profile_cube_from_grid(
                    grid=self.masked_dataset.grid_3d.grid_2d,
                    shape_3d=self.masked_dataset.grid_3d.shape_3d,
                    z_step_kms=self.masked_dataset.z_step_kms
                )
                for profile in profiles
            ]
        )

    def model_data_from_instance(self, instance):

        galaxies = []
        src_profiles = []
        for profile in instance.profiles:

            if isinstance(profile, al.mp.MassProfile):
                galaxies.append(
                    al.Galaxy(
                        redshift=self.lens_redshift,
                        mass=profile,
                    )
                )

            else:

                src_profiles.append(profile)

        galaxies.append(
            al.Galaxy(
                redshift=self.source_redshift,
                light=al.lp.LightProfile()
            )
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=galaxies
        )

        lensed_cube = autolens_tracer_utils.lensed_cube_from_tracer(
            tracer=tracer,
            grid=self.masked_dataset.grid_3d.grid_2d,
            cube=self.src_model_from_profiles(
                profiles=src_profiles
            )
        )
        # plot_utils.plot_cube(cube=lensed_cube, ncols=8)

        model_data = np.zeros(
            shape=self.masked_dataset.data.shape
        )
        for i in range(model_data.shape[0]):
            model_data[i] = self.transformers[i].visibilities_from_image(
                    image=Image(
                        array_2d=lensed_cube[i]
                    )
                )

        return model_data

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
