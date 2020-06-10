import autofit as af
import autolens as al

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

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

def cube_from_image(image, shape_3d):

    # NOTE:
    if image.in_2d.shape != shape_3d[1:]:
        raise ValueError("...")

    return np.tile(
        A=image.in_2d, reps=(shape_3d[0], 1, 1)
    )

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
        masked_datasets,
        transformers,
        transformer_continuum,
        lens_redshift,
        source_redshift,
        image_path=None
    ):

        self.masked_dataset = masked_dataset
        self.masked_datasets = masked_datasets

        self.transformers = transformers
        self.transformer_continuum = transformer_continuum

        self.lens_redshift = lens_redshift
        self.source_redshift = source_redshift

        self.visualizer = visualizer.Visualizer(
            masked_dataset=self.masked_dataset,
            transformers=self.transformers,
            image_path=image_path
        )

        self.n_tot = 0

    def fit(self, instance):

        model_data = self.model_data_from_instance(
            instance=instance
        )


        fit_continuum = self.fit_from_masked_dataset_and_model_data(
            masked_dataset=self.masked_datasets["continuum"],
            model_data=model_data["continuum"]
        )

        #print("residual_map", fit_continuum.residual_map)
        print("chi_squared = ", fit_continuum.chi_squared)
        #print(fit_continuum.likelihood)
        exit()

        """
        start = time.time()
        model_data = self.model_data_from_instance(
            instance=instance
        )
        end = time.time()
        # print(
        #     "It tool t={} to compute the model".format(end - start)
        # )

        fit = self.fit_from_model_data(model_data=model_data)
        #print(fit.likelihood)
        #exit()

        self.n_tot += 1
        print(
            "n = {}".format(self.n_tot)
        )

        return fit.likelihood
        """

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

    # NOTE: This is going to be used for the updated method to compute the lensed cube.
    # def lensed_cube_from_tracer_and_src_profiles(self, src_profiles, tracer):
    #
    #     profile_images = []
    #     for profile in src_profiles:
    #         if profile.analytic:
    #             profile_images.append(
    #                 cube_from_image(
    #                     image=tracer.profile_image_from_grid(
    #                         grid=self.masked_dataset.grid_3d.grid_2d
    #                     ),
    #                     shape_3d=self.masked_dataset.grid_3d.shape_3d
    #                 )
    #             )
    #         else:
    #             profile_images.append(
    #                 autolens_tracer_utils.lensed_cube_from_tracer(
    #                     tracer=tracer,
    #                     grid=self.masked_dataset.grid_3d.grid_2d,
    #                     cube=profile.profile_cube_from_grid(
    #                         grid=self.masked_dataset.grid_3d.grid_2d,
    #                         shape_3d=self.masked_dataset.grid_3d.shape_3d,
    #                         z_step_kms=self.masked_dataset.z_step_kms
    #                     )
    #                 )
    #             )
    #
    #     return sum(profile_images)

    def model_data_from_instance(self, instance):

        len_profiles = []
        src_profiles = []
        for profile in instance.profiles:
            if isinstance(profile, al.mp.MassProfile):
                len_profiles.append(profile)
            else:
                src_profiles.append(profile)

        galaxies = []
        for profile in len_profiles:
            galaxies.append(
                al.Galaxy(
                    redshift=self.lens_redshift,
                    mass=profile,
                )
            )

        src_cont_profiles = []
        src_line_profiles = []
        for profile in src_profiles:
            if not profile.is_3d_profile:
                src_cont_profiles.append(profile)
            else:
                src_line_profiles.append(profile)

        # tracer_cont = al.Tracer.from_galaxies(
        #     galaxies=[
        #         *galaxies,
        #         *[
        #             al.Galaxy(
        #                 redshift=self.source_redshift,
        #                 light=profile,
        #             )
        #             for profile in src_cont_profiles
        #         ]
        #     ]
        # )
        #
        # lensed_image = tracer_cont.profile_image_from_grid(
        #     grid=self.masked_dataset.grid_3d.grid_2d
        # )
        # # plt.figure()
        # # plt.imshow(lensed_image.in_2d_binned)
        # # plt.show()
        # # exit()
        #
        # lensed_cube = autolens_tracer_utils.lensed_cube_from_tracer(
        #     tracer=al.Tracer.from_galaxies(
        #         galaxies=[
        #             *galaxies,
        #             al.Galaxy(
        #                 redshift=self.source_redshift,
        #                 light=al.lp.LightProfile()
        #             )
        #         ]
        #     ),
        #     grid=self.masked_dataset.grid_3d.grid_2d,
        #     cube=self.src_model_from_profiles(
        #         profiles=src_line_profiles
        #     )
        # )
        # # plot_utils.plot_cube(
        # #     cube=lensed_cube,
        # #     ncols=8
        # # )
        # # exit()
        #
        #
        # model_data_cont = self.transformer_continuum.visibilities_from_image(
        #     image=lensed_image
        # )
        # #print(model_data_cont.shape)
        #
        # # TODO: subtract the continuum
        #
        # return {
        #     "continuum":model_data_cont
        # }


        lensed_cube_cont = autolens_tracer_utils.lensed_cube_from_tracer(
            tracer=al.Tracer.from_galaxies(
                galaxies=[
                    *galaxies,
                    al.Galaxy(
                        redshift=self.source_redshift,
                        light=al.lp.LightProfile()
                    )
                ]
            ),
            grid=self.masked_dataset.grid_3d.grid_2d,
            cube=self.src_model_from_profiles(
                profiles=src_cont_profiles
            )
        )
        # plot_utils.plot_cube(
        #     cube=lensed_cube_cont,
        #     ncols=8
        # )
        # exit()

        model_data_cont = self.transformer_continuum.visibilities_from_image(
            image=Image(
                array_2d=lensed_cube_cont[0]
            )
        )

        return {
            "continuum":model_data_cont
        }

        #exit()
        """
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
        # plot_utils.plot_cube(
        #     cube=lensed_cube,
        #     ncols=8
        # )
        # exit()

        # NOTE: This is going to be used for the updated method to compute the lensed cube.
        # lensed_cube = self.lensed_cube_from_tracer_and_src_profiles(
        #     src_profiles=src_profiles,
        #     tracer=tracer
        # )
        # plot_utils.plot_cube(
        #     cube=lensed_cube,
        #     ncols=8
        # )
        # exit()

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
        """

    # def fit_from_model_data(self, model_data):
    #
    #     return f.DatasetFit(
    #         masked_dataset=self.masked_dataset,
    #         model_data=model_data
    #     )

    def fit_from_masked_dataset_and_model_data(self, masked_dataset, model_data):

        return f.DatasetFit(
            masked_dataset=masked_dataset,
            model_data=model_data
        )

    def visualize(self, instance, during_analysis):
        # NOTE: Does the visualizer calling the same functions as "fit" does?

        model_data = self.model_data_from_instance(
            instance=instance
        )

        fit = self.fit_from_model_data(model_data=model_data)

        self.visualizer.visualize_fit(
            fit=fit,
            during_analysis=during_analysis
        )
