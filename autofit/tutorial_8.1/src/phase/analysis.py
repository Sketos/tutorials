import autofit as af
import autolens as al

from autoarray.exc import InversionException
from autofit.exc import FitException
from autoarray.util import inversion_util
from autoarray.operators.inversion import inversions as inv

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg

sys.path.append(
    "{}/tutorials/autofit/tutorial_8.1".format(
        os.environ["GitHub"]
    )
)

from src.fit import fit as f
from src.phase import (
    visualizer,
)
from src.dataset.dataset import (
    MaskedDatasetLite
)

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import plot_utils as plot_utils

#import autolens_utils.autolens_tracer_utils as autolens_tracer_utils
import autolens_utils.autolens_plot_utils as autolens_plot_utils


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


def reshape_array(array):

    return array.reshape(
        -1,
        array.shape[-1]
    )


class Analysis(af.Analysis):
    def __init__(self, masked_datasets, transformers, grid, image_path=None):

        self.masked_datasets = masked_datasets
        if "continuum" not in self.masked_datasets.keys():
            raise ValueError("...")

        self.transformers = transformers
        if "continuum" not in self.transformers.keys():
            raise ValueError("...")

        self.grid = grid

        self.visualizer = visualizer.Visualizer(
            masked_datasets=self.masked_datasets,
            image_path=image_path
        )

        self.n_tot = 0

    def fit(self, instance):

        def mapped_reconstructed_visibilities(transformed_mapping_matrices, inversion):

            real_visibilities = inversion_util.mapped_reconstructed_data_from_mapping_matrix_and_reconstruction(
                mapping_matrix=transformed_mapping_matrices[0],
                reconstruction=inversion.reconstruction,
            )

            imag_visibilities = inversion_util.mapped_reconstructed_data_from_mapping_matrix_and_reconstruction(
                mapping_matrix=transformed_mapping_matrices[1],
                reconstruction=inversion.reconstruction,
            )

            return np.stack(
                arrays=(real_visibilities, imag_visibilities),
                axis=-1
            )

        # NOTE: There is one tracer corresponding to the continuum and among the
        # remaining tracers there should be one for each region.
        # NOTE: Should we impose that the reg coefficient of the pixelization
        # corresponding to the continuum must have a lower value (because it
        # has a higher SNR)?
        tracers = self.tracers_from_instance(
            instance=instance
        )

        mappers = [self.mapper_from_tracer(tracer)
            for tracer in tracers
        ]

        inversions = {
            "continuum":self.get_inversion_continuum(
                mapper=mappers[0],
                regularization=tracers[0].source_plane.regularization
            )
        }
        # plt.figure()
        # autolens_plot_utils.draw_voronoi_pixels(
        #     mapper=inversions["continuum"].mapper,
        #     values=inversions["continuum"].reconstruction,
        # )
        # plt.show()
        # exit()

        n = 1
        for key in self.masked_datasets.keys():
            if key == "continuum":
                pass
            elif key.startswith("region"):
                if (tracers[0].source_plane.pixelization.shape == tracers[n].source_plane.pixelization.shape):
                    region_inversions = []
                    for i, transformer in enumerate(self.transformers[key]):
                        print(key, i)

                        transformed_mapping_matrices = transformer.transformed_mapping_matrices_from_mapping_matrix(
                            mapping_matrix=inversions["continuum"].mapper.mapping_matrix
                        )

                        visibilities_continuum = mapped_reconstructed_visibilities(
                            transformed_mapping_matrices=transformed_mapping_matrices,
                            inversion=inversions["continuum"]
                        )

                        region_inversions.append(
                            inv.InversionInterferometer.from_data_mapper_and_regularization_precomputed(
                                visibilities=np.stack(
                                    arrays=(
                                        np.subtract(
                                            self.masked_datasets[key].visibilities[i, :, 0],
                                            visibilities_continuum[:, 0]
                                        ),
                                        np.subtract(
                                            self.masked_datasets[key].visibilities[i, :, 1],
                                            visibilities_continuum[:, 1]
                                        )
                                    ),
                                    axis=-1
                                ),
                                noise_map=self.masked_datasets[key].noise_map[i],
                                transformer=transformer,
                                mapper=mappers[n],
                                regularization=tracers[n].source_plane.regularization,
                                transformed_mapping_matrices=transformed_mapping_matrices
                            )
                        )

                    inversions[key] = region_inversions
                else:
                    raise ValueError("Not implemented yet")
                    # TODO: In the case where the 2 pixelizations do not have the
                    # same number of pixels.

                # autolens_plot_utils.plot_reconstructions_from_inversions(
                #     inversions=inversions[key],
                #     nrows=3,
                #     ncols=5,
                #     figsize=(12.5, 5),
                #     xlim=(-1.5, 1.5),
                #     ylim=(-1.5, 1.5),
                # )
                # exit()

                n += 1

        try:
            fits = {}
            for key in self.masked_datasets.keys():
                if key == "continuum":
                    fits[key] = self.fit_from_masked_dataset_model_data_and_inversion(
                        masked_dataset=self.masked_datasets[key],
                        model_data=inversions[key].mapped_reconstructed_visibilities,
                        inversion=inversions[key]
                    )
                elif key.startswith("region"):
                    region_fits = []

                    for i, inversion in enumerate(inversions[key]):
                        region_fits.append(
                            self.fit_from_masked_dataset_model_data_and_inversion(
                                masked_dataset=MaskedDatasetLite(
                                    uv_wavelengths=self.masked_datasets[key].uv_wavelengths[i],
                                    visibilities=self.masked_datasets[key].visibilities[i],
                                    noise_map=self.masked_datasets[key].noise_map[i],
                                    noise_map_real_and_imag_averaged=self.masked_datasets[key].noise_map_real_and_imag_averaged[i],
                                    uv_mask=self.masked_datasets[key].uv_mask[i],
                                    uv_mask_real_and_imag_averaged=self.masked_datasets[key].uv_mask_real_and_imag_averaged[i]
                                ),
                                model_data=inversion.mapped_reconstructed_visibilities,
                                inversion=inversion
                            )
                        )

                    fits[key] = region_fits

            figure_of_merit = 0.0
            for key in fits.keys():
                if key == "continuum":
                    figure_of_merit += fits[key].figure_of_merit
                elif key.startswith("region"):
                    for fit in fits[key]:
                        figure_of_merit += fit.figure_of_merit

            self.n_tot += 1
            print(
                "n = {}".format(self.n_tot)
            )

            print("figure_of_merit = ", figure_of_merit)

            return figure_of_merit
        except InversionException as e:
            raise FitException from e


    def get_inversion_continuum(self, mapper, regularization):

        return inv.InversionInterferometer.from_data_mapper_and_regularization(
            visibilities=self.masked_datasets["continuum"].visibilities,
            noise_map=self.masked_datasets["continuum"].noise_map,
            transformer=self.transformers["continuum"],
            mapper=mapper,
            regularization=regularization
        )


    def mapper_from_tracer(self, tracer):

        mappers_of_planes = tracer.mappers_of_planes_from_grid(
            grid=self.grid,
            inversion_uses_border=False,
            preload_sparse_grids_of_planes=None
        )

        return mappers_of_planes[-1]


    def tracers_from_instance(self, instance):

        len_galaxies = []
        src_galaxies = []
        n = 0
        for galaxy in instance.galaxies:
            if galaxy.has_mass_profile:
                #print("mp")
                len_galaxies.append(galaxy)
            elif galaxy.has_pixelization:
                #print("pix")
                src_galaxies.append(galaxy)
                n += 1
            else:
                raise ValueError(
                    "This galaxy is not appropriate for this analysis"
                )

            if n > 2:
                # TODO: This condition should be modified to "if n > len(self.regions):"
                # if more than one region is being used in the case where we have multiple
                # emission lines in the ALMA cube. This assumes that self.regions is now
                # a list of lists. (FEATURE; not implemented yet)
                raise ValueError(
                    "We only need 2 source galaxies, one for the continuum and one for the line emission"
                )

        tracers = []
        for i, src_galaxy in enumerate(src_galaxies):
            tracers.append(
                al.Tracer.from_galaxies(
                    galaxies=len_galaxies + [src_galaxy, ]
                )
            )

        return tracers

    # TODO: Rename to fit_from_masked_dataset_and_inversion and get the model_data
    # from the inversion.
    def fit_from_masked_dataset_model_data_and_inversion(self, masked_dataset, model_data, inversion):

        return f.DatasetFit(
            masked_dataset=masked_dataset,
            model_data=model_data,
            inversion=inversion
        )

    def visualize(self, instance, during_analysis):
        pass
        # model_data = self.model_data_from_instance(
        #     instance=instance
        # )
        #
        # fit = self.fit_from_model_data(model_data=model_data)
        #
        # self.visualizer.visualize_fit(
        #     fit=fit,
        #     during_analysis=during_analysis
        # )
