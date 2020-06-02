import autofit as af
import autolens as al

from autoarray.util import inversion_util
from autoarray.operators.inversion import inversions as inv

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg

sys.path.append(
    "{}/tutorials/autofit/tutorial_7".format(
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
    def __init__(self, masked_dataset, region_masked_datasets, transformers, transformer_continuum, image_path=None):

        # NOTE: I can get rid of the masked_dataset alltogether (currently it's
        # only use is that it has a grid3D associated with it) and work with the
        # region_masked_datasets
        # NOTE: region_masked_datasets[0] is the continuum and region_masked_datasets[1]
        # is the line emission.
        self.masked_dataset = masked_dataset
        self.region_masked_datasets = region_masked_datasets

        self.transformers = transformers
        self.transformer_continuum = transformer_continuum

        self.visualizer = visualizer.Visualizer(
            masked_dataset=self.masked_dataset, image_path=image_path
        )

    def mapper_from_tracer(self, tracer):

        mappers_of_planes = tracer.mappers_of_planes_from_grid(
            grid=self.masked_dataset.grid_3d.grid_2d,
            inversion_uses_border=False,
            preload_sparse_grids_of_planes=None
        )

        return mappers_of_planes[-1]

    def fit(self, instance):

        def funcname(transformed_mapping_matrices, inversion):

            real_visibilities = inversion_util.mapped_reconstructed_data_from_mapping_matrix_and_reconstruction(
                mapping_matrix=transformed_mapping_matrices[0],
                reconstruction=inversion.reconstruction,
            )

            imag_visibilities = inversion_util.mapped_reconstructed_data_from_mapping_matrix_and_reconstruction(
                mapping_matrix=transformed_mapping_matrices[1],
                reconstruction=inversion.reconstruction,
            )

            return real_visibilities, imag_visibilities

        tracers = self.tracers_from_instance(
            instance=instance
        )
        print(tracers[0].source_plane.regularization.coefficient)
        print(tracers[0].source_plane.pixelization.shape)
        print(tracers[1].source_plane.regularization.coefficient)
        print(tracers[1].source_plane.pixelization.shape)

        mappers = [self.mapper_from_tracer(tracer)
            for tracer in tracers
        ]

        inversion_continuum = inv.InversionInterferometer.from_data_mapper_and_regularization(
            visibilities=self.region_masked_datasets[0].visibilities,
            noise_map=self.region_masked_datasets[0].noise_map,
            transformer=self.transformer_continuum,
            mapper=mappers[0],
            regularization=tracers[0].source_plane.regularization
        )
        # plt.figure()
        # autolens_plot_utils.draw_voronoi_pixels(
        #     mapper=inversion_continuum.mapper,
        #     values=inversion_continuum.reconstruction,
        # )
        # plt.show()
        # exit()

        inversions = []
        j = 0
        for i, transformer in enumerate(self.transformers):

            if self.masked_dataset.region[i]:
                print(i, )

                transformed_mapping_matrices = transformer.transformed_mapping_matrices_from_mapping_matrix(
                    mapping_matrix=mappers[0].mapping_matrix
                )

                real_visibilities_continuum, imag_visibilities_continuum = funcname(
                    transformed_mapping_matrices=transformed_mapping_matrices,
                    inversion=inversion_continuum
                )

                # NOTE: If the 2 pixelizations have the same number shape the
                # precomputed mapping matrices can be used. The only distinction
                # between from_data_mapper_and_regularization_precomputed and
                # from_data_mapper_and_regularization is that if
                # transformed_mapping_matrices is passed to it then it skips it's
                # calculation
                # NOTE: The noise map does not change if a contant is subtracted
                # from the data
                # NOTE: The visibilities of each inversion in the line region are
                # now the continuum subtracted visibilties
                inversions.append(
                    inv.InversionInterferometer.from_data_mapper_and_regularization_precomputed(
                        visibilities=np.stack(
                            arrays=(
                                np.subtract(
                                    self.region_masked_datasets[1].visibilities[j, :, 0],
                                    real_visibilities_continuum
                                ),
                                np.subtract(
                                    self.region_masked_datasets[1].visibilities[j, :, 1],
                                    imag_visibilities_continuum
                                )
                            ),
                            axis=-1
                        ),
                        noise_map=self.region_masked_datasets[1].noise_map[j],
                        transformer=transformer,
                        mapper=mappers[1],
                        regularization=tracers[1].source_plane.regularization,
                        transformed_mapping_matrices=transformed_mapping_matrices
                    )
                )

                j += 1
        # autolens_plot_utils.plot_reconstructions_from_inversions(
        #     inversions=inversions,
        #     nrows=3,
        #     ncols=5,
        #     figsize=(20, 5),
        #     xlim=(-1.5, 1.5),
        #     ylim=(-1.5, 1.5),
        # )
        # exit()

        fit_continuum = self.fit_from_masked_dataset_model_data_and_inversion(
            masked_dataset=self.region_masked_datasets[0],
            model_data=inversion_continuum.mapped_reconstructed_visibilities,
            inversion=inversion_continuum
        )

        fits = []
        for i, inversion in enumerate(inversions):

            # NOTE: ugly ...
            fits.append(
                self.fit_from_masked_dataset_model_data_and_inversion(
                    masked_dataset=MaskedDatasetLite(
                        visibilities=self.region_masked_datasets[1].visibilities[i],
                        noise_map=self.region_masked_datasets[1].noise_map[i],
                        noise_map_real_and_imag_averaged=self.region_masked_datasets[1].noise_map_real_and_imag_averaged[i],
                        uv_mask=self.region_masked_datasets[1].uv_mask[i],
                        uv_mask_real_and_imag_averaged=self.region_masked_datasets[1].uv_mask_real_and_imag_averaged[i]
                    ),
                    model_data=inversion.mapped_reconstructed_visibilities,
                    inversion=inversion
                )
            )

        figure_of_merit_continuum = fit_continuum.figure_of_merit
        #print(figure_of_merit_continuum)

        figure_of_merit_line = sum(
            [fit.figure_of_merit for fit in fits]
        )
        #print(figure_of_merit_line)

        figure_of_merit = figure_of_merit_continuum + figure_of_merit_line
        print(figure_of_merit)

        return figure_of_merit


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
                    "We only need to source galaxies, one for the continuum and one for the line emission"
                )

        tracers = []
        for i, src_galaxy in enumerate(src_galaxies):
            tracers.append(
                al.Tracer.from_galaxies(
                    galaxies=len_galaxies + [src_galaxy, ]
                )
            )

        return tracers

    def fit_from_masked_dataset_model_data_and_inversion(self, masked_dataset, model_data, inversion):

        return f.DatasetFit(
            masked_dataset=masked_dataset,
            model_data=model_data,
            inversion=inversion
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




"""
plt.figure(
    figsize=(18, 10)
)
nrows = 4
ncols = 8
xlim_min = -0.5
xlim_max = 0.5
ylim_min = -0.5
ylim_max = 0.5
xticks = np.linspace(xlim_min, xlim_max, 5)
yticks = np.linspace(ylim_min, ylim_max, 5)
i = 0
j = 0
for n in range(len(inversions)):
    print("n = ", n)

    plt.subplot(
        nrows,
        ncols,
        n+1
    )

    # autolens_plot_utils.draw_voronoi_pixels(
    #     mapper=mapper,
    #     values=reconstructions[n]
    # )
    autolens_plot_utils.draw_voronoi_pixels(
        mapper=inversions[n].mapper,
        values=inversions[n].reconstruction
    )

    plt.xlim((xlim_min, xlim_max))
    plt.ylim((ylim_min, ylim_max))

    if i == nrows - 1:
        plt.xticks(xticks[1:-1])
    else:
        plt.xticks([])
    if j == 0:
        plt.yticks(yticks[1:-1])
    else:
        plt.yticks([])

    j += 1
    if j == ncols:
        j = 0
        i += 1

plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.05, right=0.95, bottom=0.05, top=0.95)
plt.show()
exit()
"""
