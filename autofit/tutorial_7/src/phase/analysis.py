import autofit as af
import autolens as al

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


class Analysis(af.Analysis):
    def __init__(self, masked_dataset, transformers, image_path=None):

        self.masked_dataset = masked_dataset

        self.transformers = transformers

        self.visualizer = visualizer.Visualizer(
            masked_dataset=self.masked_dataset, image_path=image_path
        )

    def fit(self, instance):

        start = time.time()
        inversions = self.inversions_from_instance(
            instance=instance
        )
        end = time.time()
        print(
            "It tool t={} to compute the inversions".format(end - start)
        )

        # autolens_plot_utils.plot_reconstructions_from_inversions(
        #     inversions=inversions,
        #     nrows=4,
        #     ncols=8,
        #     figsize=(20, 10),
        #     xlim=(-1.5, 1.5),
        #     ylim=(-1.5, 1.5)
        # )
        # exit()

        # NOTE: The initialization of the lite masked datasets should happen
        # at the phase level and be passed to analysis
        fits = []
        for i, inversion in enumerate(inversions):

            fits.append(
                self.fit_from_masked_dataset_model_data_and_inversion(
                    masked_dataset=MaskedDatasetLite(
                        visibilities=self.masked_dataset.visibilities[i],
                        noise_map=self.masked_dataset.noise_map[i],
                        noise_map_real_and_imag_averaged=self.masked_dataset.noise_map_real_and_imag_averaged[i],
                        uv_mask=self.masked_dataset.uv_mask[i],
                        uv_mask_real_and_imag_averaged=self.masked_dataset.uv_mask_real_and_imag_averaged[i]
                    ),
                    model_data=inversion.mapped_reconstructed_visibilities,
                    inversion=inversion
                )
            )

        figure_of_merit = sum(
            [fit.figure_of_merit for fit in fits]
        )
        print(figure_of_merit)

        return figure_of_merit

    def inversions_from_instance(self, instance):

        tracer = al.Tracer.from_galaxies(
            galaxies=instance.galaxies
        )

        print(
            "pix = {}".format(
                tracer.source_plane.pixelization.shape
            )
        )
        print(
            "reg = {}".format(
                tracer.source_plane.regularization.coefficient
            )
        )

        mappers_of_planes = tracer.mappers_of_planes_from_grid(
            grid=self.masked_dataset.grid_3d.grid_2d,
            inversion_uses_border=False,
            preload_sparse_grids_of_planes=None
        )

        # TODO: THIS DOES NOT CHANGE ...
        regularization = tracer.source_plane.regularization

        inversions = []
        for i, transformer in enumerate(self.transformers):
            inversions.append(
                inv.InversionInterferometer.from_data_mapper_and_regularization(
                    visibilities=self.masked_dataset.visibilities[i],
                    noise_map=self.masked_dataset.noise_map[i],
                    transformer=transformer,
                    mapper=mappers_of_planes[-1],
                    regularization=regularization
                )
            )

        return inversions

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
