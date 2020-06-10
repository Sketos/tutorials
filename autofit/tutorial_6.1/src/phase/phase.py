import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

import autofit as af
import autolens as al

sys.path.append(
    "{}/tutorials/autofit/tutorial_6".format(
        os.environ["GitHub"]
    )
)

from src.dataset.dataset import (
    Dataset, MaskedDataset, MaskedDatasetLite
)
from src.phase.result import (
    Result,
)
from src.phase.analysis import (
    Analysis,
)


# def regions_overlap(regions):
#     pass
#
#
#     # TODO: Check that all regions have the same number of channels.
#     # NOTE: Have another arguments which is the n_channels of the data
#     # and compare to that
#     if all(region.n_channels==regions[0].n_channels for region in regions):
#         pass
#     else:
#         raise ValueError(
#             "The individual regions do not have the same number of channels."
#             )
#
#     # TODO: Return True if regions overlap, otherwise return False. Iterate through each
#     # element of the 1D arrays and if there are more than one values that are True then
#     # the regions overlap.

def reshape_array(array):

    return array.reshape(
        -1,
        array.shape[-1]
    )


class Phase(af.AbstractPhase):

    profiles = af.PhaseProperty("profiles")

    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        *,
        profiles,
        lens_redshift,
        source_redshift,
        regions=[],
        non_linear_class=af.MultiNest,
        transformer_class=al.TransformerFINUFFT
    ):

        super().__init__(paths=paths, non_linear_class=non_linear_class)

        self.profiles = profiles

        if lens_redshift < source_redshift:
            self.lens_redshift = lens_redshift
            self.source_redshift = source_redshift
        else:
            raise ValueError(
                "The len's z={} must be lower than the source's z={}",format(
                    lens_redshift, source_redshift
                )
            )

        self.transformer_class = transformer_class

        if not isinstance(regions, list):
            raise ValueError(
                """The variable "regions" must be a list."""
            )
        else:
            self.regions = regions

    @property
    def phase_folders(self):
        return self.optimizer.phase_folders

    def run(self, dataset: Dataset, xy_mask):

        analysis = self.make_analysis(
            dataset=dataset,
            xy_mask=xy_mask
        )

        result = self.run_analysis(analysis=analysis)

        return self.make_result(
            result=result,
            analysis=analysis
        )

    def make_analysis(self, dataset, xy_mask):

        masked_dataset = MaskedDataset(
            dataset=dataset,
            xy_mask=xy_mask
        )

        transformers = []
        for i in range(masked_dataset.uv_wavelengths.shape[0]):
            transformers.append(
                self.transformer_class(
                    uv_wavelengths=masked_dataset.uv_wavelengths[i],
                    grid=masked_dataset.grid_3d.grid_2d.in_radians
                )
            )


        def get_continuum(masked_dataset, regions):
            pass

        continuum = np.zeros(
            shape=(dataset.visibilities.shape[0], ),
            dtype=bool
        )
        for region in self.regions:
            continuum += region


        def func(masked_dataset, continuum):

            argspec = inspect.getargspec(MaskedDatasetLite.__init__)
            args = {}
            for argname in argspec.args:
                if argname not in ["self"]:
                    if hasattr(masked_dataset, argname):
                        array = getattr(masked_dataset, argname)
                        args[argname] = reshape_array(
                            array=array[~continuum]
                        )

            return MaskedDatasetLite(**args)

        masked_dataset_continuum = func(masked_dataset, continuum)
        #print(masked_dataset_continuum.uv_wavelengths.shape)

        transformer_continuum = self.transformer_class(
            uv_wavelengths=masked_dataset_continuum.uv_wavelengths,
            grid=masked_dataset.grid_3d.grid_2d.in_radians
        )

        # dirty_image = transformer_continuum.image_from_visibilities(
        #     visibilities=masked_dataset_continuum.visibilities
        # )
        # plt.figure()
        # plt.imshow(dirty_image[::-1], cmap="jet")
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()


        return Analysis(
            masked_dataset=masked_dataset,
            masked_datasets={
                "continuum":masked_dataset_continuum
            },
            transformers=transformers,
            transformer_continuum=transformer_continuum,
            lens_redshift=self.lens_redshift,
            source_redshift=self.source_redshift,
            image_path=self.optimizer.paths.image_path
        )

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            likelihood=result.likelihood,
            analysis=analysis,
            output=result.output,
        )
