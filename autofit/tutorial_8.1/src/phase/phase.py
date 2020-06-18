import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

import autofit as af
import autolens as al

sys.path.append(
    "{}/tutorials/autofit/tutorial_8.1".format(
        os.environ["GitHub"]
    )
)

from src.dataset.dataset import (
    Dataset, MaskedDataset, RegionMaskedDataset, MaskedDatasetLite
)
from src.phase.result import (
    Result,
)
from src.phase.analysis import (
    Analysis,
)


def reshape_array(array):

    return array.reshape(
        -1,
        array.shape[-1]
    )


class Phase(af.AbstractPhase):

    galaxies = af.PhaseProperty("galaxies")

    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        *,
        galaxies,
        regions,
        non_linear_class=af.MultiNest,
        transformer_class=al.TransformerFINUFFT
    ):

        super().__init__(paths=paths, non_linear_class=non_linear_class)

        self.galaxies = galaxies

        if not isinstance(regions, list):
            raise ValueError(
                """The variable "regions" must be a list."""
            )
        else:
            self.regions = regions

        self.transformer_class = transformer_class

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

        # NOTE: The masked_dataset is no longer being used, instead each compoent
        # (i.e. continuum + emission line regions) each have their own masked_dataset
        # which is initialized here.
        def masked_datasets_from_regions(masked_dataset, regions):

            args = {
                "continuum":{}
            }

            idx = np.zeros(
                shape=(masked_dataset.visibilities.shape[0], ),
                dtype=bool
            )
            for i, region in enumerate(regions):
                idx += region.idx
                args["region_{}".format(i)] = {}

            if all(idx):
                continuum = False
            else:
                continuum = True

            argspec = inspect.getargspec(MaskedDatasetLite.__init__)

            #args = {}
            for argname in argspec.args:
                if argname not in ["self"]:
                    if hasattr(masked_dataset, argname):
                        array = getattr(masked_dataset, argname)

                        if continuum:
                            args["continuum"][argname] = reshape_array(
                                array=array[~idx]
                            )

                        for i, region in enumerate(regions):
                            args["region_{}".format(i)][argname] = array[region.idx]

            masked_datasets = {
                "continuum":MaskedDatasetLite(**args["continuum"]) if continuum else None
            }
            for i, region in enumerate(regions):
                masked_datasets["region_{}".format(i)] = MaskedDatasetLite(**args["region_{}".format(i)])

            return masked_datasets

        # NOTE: Multiple lines can be present in a cube, in which
        # case region will be a list (renamed to regions) - DONE
        # NOTE: Can we skip the initialization of the masked dataset?
        masked_dataset = MaskedDataset(
            dataset=dataset,
            xy_mask=xy_mask,
        )

        masked_datasets = masked_datasets_from_regions(
            masked_dataset=masked_dataset,
            regions=self.regions
        )

        transformers = {}
        for key in masked_datasets.keys():
            if key == "continuum":
                if masked_datasets[key] is not None:
                    transformers[key] = self.transformer_class(
                        uv_wavelengths=masked_datasets[key].uv_wavelengths,
                        grid=masked_dataset.grid_3d.grid_2d.in_radians
                    )
                else:
                    transformers[key] = None
            elif key.startswith("region"):
                region_transformers = []
                for i in range(masked_datasets[key].uv_wavelengths.shape[0]):
                    region_transformers.append(
                        self.transformer_class(
                            uv_wavelengths=masked_datasets[key].uv_wavelengths[i],
                            grid=masked_dataset.grid_3d.grid_2d.in_radians
                        )
                    )
                transformers[key] = region_transformers
            else:
                raise ValueError("...")

        return Analysis(
            masked_datasets=masked_datasets,
            transformers=transformers,
            grid=masked_dataset.grid_3d.grid_2d,
            image_path=self.optimizer.paths.image_path
        )


    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            likelihood=result.likelihood,
            analysis=analysis,
            output=result.output,
        )
