import os
import sys

import autofit as af
import autolens as al

sys.path.append(
    "{}/tutorials/autofit/tutorial_7".format(
        os.environ["GitHub"]
    )
)

from src.dataset.dataset import (
    Dataset, MaskedDataset, RegionMaskedDataset
)
from src.phase.result import (
    Result,
)
from src.phase.analysis import (
    Analysis,
)

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import variable_utils as variable_utils


def reshape_array(array):

    return array.reshape(
        -1,
        array.shape[-1]
    )


class RegionMaskedDatasetsHolder:
    def __init__(self, region_masked_datasets):

        if not isinstance(region_masked_datasets, list):
            raise ValueError(
                "\"{}\" must be a list".format(
                    variable_utils.variable_name(
                        region_masked_datasets,
                        globals()
                    )[0]
                )
            )
        else:
            self.region_masked_datasets = region_masked_datasets

        self.idx = [region_masked_dataset.continuum
            for region_masked_dataset
            in self.region_masked_datasets
        ]

    @property
    def masked_dataset_continuum(self):
        pass




class Phase(af.AbstractPhase):

    galaxies = af.PhaseProperty("galaxies")

    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        galaxies,
        region,
        non_linear_class=af.MultiNest,
        transformer_class=al.TransformerFINUFFT
    ):

        super().__init__(paths=paths, non_linear_class=non_linear_class)

        self.galaxies = galaxies
        self.region = region
        self.transformer_class = transformer_class

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

        # NOTE: Multiple lines can be present in a cube, in which
        # casa region will be a list (renamed to regions)
        masked_dataset = MaskedDataset(
            dataset=dataset,
            xy_mask=xy_mask,
            region=self.region
        )

        # print(masked_dataset.uv_wavelengths_outside_region.shape)
        # print(masked_dataset.visibilities_outside_region.shape)
        # print(masked_dataset.noise_map_outside_region.shape)
        # print(masked_dataset.uv_wavelengths_inside_region.shape)
        # print(masked_dataset.visibilities_inside_region.shape)
        # print(masked_dataset.noise_map_inside_region.shape)

        masked_dataset_continuum = RegionMaskedDataset(
            dataset=masked_dataset.dataset_outside_region,
            uv_mask=masked_dataset.uv_mask_outside_region,
            continuum=True
        )

        masked_dataset_line = RegionMaskedDataset(
            dataset=masked_dataset.dataset_inside_region,
            uv_mask=masked_dataset.uv_mask_inside_region,
            continuum=False
        )

        # print(masked_dataset_continuum.visibilities.shape)
        # print(masked_dataset_continuum.uv_wavelengths.shape)
        # print(masked_dataset_continuum.noise_map.shape)
        # print(masked_dataset_continuum.uv_mask.shape)
        # print(masked_dataset_line.visibilities.shape)
        # print(masked_dataset_line.uv_wavelengths.shape)
        # print(masked_dataset_line.noise_map.shape)
        # print(masked_dataset_line.uv_mask.shape)
        # exit()

        transformers = []
        for i in range(masked_dataset.uv_wavelengths.shape[0]):
            transformers.append(
                self.transformer_class(
                    uv_wavelengths=masked_dataset.uv_wavelengths[i],
                    grid=masked_dataset.grid_3d.grid_2d.in_radians
                )
            )

        transformer_continuum = self.transformer_class(
            uv_wavelengths=masked_dataset_continuum.uv_wavelengths,
            grid=masked_dataset.grid_3d.grid_2d.in_radians
        )

        # # NOTE: EXPERIMENTAL
        # holder = RegionMaskedDatasetsHolder(
        #     region_masked_datasets=[
        #         masked_dataset_continuum,
        #         masked_dataset_line
        #     ]
        # )
        # exit()

        # TODO: region_masked_datasets can be a class that holds individual
        # masked datasets and it's only function will be to differentiate between
        # a masked_dataset corresponding to the continuum and the rest.
        return Analysis(
            masked_dataset=masked_dataset,
            region_masked_datasets=[
                masked_dataset_continuum,
                masked_dataset_line
            ],
            transformers=transformers,
            transformer_continuum=transformer_continuum,
            image_path=self.optimizer.paths.image_path
        )

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            likelihood=result.likelihood,
            analysis=analysis,
            output=result.output,
        )
