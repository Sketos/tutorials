import os
import sys

import autofit as af
import autolens as al

sys.path.append(
    "{}/tutorials/autofit/tutorial_4".format(
        os.environ["GitHub"]
    )
)

from src.dataset.dataset import (
    Dataset, MaskedDataset
)
from src.phase.result import (
    Result,
)
from src.phase.analysis import (
    Analysis,
)

class Phase(af.AbstractPhase):

    profiles = af.PhaseProperty("profiles")

    Result = Result

    @af.convert_paths
    def __init__(self, paths, profiles, non_linear_class=af.MultiNest, transformer_class=al.TransformerFINUFFT):

        super().__init__(paths=paths, non_linear_class=non_linear_class)

        self.profiles = profiles

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

        return Analysis(
            masked_dataset=masked_dataset,
            transformers=transformers,
            image_path=self.optimizer.paths.image_path
        )

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            likelihood=result.likelihood,
            analysis=analysis,
            output=result.output,
        )
