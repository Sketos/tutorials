import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

import autofit as af
import autolens as al

sys.path.append(
    "{}/tutorials/autofit/tutorial_0".format(
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

    galaxies = af.PhaseProperty("galaxies")

    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        *,
        galaxies,
        self_calibration=False,
        non_linear_class=af.MultiNest,
        transformer_class=al.TransformerFINUFFT
    ):

        super().__init__(paths=paths, non_linear_class=non_linear_class)

        self.galaxies = galaxies

        self.transformer_class = transformer_class

        self.self_calibration = self_calibration

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

        transformer = self.transformer_class(
            uv_wavelengths=masked_dataset.uv_wavelengths,
            grid=masked_dataset.grid.in_radians
        )

        return Analysis(
            masked_dataset=masked_dataset,
            transformer=transformer,
            self_calibration=self.self_calibration,
            image_path=self.optimizer.paths.image_path
        )

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            likelihood=result.likelihood,
            analysis=analysis,
            output=result.output,
        )
