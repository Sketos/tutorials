import os
import sys

import autofit as af

sys.path.append(
    "{}/tutorials/autofit/tutorial_1".format(
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
    def __init__(self, paths, profiles, non_linear_class=af.MultiNest):

        super().__init__(paths=paths, non_linear_class=non_linear_class)

        self.profiles = profiles

    def run(self, dataset: Dataset, mask):

        analysis = self.make_analysis(
            dataset=dataset,
            mask=mask
        )

        result = self.run_analysis(analysis=analysis)

        return self.make_result(
            result=result,
            analysis=analysis
        )

    def make_analysis(self, dataset, mask):

        masked_dataset = MaskedDataset(
            dataset=dataset,
            mask=mask
        )

        return Analysis(
            masked_dataset=masked_dataset, image_path=self.optimizer.paths.image_path
        )

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            likelihood=result.likelihood,
            analysis=analysis,
            output=result.output,
        )
