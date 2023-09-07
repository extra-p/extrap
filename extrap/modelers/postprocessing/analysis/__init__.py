# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from abc import ABC

from extrap.modelers.postprocessing import PostProcess, PostProcessSchema


class PostProcessAnalysis(PostProcess, ABC):
    """
    A marker class, that should be used to identify post processes as an analysis.

    This information is used to determine that a model set is a result of an analysis and should not be used in another
    analysis because the values/metrics represent the analysis results and not the original values.
    """

    def supports_processing(self, post_processing_history: list[PostProcess]) -> bool:
        return not any(isinstance(p, PostProcessAnalysis) for p in post_processing_history)


class PostProcessAnalysisSchema(PostProcessSchema):
    pass
