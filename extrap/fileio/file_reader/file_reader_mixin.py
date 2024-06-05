# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.entities.scaling_type import ScalingType
from extrap.util.dynamic_options import DynamicOptions


class TKeepValuesReader(DynamicOptions):
    keep_values: ScalingType = DynamicOptions.add(False, bool)
    keep_values.explanation_below = ("Keeps the individual measurement values.\n"
                                     "Required for measurement point suggestions.")
