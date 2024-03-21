# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.util.dynamic_options import DynamicOptions


class TKeepValuesReader(DynamicOptions):
    keep_values: bool = DynamicOptions.add(False, bool)
