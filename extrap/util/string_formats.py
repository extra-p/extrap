# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import enum


class FunctionFormats(enum.Enum):
    NONE = None,
    PYTHON = enum.auto()
    LATEX = enum.auto()
