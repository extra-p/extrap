# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from enum import Enum


class ScalingType(Enum):
    WEAK = "weak"
    WEAK_THREADED = "weak_threaded"
    STRONG = "strong"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, name: str):
        return cls[name.upper()]
