# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from itertools import tee

try:
    from itertools import pairwise
except ImportError:
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


def group_all_by(iterable, key):
    result = {}
    for element in iterable:
        k = key(element)
        if k not in result:
            result[k] = [element]
        else:
            result[k].append(element)
    return result
