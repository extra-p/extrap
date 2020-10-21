# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

class classproperty(object):  # noqa
    """Class property decorator.

    Read only."""

    def __init__(self, func):
        self._func = func

    def __get__(self, owner_self, owner_cls):
        return self._func(owner_cls)
