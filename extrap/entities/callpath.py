# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools

from extrap.entities.named_entity import NamedEntityWithTags
from extrap.util.serialization_schema import make_value_schema


class Callpath(NamedEntityWithTags):
    """
    This class represents a callpath of an application.
    """

    TYPENAME = 'Callpath'

    ID_COUNTER = itertools.count()
    """
    Counter for global callpath ids
    """

    EMPTY: 'Callpath'
    """
    Empty callpath. Can be used as placeholder.
    """

    def concat(self, *other, copy_tags=False):
        cp = Callpath('->'.join(itertools.chain((self.name,), other)))
        if copy_tags:
            cp.tags = self.tags.copy()
        return cp


class _EmptyCallpath(Callpath):

    def __init__(self):
        self.__is_init = True
        super().__init__("")
        self.__is_init = False

    def __copy__(self):
        raise NotImplementedError()

    @property
    def name(self):
        return ""

    def __hash__(self):
        return 0

    @name.setter
    def name(self, val):
        if not self.__is_init:
            raise NotImplementedError()

    @property
    def tags(self):
        return {}

    @tags.setter
    def tags(self, val):
        if not self.__is_init:
            raise NotImplementedError()

    def concat(self, *other, copy_tags=False):
        raise NotImplementedError()


CallpathSchema = make_value_schema(Callpath, '_data')
Callpath.EMPTY = _EmptyCallpath()
