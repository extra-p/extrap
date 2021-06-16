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

    @property
    def mangled_name(self) -> str:
        if '__mangled_name' not in self.tags:
            return ''
        return self.tags['__mangled_name']

    @mangled_name.setter
    def mangled_name(self, val: str):
        if not val:
            del self.tags['__mangled_name']
        else:
            self.tags['__mangled_name'] = val


CallpathSchema = make_value_schema(Callpath, '_data')
Callpath.EMPTY = Callpath("")
