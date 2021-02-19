# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import json
from abc import abstractmethod, ABC
from typing import Iterator

from extrap.util.classproperty import classproperty


class NamedEntity(ABC):
    @classproperty
    @abstractmethod
    def TYPENAME(cls) -> str:  # noqa
        """
        Name of the subtype
        """
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def ID_COUNTER(cls) -> Iterator[int]:  # noqa
        """
        Counter for global subtype ids
        """
        raise NotImplementedError

    def __init__(self, name):
        self.name = name
        self.id = next(type(self).ID_COUNTER)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self is other or self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{type(self).TYPENAME}({self.name})"


_DATA_SEPARATOR = '\x03'


class NamedEntityWithTags(NamedEntity, ABC):

    def __init__(self, name, **tags):
        super(NamedEntityWithTags, self).__init__(None)
        self.tags = tags or {}
        self._data = name

    @property
    def _data(self):
        if self.tags:
            return self.name + _DATA_SEPARATOR + json.dumps(self.tags)
        else:
            return self.name

    @_data.setter
    def _data(self, val: str):
        if _DATA_SEPARATOR in val:
            self.name, tag_string = val.split(_DATA_SEPARATOR, 1)
            self.tags = json.loads(tag_string)
        else:
            self.name = val

    def __repr__(self):
        if self.tags:
            return f"{type(self).TYPENAME}({self.name}:{self.tags})"
        else:
            return super().__repr__()
