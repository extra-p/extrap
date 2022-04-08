# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
import json
from abc import abstractmethod, ABC
from typing import Iterator, Any, MutableMapping

from marshmallow import fields

from extrap.util.classproperty import classproperty
from extrap.util.deprecation import deprecated
from extrap.util.serialization_schema import Schema

TAG_SEPARATOR = '__'


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

    def __init__(self, name: str):
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

    def copy(self):
        return copy.copy(self)


_DATA_SEPARATOR = '\x03'


class NamedEntityWithTags(NamedEntity, ABC):

    def __init__(self, name, **tags):
        super(NamedEntityWithTags, self).__init__(name)
        self.tags: MutableMapping[str, Any] = tags or {}

    def lookup_tag(self, tag: str, default=None, prefix_len=1, suffix=None):
        if suffix:
            tag_with_suffix = tag + TAG_SEPARATOR + suffix
            if tag_with_suffix in self.tags:
                return self.tags[tag_with_suffix]
            else:
                path = tag.split(TAG_SEPARATOR)
                l_suffix = [suffix]
                for i in range(-1, -len(path) + prefix_len, -1):
                    tag_with_suffix = TAG_SEPARATOR.join(path[:i] + l_suffix)
                    if tag_with_suffix in self.tags:
                        return self.tags[tag_with_suffix]
            # if nothing found with suffix continue with suffix-less tag
        if tag in self.tags:
            return self.tags[tag]
        else:
            path = tag.split(TAG_SEPARATOR)
            for i in range(-1, -len(path) + prefix_len, -1):
                tag = TAG_SEPARATOR.join(path[:i])
                if tag in self.tags:
                    return self.tags[tag]
            return default

    def __repr__(self):
        if self.tags:
            return f"{type(self).TYPENAME}({self.name}:{self.tags})"
        else:
            return super().__repr__()

    def exactly_equal(self, other):
        return self == other and self.tags == other.tags


class NamedEntitySchema(Schema):
    name = fields.Str()


class NamedEntityWithTagsSchema(NamedEntitySchema):
    tags = fields.Mapping(fields.Str())

    def create_object(self):
        return NotImplemented, NamedEntityWithTags

    def postprocess_object(self, obj: object) -> object:
        if _DATA_SEPARATOR in obj.name:
            deprecated.code("Please use separate encoding via tags schema.",
                            "Encountered tags encoded in the name string.")
            obj.name, tag_string = obj.name.split(_DATA_SEPARATOR, 1)
            obj.tags = json.loads(tag_string)
        return obj
