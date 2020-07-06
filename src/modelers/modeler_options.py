"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""
from collections import namedtuple
from dataclasses import dataclass
from typing import TypeVar, Generic, Sequence, Type, Collection, AnyStr, Mapping, Union

from modelers.abstract_modeler import AbstractModeler

T = TypeVar('T')


@dataclass
class ModelerOption(Generic[T]):
    value: T
    type: Type
    description: str
    name: str = None
    range: Union[Mapping[AnyStr, T], range] = None
    group = None
    field: str = None


@dataclass
class ModelerOptionsGroup:
    name: str
    options: Sequence[ModelerOption]
    description: str

    def items(self):
        return ((o.field, o) for o in self.options)


def modeler_options(original_class: AbstractModeler) -> AbstractModeler:
    if hasattr(super(original_class), 'OPTIONS'):
        original_class.OPTIONS = dict(getattr(super(original_class), 'OPTIONS'))
    else:
        original_class.OPTIONS = {}

    for name in original_class.__dict__:
        option: ModelerOption = original_class.__dict__[name]
        if isinstance(option, ModelerOption):
            if option.group is not None:
                original_class.OPTIONS[option.group.name] = option.group
            else:
                original_class.OPTIONS[name] = option
            option.field = name
            setattr(original_class, name, option.value)

    return original_class


def _add_option(value: T, type: Type = str, description: str = None, name: str = None,
                range: Union[Mapping[AnyStr, T], range] = None) -> T:
    return ModelerOption(value=value, type=type, description=description, name=name, range=range)


def _group_options(name: str, *options: ModelerOption, description: str = None) -> ModelerOptionsGroup:
    group = ModelerOptionsGroup(name=name, description=description, options=options)
    for o in options:
        o.group = group
    return group


modeler_options.add = _add_option
modeler_options.group = _group_options
