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
from typing import TypeVar, Generic, Sequence, Type, Collection, AnyStr, Mapping, Union, Callable, Any

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
    on_change: Callable[[Any, T], None] = None


@dataclass
class ModelerOptionsGroup:
    name: str
    options: Sequence[ModelerOption]
    description: str

    def items(self):
        return ((o.field, o) for o in self.options)


class _ModelerOptionsClass:
    def __call__(self, original_class: AbstractModeler) -> AbstractModeler:
        if hasattr(original_class, 'OPTIONS'):
            original_class.OPTIONS = dict(getattr(original_class, 'OPTIONS'))
        else:
            original_class.OPTIONS = {}

        option_storage = []
        for name in original_class.__dict__:
            option: ModelerOption = original_class.__dict__[name]
            if isinstance(option, ModelerOption):
                if option.group is not None:
                    original_class.OPTIONS[option.group.name] = option.group
                else:
                    original_class.OPTIONS[name] = option
                option.field = name
                if option.on_change is not None:
                    def make_property(opt):
                        # required for closure
                        def getter(m_self):
                            return getattr(m_self, '__OPTION_' + opt.field)

                        def setter(m_self, v):
                            setattr(m_self, '__OPTION_' + opt.field, v)
                            opt.on_change(m_self, v)

                        return property(getter, setter)

                    setattr(original_class, name, make_property(option))
                    option_storage.append(('__OPTION_' + option.field, option.value))
                else:
                    setattr(original_class, name, option.value)
        for n, v in option_storage:
            setattr(original_class, n, v)

        return original_class

    def add(self, value: T, type: Type = str, description: str = None, name: str = None,
            range: Union[Mapping[AnyStr, T], range] = None,
            on_change: Callable[[Any, T], None] = None) -> T:
        return ModelerOption(value=value, type=type, description=description, name=name, range=range,
                             on_change=on_change)

    def group(self, name: str, *options: ModelerOption, description: str = None) -> ModelerOptionsGroup:
        group = ModelerOptionsGroup(name=name, description=description, options=options)
        for o in options:
            o.group = group
        return group


modeler_options = _ModelerOptionsClass()
