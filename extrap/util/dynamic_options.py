# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from dataclasses import field as d_field
from typing import TypeVar, Generic, Sequence, Type, AnyStr, Mapping, Union, Callable, Any

from extrap.util.unique_list import UniqueList

T = TypeVar('T')


@dataclass
class DynamicOption(Generic[T]):
    value: T
    type: Type
    description: str
    name: str = None
    range: Union[Mapping[AnyStr, T], range] = None
    group = None
    field: str = None
    on_change: Callable[[Any, T], None] = d_field(default=None, compare=False, hash=False)
    explanation_above: str = None


@dataclass
class DynamicOptionsGroup:
    name: str
    options: Sequence[DynamicOption]
    description: str

    def items(self):
        return ((o.field, o) for o in self.options)


def getmembers(object, predicate=None):
    """Return all members of an object as (name, value) pairs sorted by name.
    Optionally, only return members that satisfy a given predicate."""
    if inspect.isclass(object):
        mro = (object,) + inspect.getmro(object)
    else:
        mro = ()
    results = []
    processed = set()
    names = dir(object)
    # :dd any DynamicClassAttributes to the list of names if object is a class;
    # this may result in duplicate entries if, for example, a virtual
    # attribute with the same name as a DynamicClassAttribute exists
    try:
        for base in object.__bases__:
            for k, v in base.__dict__.items():
                if isinstance(v, types.DynamicClassAttribute):
                    names.append(k)
    except AttributeError:
        pass
    for key in names:
        # First try to get the value via getattr.  Some descriptors don't
        # like calling their __get__ (see bug #1785), so fall back to
        # looking in the __dict__.
        try:
            value = getattr(object, key)
            # handle the duplicate key
            if key in processed:
                raise AttributeError
        except AttributeError:
            for base in mro:
                if key in base.__dict__:
                    value = base.__dict__[key]
                    break
            else:
                # could be a (currently) missing slot member, or a buggy
                # __dir__; discard and move on
                continue
        if not predicate or predicate(value):
            results.append((key, value))
        processed.add(key)
    results.sort(key=lambda pair: pair[0])
    return results


def convert_dynamic_options(object_with_options):
    option_storage = []

    names = UniqueList()
    for base in object_with_options.__bases__:
        names.extend(base.__dict__)
        
    names.extend(object_with_options.__dict__)

    for name in names:
        try:
            option: DynamicOption = getattr(object_with_options, name)
            if isinstance(option, DynamicOption):
                if option.group is not None:
                    object_with_options.OPTIONS[option.group.name] = option.group
                else:
                    object_with_options.OPTIONS[name] = option
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

                    setattr(object_with_options, name, make_property(option))
                    option_storage.append(('__OPTION_' + option.field, option.value))
                else:
                    setattr(object_with_options, name, option.value)
        except Exception as e:
            pass
    for n, v in option_storage:
        setattr(object_with_options, n, v)


class DynamicOptions:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'OPTIONS'):
            cls.OPTIONS = {}
        convert_dynamic_options(cls)
        return super().__new__(cls)

    @staticmethod
    def add(value: T, type: Type = str, description: str = None, name: str = None,
            range: Union[Mapping[AnyStr, T], range] = None,
            on_change: Callable[[Any, T], None] = None) -> T:
        """
        Creates a new dynamic option.

        :param value:       The default value of the option
        :param type:        The data type of the value.
                            Must support the conversion from a string if called with a string parameter.
        :param description: A description of the option. Is shown in the command-line help and as tooltip in the GUI.
        :param name:        Alternative display name. Replaces the automatic conversion of the field name.
        :param range:       A range of possible alternatives, either a list, a mapping or a range.
        :param on_change:   Callback function, that is called when the value of the option is changed.
                            Its arguments are the instance self and the new value.
        :return:            A DynamicOption object, that is processed during options creation and replaced by the value.
        """
        return DynamicOption(value=value, type=type, description=description, name=name, range=range,
                             on_change=on_change)

    @staticmethod
    def group(name: str, *options: DynamicOption, description: str = None) -> DynamicOptionsGroup:
        """
        Groups options under the given name and adds an optional description for the group.
        :param name:        Name of the group. Is shown in the command-line help and in the GUI.
        :param options:     One or more options to group.
        :param description: A description of the option. Is shown as tooltip in the GUI.
        """
        group = DynamicOptionsGroup(name=name, description=description, options=options)
        for o in options:
            o.group = group
        return group

    def options_equal(self, other: DynamicOptions) -> bool:
        if not isinstance(other, DynamicOptions):
            return NotImplemented
        if self is other:
            return True
        elif self.OPTIONS != other.OPTIONS:
            return False
        for o in self.options_iter():
            if getattr(self, o.field) != getattr(other, o.field):
                return False
        return True

    def options_iter(self):
        for o in self.OPTIONS.values():
            if isinstance(o, DynamicOptionsGroup):
                for go in o.options:
                    yield go
            else:
                yield o
