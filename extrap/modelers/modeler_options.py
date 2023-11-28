# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import TypeVar, Type, AnyStr, Mapping, Union, Callable, Any

from extrap.modelers.abstract_modeler import AbstractModeler
from extrap.util.dynamic_options import DynamicOption, DynamicOptionsGroup, convert_dynamic_options

T = TypeVar('T')

ModelerOption = DynamicOption
ModelerOptionsGroup = DynamicOptionsGroup


class _ModelerOptionsClass:
    def __call__(self, original_class: AbstractModeler) -> AbstractModeler:
        if hasattr(original_class, 'OPTIONS'):
            original_class.OPTIONS = dict(getattr(original_class, 'OPTIONS'))
        else:
            original_class.OPTIONS = {}

        convert_dynamic_options(original_class)
        original_class.options_iter = self.iter
        original_class.options_equal = self.equal

        return original_class

    @staticmethod
    def add(value: T, type: Type = str, description: str = None, name: str = None,
            range: Union[Mapping[AnyStr, T], range] = None,
            on_change: Callable[[Any, T], None] = None) -> T:
        """
        Creates a new modeler option.

        :param value:       The default value of the option
        :param type:        The data type of the value.
                            Must support the conversion from a string if called with a string parameter.
        :param description: A description of the option. Is shown in the command-line help and as tooltip in the GUI.
        :param name:        Alternative display name. Replaces the automatic conversion of the field name.
        :param range:       A range of possible alternatives, either a list, a mapping or a range.
        :param on_change:   Callback function, that is called when the value of the option is changed.
                            Its arguments are the instance self and the new value.
        :return:            A ModelerOption object, that is processed during options creation and replaced by the value.
        """
        return ModelerOption(value=value, type=type, description=description, name=name, range=range,
                             on_change=on_change)

    @staticmethod
    def group(name: str, *options: ModelerOption, description: str = None) -> ModelerOptionsGroup:
        """
        Groups options under the given name and adds an optional description for the group.
        :param name:        Name of the group. Is shown in the command-line help and in the GUI.
        :param options:     One or more options to group.
        :param description: A description of the option. Is shown as tooltip in the GUI.
        """
        group = ModelerOptionsGroup(name=name, description=description, options=options)
        for o in options:
            o.group = group
        return group

    @staticmethod
    def equal(self: AbstractModeler, other: AbstractModeler) -> bool:
        if not isinstance(other, AbstractModeler):
            return NotImplemented
        elif self is other:
            return True
        elif hasattr(self, 'OPTIONS') != hasattr(other, 'OPTIONS'):
            return False
        elif not hasattr(self, 'OPTIONS'):
            return True
        elif self.OPTIONS != other.OPTIONS:
            return False
        for o in modeler_options.iter(self):
            if getattr(self, o.field) != getattr(other, o.field):
                return False
        return True

    @staticmethod
    def iter(modeler):
        if not hasattr(modeler, 'OPTIONS'):
            return
        for o in getattr(modeler, 'OPTIONS').values():
            if isinstance(o, ModelerOptionsGroup):
                for go in o.options:
                    yield go
            else:
                yield o


modeler_options = _ModelerOptionsClass()
