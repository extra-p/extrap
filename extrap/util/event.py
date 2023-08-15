# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from inspect import signature, Parameter
from typing import Callable, Any, Type, Union

from typing_extensions import TypeVarTuple, Unpack

_Ts = TypeVarTuple('_Ts')


class Event(Callable[[Unpack[_Ts]], None]):

    def __init__(self, *args: Union[Type[Unpack[_Ts]], type]):
        if args:
            if isinstance(args[0], int):
                self.len_args = args[0]
            else:
                self.len_args = len(args)
        else:
            self.len_args = None
        self._event_listeners: set[Callable[[Unpack[_Ts]], Any]] = set()

    def __iadd__(self, event_listener: Callable[[Unpack[_Ts]], Any]):
        if self.len_args is not None:
            parameters = signature(event_listener).parameters
            len_el_args = len(parameters)
            if len_el_args != self.len_args and (len_el_args != 1 or next(
                    iter(parameters.values())).kind != Parameter.VAR_POSITIONAL):
                raise ValueError(f"The event listener takes {len_el_args} instead of {self.len_args} arguments.")
        self._event_listeners.add(event_listener)
        return self

    def __isub__(self, other: Callable[[Unpack[_Ts]], Any]):
        if other in self._event_listeners:
            self._event_listeners.remove(other)
        return self

    def __call__(self, *args: Unpack[_Ts]) -> None:
        for el in self._event_listeners:
            el(*args)
