# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from typing import Union, TypeVar

from typing_extensions import Protocol


class CalculationElement(Protocol):
    def __add__(self, other) -> CalculationElement:
        pass

    def __sub__(self, other) -> CalculationElement:
        pass

    def __mul__(self, other) -> CalculationElement:
        pass

    def __truediv__(self, other) -> CalculationElement:
        pass

    def __radd__(self, other) -> CalculationElement:
        pass

    def __rmul__(self, other) -> CalculationElement:
        pass

    def __rsub__(self, other) -> CalculationElement:
        pass

    def __rtruediv__(self, other) -> CalculationElement:
        pass

    def __neg__(self) -> CalculationElement:
        pass

    def make_one(self) -> CalculationElement:
        pass


_T = TypeVar('_T')
_S = TypeVar('_S')


def divide_no0(a: _T, b: _S) -> Union[_T, _S, CalculationElement]:
    if a == 0 and b == 0:
        if hasattr(a, 'make_one'):
            return a.make_one()
        if hasattr(b, 'make_one'):
            return b.make_one()
        return 1
    elif b == 0:
        raise ZeroDivisionError()
    else:
        return a / b
