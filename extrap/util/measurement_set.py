# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2024-2025, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Sequence, Callable

from extrap.entities.measurement import Measurement


def sum(*measurement_sets: Sequence[Measurement]) -> Sequence[Measurement]:
    return aggregate(Measurement.__add__, *measurement_sets)


def subtract(*measurement_sets: Sequence[Measurement]) -> Sequence[Measurement]:
    return aggregate(Measurement.__sub__, *measurement_sets)


def multiply_factor(factor, measurement_set: Sequence[Measurement]) -> Sequence[Measurement]:
    data = []
    if not measurement_set:
        return []
    for m in measurement_set:
        data.append(m * factor)
    return data


def aggregate(binary_operator: Callable[[Measurement, Measurement], Measurement],
              *measurement_sets: Sequence[Measurement]):
    rest = iter(measurement_sets)
    first = next(rest)
    data = {}
    if not first:
        return []
    for m in first:
        data[m.coordinate] = m.copy()
    for m_set in rest:
        if not m_set:
            continue
        for m in m_set:
            if m.coordinate not in data:
                data[m.coordinate] = m.copy()
                data[m.coordinate].callpath = first.callpath
            else:
                data[m.coordinate] = binary_operator(data[m.coordinate], m)
    return list(data.values())
