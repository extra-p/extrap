# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Union

import numpy as np
from marshmallow import fields

from extrap.entities.functions import FunctionSchema, TermlessFunctionSchema, TermlessFunction
from extrap.entities.parameter import Parameter


class ComparisonFunction(TermlessFunction):
    def __init__(self, functions):
        super().__init__()
        self.functions = functions

    def evaluate(self, parameter_value):
        """
        Evaluate the function according to the given value and return the result.
        """
        # if isinstance(parameter_value, Mapping):
        #     pos_hash = self._pos_hash(sum(parameter_value.values()))
        # else:
        #     pos_hash = self._pos_hash(parameter_value)
        function_value = [f.evaluate(parameter_value) for f in self.functions]
        # if isinstance(parameter_value, np.ndarray):
        #     function_value = np.array(function_value)
        #     if len(parameter_value.shape) == 2:
        #         pos_hash = self._pos_hash(np.sum(parameter_value, axis=0))

        # selected_function = np.linalg.norm(pos_hash) % len(self.functions)
        # if isinstance(function_value, np.ndarray):
        #     if len(function_value.shape) == 2:
        #         return function_value[selected_function, :]
        #     else:
        #         return function_value[selected_function]
        return np.max(function_value)

    def _pos_hash(self, pos):
        pos = np.cast['int32'](pos)
        hash = 0
        for i in range(4):
            hash += pos & (0xFF << i * 8)
            hash += hash << 10
            hash ^= hash >> 6

        hash += hash << 3
        hash ^= hash >> 11
        hash += hash << 15
        return hash

    def to_string(self, *parameters: Union[str, Parameter]):
        """
        Return a string representation of the function.
        """
        return '(' + ', '.join(t.to_string(*parameters) for t in self.functions) + ')'


class ComparisonFunctionSchema(TermlessFunctionSchema):
    constant_coefficient = fields.Constant(None, load_only=True, dump_only=True)

    def create_object(self):
        return ComparisonFunction(None)

    def postprocess_object(self, obj: object) -> object:
        return obj

    functions = fields.List(fields.Nested(FunctionSchema))
