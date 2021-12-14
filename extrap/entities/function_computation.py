# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import copy
import operator
import re
from enum import Enum
from numbers import Number
from token import ERRORTOKEN, NUMBER, NAME
from typing import Union, Optional, Sequence, Mapping, Tuple, cast

import numpy
import sympy
from marshmallow import fields
from sympy.printing.precedence import precedence
from sympy.printing.str import StrPrinter

from extrap.entities.calculation_element import CalculationElement
from extrap.entities.functions import TermlessFunction, Function, TermlessFunctionSchema, SingleParameterFunction, \
    MultiParameterFunction, ConstantFunction
from extrap.entities.parameter import Parameter
from extrap.entities.terms import DEFAULT_PARAM_NAMES
from extrap.util import sympy_functions

PARAM_TOKEN = '\x1B'


class CFType(Enum):
    NONE = 0x0
    SINGLE_PARAMETER = 0x1
    MULTI_PARAMETER = 0x2
    SINGLE_MULTI_PARAMETER = 0x3

    def __eq__(self, other):
        if not isinstance(other, CFType):
            return NotImplemented
        elif self is other:
            return True
        elif self is CFType.SINGLE_MULTI_PARAMETER:
            return other is CFType.MULTI_PARAMETER or other is CFType.SINGLE_PARAMETER
        elif other is CFType.SINGLE_MULTI_PARAMETER:
            return self is CFType.MULTI_PARAMETER or self is CFType.SINGLE_PARAMETER
        else:
            return False

    def __and__(self, other):
        if not isinstance(other, CFType):
            return NotImplemented
        elif self is CFType.SINGLE_MULTI_PARAMETER:
            return other
        elif other is CFType.SINGLE_MULTI_PARAMETER:
            return self
        elif self is other:
            return self
        else:
            return CFType.NONE


def _make_op(op):
    def _op(self: ComputationFunction, other):
        result = ComputationFunction(None)

        if isinstance(other, Function):
            if isinstance(other, ComputationFunction):
                ftype = other._ftype
                result.sympy_function = op(self.sympy_function, other.sympy_function)
            else:
                params1, ftype = self._determine_params(other)
                params = params1 if len(params1) > len(self._params) else self._params
                result.sympy_function = op(self.sympy_function, other.evaluate(params))
            if self._ftype != ftype:
                raise ValueError(f"Cannot {op.__name__} a single parameter and a multi parameter function. "
                                 f"Both functions need to be of the same type.")
            result._ftype = self._ftype & ftype
        else:
            result.sympy_function = op(self.sympy_function, other)
            result._ftype = self._ftype
        return result

    return _op


def _make_rop(op):
    def _op(self, other):
        result = ComputationFunction(None)

        if isinstance(other, Function):
            if isinstance(other, ComputationFunction):
                ftype = other._ftype
                result.sympy_function = op(other.sympy_function, self.sympy_function)
            else:
                params1, ftype = self._determine_params(other)
                params = params1 if len(params1) > len(self._params) else self._params
                result.sympy_function = op(other.evaluate(params), self.sympy_function)
            if self._ftype != ftype:
                raise ValueError(f"Cannot {op.__name__} a single parameter and a multi parameter function. "
                                 f"Both functions need to be of the same type.")
            result._ftype = self._ftype & ftype
        else:
            result.sympy_function = op(other, self.sympy_function)
            result._ftype = self._ftype
        return result

    return _op


class FunctionPrinter(StrPrinter):
    """Print derivative of a function of symbols in a shorter form.
    """

    def _print_Pow(self, expr, rational=False):
        PREC = precedence(expr)
        result = f'{self.parenthesize(expr.base, PREC, strict=False)}^({expr.exp})'
        return result

    def _print_Add(self, expr, order=None):
        if self._print_level == 1:
            return super()._print_Add(expr, order='ilex')
        return super()._print_Add(expr, order)

    def _print_Mul(self, expr):
        result = super(FunctionPrinter, self)._print_Mul(expr)
        result = re.sub(r'(\S)\*', r'\1 * ', result)
        return result


class ComputationFunction(TermlessFunction, CalculationElement):
    PRINTER = FunctionPrinter()

    def __init__(self, function: Optional[Function]):
        super().__init__()
        self.original_function: Optional[Function] = function
        self._params = cast(Tuple[sympy.Symbol], ())
        self._ftype = CFType.NONE
        self._sympy_function: Optional[sympy.Expr] = None
        if not self.original_function:
            return
        _params, self._ftype = self._determine_params(self.original_function)
        if isinstance(self.original_function, ComputationFunction):
            self.sympy_function = self.original_function.sympy_function
        else:
            self.sympy_function = self.original_function.evaluate(_params)

    @property
    def constant_coefficient(self):
        return 0

    @constant_coefficient.setter
    def constant_coefficient(self, val):
        if val != 0:
            raise ValueError('Constant coefficient can only be 0.')

    @property
    def sympy_function(self) -> sympy.Expr:
        return self._sympy_function

    @sympy_function.setter
    def sympy_function(self, val: sympy.Expr):
        self._sympy_function = sympy.sympify(val)
        max_param_id = max((int(s.name[1:]) for s in self._sympy_function.atoms(sympy.Symbol)), default=-1)
        if self._ftype == CFType.SINGLE_PARAMETER and max_param_id > 0:
            raise ValueError("This is a single parameter function, you cannot add a multi parameter sympy function.")
        self._params = sympy.symbols(tuple(PARAM_TOKEN + str(i) for i in range(max_param_id + 1)))
        self._evaluation_function = sympy.lambdify(self._params, self.sympy_function, 'numpy')

    def to_string(self, *parameters: Union[str, Parameter]):
        if not parameters:
            parameters = DEFAULT_PARAM_NAMES
        result = self.sympy_function
        for param, new_param in zip(self._params, parameters):
            result = result.subs(param, str(new_param))

        return self.PRINTER.doprint(result)

    def evaluate(self, parameter_value: Union[Number, numpy.ndarray, Mapping[int, Union[Number, numpy.ndarray]],
                                              Sequence[Union[Number, numpy.ndarray]]]) -> Union[Number, numpy.ndarray]:
        if not self._params:
            if isinstance(parameter_value, numpy.ndarray):
                shape = parameter_value.shape
                if len(shape) == 2:
                    shape = (shape[1],)
                return numpy.full(shape, self._evaluation_function())
            else:
                return self._evaluation_function()

        if self._ftype is CFType.SINGLE_PARAMETER:  # Handle single parameter
            if hasattr(parameter_value, '__len__') and (
                    len(parameter_value) == 1 or isinstance(parameter_value, Mapping)):
                parameter_value = parameter_value[0]
            return self._evaluation_function(parameter_value)

        if isinstance(parameter_value, Mapping):
            parameter_value = [parameter_value[i] for i in range(len(self._params))]

        if len(parameter_value) > len(self._params):
            parameter_value = parameter_value[:len(self._params)]
        return self._evaluation_function(*parameter_value)

    def __repr__(self):
        return f"SympyFunction({self.to_string()})"

    __add__ = _make_op(operator.__add__)
    __sub__ = _make_op(operator.__sub__)
    __mul__ = _make_op(operator.__mul__)
    __truediv__ = _make_op(operator.__truediv__)

    __radd__ = _make_rop(operator.__add__)
    __rsub__ = _make_rop(operator.__sub__)
    __rmul__ = _make_rop(operator.__mul__)
    __rtruediv__ = _make_rop(operator.__truediv__)

    def __neg__(self):
        result = copy.copy(self)
        result.original_function = None
        result.sympy_function = -result.sympy_function
        return result

    def _determine_params(self, function) -> Tuple[Tuple[sympy.Symbol], CFType]:
        if not function:
            num_params = 0
            ftype = CFType.NONE
        elif isinstance(function, SingleParameterFunction):
            num_params = 1
            ftype = CFType.SINGLE_PARAMETER
        elif isinstance(function, MultiParameterFunction):
            num_params = max([k for t in function.compound_terms for k, v in t.parameter_term_pairs]) + 1
            ftype = CFType.MULTI_PARAMETER
        elif isinstance(function, ConstantFunction):
            num_params = 0
            ftype = CFType.SINGLE_MULTI_PARAMETER
        elif isinstance(function, ComputationFunction):
            return function._params, function._ftype
        else:
            raise NotImplementedError(f"Cannot determine parameters for function of type {type(function).__name__}")
        return sympy.symbols(tuple(PARAM_TOKEN + str(i) for i in range(num_params))), ftype

    def __eq__(self, other):
        if not isinstance(other, ComputationFunction):
            return NotImplemented
        elif self is other:
            return True
        else:
            return self._sympy_function.evalf(15) == other._sympy_function.evalf(15) and \
                   self._ftype is other._ftype


def param_transformation(tokens, local_dict, global_dict):
    result = []
    seen_param_token = False

    for tok_num, tok_val in tokens:
        if tok_num == ERRORTOKEN and tok_val == PARAM_TOKEN:
            seen_param_token = True
            continue
        elif seen_param_token and tok_num == NUMBER:
            result.append((NAME, PARAM_TOKEN + str(tok_val)))
        elif seen_param_token and tok_num != NUMBER:
            result.append((ERRORTOKEN, PARAM_TOKEN))
            result.append((tok_num, tok_val))
        else:
            result.append((tok_num, tok_val))
        seen_param_token = False
    return result


class ComputationFunctionSchema(TermlessFunctionSchema):
    sympy_function = fields.Function(
        lambda x: sympy.srepr(x.sympy_function),
        lambda x: sympy.parse_expr(x, local_dict={'log2': sympy_functions.log2}))
    _ftype = fields.Function(
        lambda x: x._ftype.name.lower(),
        lambda x: CFType[x.upper()]
    )
    constant_coefficient = fields.Constant(0, load_only=True, dump_only=True)

    def create_object(self):
        return ComputationFunction(None)

    # def postprocess_object(self, obj: object) -> object:
    #     pass
