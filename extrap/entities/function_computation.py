# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import copy
import numbers
import operator
import re
from enum import Enum
from numbers import Number
from token import ERRORTOKEN, NUMBER, NAME
from typing import Union, Optional, Sequence, Mapping, Tuple

import numpy
import sympy
from marshmallow import fields
from mpmath.libmp import fzero, finf, fninf, fnan, to_digits_exp
from sympy import Float
from sympy.printing.precedence import precedence
from sympy.printing.str import StrPrinter

from extrap.entities.calculation_element import CalculationElement
from extrap.entities.functions import TermlessFunction, Function, TermlessFunctionSchema, SingleParameterFunction, \
    MultiParameterFunction, ConstantFunction
from extrap.entities.parameter import Parameter
from extrap.entities.terms import DEFAULT_PARAM_NAMES
from extrap.util import sympy_functions
from extrap.util.formatting_helper import format_number_html

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


_mul_without_spaces = re.compile(r'(\S)\*')


class FunctionPrinter(StrPrinter):
    """Print derivative of a function of symbols in a shorter form.
    """

    def _print_Pow(self, expr, rational=False):
        PREC = precedence(expr)
        if expr.exp != 1:
            exponent = self._print(expr.exp)
            result = f'{self.parenthesize(expr.base, PREC, strict=False)}^({exponent})'
        else:
            result = self.parenthesize(expr.base, PREC, strict=False)

        return result

    def _print_Add(self, expr, order=None):
        if self._print_level == 1:
            return super()._print_Add(expr, order='ilex')
        return super()._print_Add(expr, order)

    def _print_Mul(self, expr):
        result = super()._print_Mul(expr)
        result = _mul_without_spaces.sub(r'\1 * ', result)
        return result


class HTMLPrinter(StrPrinter):
    _default_settings = {
        "order": None,
        "full_prec": False,
        "sympy_integers": False,
        "abbrev": False,
        "perm_cyclic": True,
        "min": 10 ** -3,
        "max": 10 ** 5,
    }

    def _print_Pow(self, expr, rational=False):
        PREC = precedence(expr)
        if expr.exp != 1:
            exponent = self._print(expr.exp)
            result = f'{self.parenthesize(expr.base, PREC, strict=False)}<sup>{exponent}</sup>'
        else:
            result = self.parenthesize(expr.base, PREC, strict=False)
        return result

    def _print_Add(self, expr, order=None):
        if self._print_level == 1:
            return super()._print_Add(expr, order="grlex")
        return super()._print_Add(expr, order)

    def _print_Mul(self, expr):
        result = super()._print_Mul(expr)
        result = _mul_without_spaces.sub(r'\1 * ', result)
        return result

    def _print_Integer(self, expr):
        return format_number_html(expr.p)

    def _print_int(self, expr):
        return format_number_html(expr)

    def _print_mpz(self, expr):
        return format_number_html(float(expr))

    def _print_Rational(self, expr):
        if expr.q == 1:
            return format_number_html(expr.p)
        else:
            return "%s/%s" % (expr.p, expr.q)

    def _print_PythonRational(self, expr):
        if expr.q == 1:
            return format_number_html(float(expr))
        else:
            return "%d/%d" % (expr.p, expr.q)

    def _print_Fraction(self, expr):
        if expr.denominator == 1:
            return format_number_html(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    def _print_mpq(self, expr):
        if expr.denominator == 1:
            return format_number_html(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    def _print_Float(self, expr: Float):
        s = expr._mpf_
        if not s[1]:
            if s == fzero:
                t = '0'
                return t
            if s == finf: return '+inf'
            if s == fninf: return '-inf'
            if s == fnan: return 'nan'
            raise ValueError

        dps = 4

        sign, digits, exponent = to_digits_exp(s, dps + 3)

        # Rounding up kills some instances of "...99999"
        if len(digits) > dps and digits[dps] in '56789':
            digits = digits[:dps]
            i = dps - 1
            while i >= 0 and digits[i] == '9':
                i -= 1
            if i >= 0:
                digits = digits[:i] + str(int(digits[i]) + 1) + '0' * (dps - i - 1)
            else:
                digits = '1' + '0' * (dps - 1)
                exponent += 1
        else:
            digits = digits[:dps]

        # Prettify numbers close to unit magnitude
        if -4 < exponent < 5:
            if exponent < 0:
                digits = ("0" * int(-exponent)) + digits
                split = 1
            else:
                split = exponent + 1
                if split > dps:
                    digits += "0" * (split - dps)
            exponent = 0
        else:
            split = 1

        digits = (digits[:split] + "." + digits[split:])

        # Clean up trailing zeros
        digits = digits.rstrip('0')
        if digits[-1] == ".":
            digits = digits[:-1]

        rv = sign + digits
        if exponent != 0:
            rv += f"<small>&times;10</small><sup>{exponent}</sup>"

        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]
        if rv.startswith('+'):
            # e.g., +inf -> inf
            rv = rv[1:]
        return rv

    def _print_log2(self, expr):
        return f"log<sub>2</sub>({self.stringify(expr.args, ', ')})"


class ComputationFunction(TermlessFunction, CalculationElement):
    _PRINTER = FunctionPrinter()
    _PRINTER_HTML = HTMLPrinter()

    def __init__(self, function: Optional[Function]):
        super().__init__()
        self.original_function: Optional[Function] = function
        self._params: Tuple[sympy.Symbol, ...] = ()
        self._ftype = CFType.NONE
        self._sympy_function: Optional[sympy.Expr] = None
        self._evaluation_function = None
        if not self.original_function:
            return
        _params, self._ftype = self._determine_params(self.original_function)
        if isinstance(self.original_function, ComputationFunction):
            self.sympy_function = self.original_function.sympy_function
        else:
            self.sympy_function = self.original_function.evaluate(_params)

    @classmethod
    def from_string(cls, expr: str, auto_convert_params=False, ftype: CFType = None) -> ComputationFunction:
        sympy_f: sympy.Expr = sympy.parse_expr(expr, local_dict={'log2': sympy_functions.log2})

        return cls.from_sympy(sympy_f, auto_convert_params, ftype)

    @classmethod
    def from_sympy(cls, sympy_f: sympy.Expr, auto_convert_params=False, ftype: CFType = None) -> ComputationFunction:
        f = ComputationFunction(None)

        if auto_convert_params:
            org_params = sorted(list(sympy_f.free_symbols), key=lambda s: s.name)
            sympy_params = sympy.symbols(tuple(PARAM_TOKEN + str(i) for i in range(len(org_params))))
            f.sympy_function = sympy_f.subs({o: s for o, s in zip(org_params, sympy_params)})
        else:
            f.sympy_function = sympy_f

        if ftype is not None:
            f._ftype = ftype
        elif len(sympy_f.free_symbols) > 1:
            f._ftype = CFType.MULTI_PARAMETER
        else:
            f._ftype = CFType.SINGLE_MULTI_PARAMETER

        return f

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
        max_param_id = max((int(s.name[1:]) for s in self._sympy_function.free_symbols), default=-1)
        if self._ftype == CFType.SINGLE_PARAMETER and max_param_id > 0:
            raise ValueError("This is a single parameter function, you cannot add a multi parameter sympy function.")
        self._params = sympy.symbols(tuple(PARAM_TOKEN + str(i) for i in range(max_param_id + 1)))
        self._evaluation_function = None  # reset evaluation function

    def to_string(self, *parameters: Union[str, Parameter]):
        if not parameters:
            parameters = DEFAULT_PARAM_NAMES
        result = self.sympy_function
        for param, new_param in zip(self._params, parameters):
            result = result.subs(param, sympy.Symbol(str(new_param)))

        return self._PRINTER.doprint(result)

    def to_html(self, *parameters: Union[str, Parameter]):
        if not parameters:
            parameters = DEFAULT_PARAM_NAMES
        result = self.sympy_function
        for param, new_param in zip(self._params, parameters):
            result = result.subs(param, sympy.Symbol(str(new_param)))

        return self._PRINTER_HTML.doprint(result)

    @staticmethod
    def get_param(param_idx: int):
        return sympy.Symbol(PARAM_TOKEN + str(param_idx))

    def evaluate(self, parameter_value: Union[Number, numpy.ndarray, Mapping[int, Union[Number, numpy.ndarray]],
    Sequence[Union[Number, numpy.ndarray]]]) -> Union[Number, numpy.ndarray]:
        if not self._evaluation_function:  # lazy init of evaluation function
            self._evaluation_function = sympy.lambdify(self._params, self.sympy_function, 'numpy')

        if not self._params:  # Handle no parameter
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
            if isinstance(other, numbers.Number):
                return self.sympy_function == other
            return NotImplemented
        elif self is other:
            return True
        else:
            return self._sympy_function.evalf(15) == other._sympy_function.evalf(15) and \
                self._ftype is other._ftype

    def partial_compare(self, other: Function):
        if not isinstance(other, Function):
            return NotImplemented
        if not isinstance(other, ComputationFunction):
            other = ComputationFunction(other)

        params = self._params
        if len(params) < len(other._params):
            params = other._params

        self_func = self._sympy_function
        other_func = other._sympy_function
        # self_func = self._remove_negative_terms(self._sympy_function)
        # other_func = self._remove_negative_terms(other._sympy_function)

        comp_func0 = self_func - other_func
        dummy_params = {p: sympy.Dummy(str(p)[1:], real=True, positive=True) for p in params}
        comp_func = comp_func0.subs((o, n + 1) for o, n in dummy_params.items())

        if comp_func.is_number:
            res = comp_func
        else:
            all_res = []
            for p in params:
                try:
                    all_res.append(sympy.limit(comp_func, dummy_params[p], sympy.oo))
                except RecursionError:
                    all_res.append(sympy.nan)
            for i, r in enumerate(all_res):
                while not r.is_number and r.free_symbols:
                    d = sympy.Dummy(real=True, positive=True, nonzero=True)
                    r = r.subs(next(iter(r.free_symbols)), d + 1)
                    r = sympy.limit(r, d, sympy.oo)
                all_res[i] = r

            if all(r == all_res[0] for r in all_res):
                res = all_res[0]
            else:
                try:
                    return tuple(1 if r > 0 else (0 if r == 0 else -1) for r in all_res)
                except TypeError:
                    return tuple("Cannot evaluate: " + str(r) for r in all_res)

        if res > 0:
            return 1
        elif res == 0:
            return 0
        else:
            return -1

    @classmethod
    def make_one(cls):
        return ComputationFunction(ConstantFunction(1))

    @staticmethod
    def _remove_negative_terms(_sympy_function):
        nodes_to_replace = []
        for node in sympy.preorder_traversal(_sympy_function):
            if isinstance(node, sympy.Mul):
                for arg in node.args:
                    if arg.is_number and (arg < 0) == sympy.S.true:
                        nodes_to_replace.append(node)
                        break

        function_subs = _sympy_function.subs((old, 0) for old in nodes_to_replace)
        return function_subs


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
