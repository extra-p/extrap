# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from sympy import sympify, AccumBounds
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.sets.setexpr import SetExpr


class log2(log):

    @classmethod
    def eval(cls, arg, base=2):
        assert (base == 2)
        arg = sympify(arg)

        if arg.is_Number:
            return log(arg, 2)

        if arg.is_Pow and arg.base is S(2) and arg.exp.is_extended_real:
            return arg.exp

        if isinstance(arg, AccumBounds):
            if arg.min.is_positive:
                return AccumBounds(log2(arg.min), log2(arg.max))
            else:
                return
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)

        # don't autoexpand Pow or Mul (see the issue 3351):
        if not arg.is_Add:
            coeff = arg.as_coefficient(S.ImaginaryUnit)
            if coeff is not None:
                return log(arg, 2)

        if arg.is_number and arg.is_algebraic:
            # Match arg = coeff*(r_ + i_*I) with coeff>0, r_ and i_ real.
            return log(arg, 2)
