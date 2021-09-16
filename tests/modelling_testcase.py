# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest
from operator import itemgetter

from extrap.entities.functions import MultiParameterFunction
from extrap.entities.terms import CompoundTerm, MultiParameterTerm, SimpleTerm


class TestCaseWithFunctionAssertions(unittest.TestCase):
    def assertApprox(self, function, other, places=6, ctxt=""):
        import math
        diff = abs(other - function)
        reference = min(abs(function), abs(other))
        if reference != 0:
            nondecimal_places = int(math.log10(reference)) + 1
            diff_scaled = diff / (10 ** nondecimal_places)
        else:
            diff_scaled = diff
        diff_rounded = round(diff_scaled, places)
        self.assertTrue(diff_rounded == 0, msg=f"{other} != {function} in {places} places {ctxt}")

    # def test_assertApprox(self):
    #     self.assertApprox(200, 200.001)
    #     self.assertApprox(200.002, 200.001)
    #     self.assertApprox(200.0049, 200.000)
    #     self.assertApprox(200.000, 199.9951)
    #     self.assertApprox(200, 200.4999, places=3)

    def assertApproxFunction(self, function, other, **kwargs):
        if len(kwargs) == 0:
            kwargs['places'] = 6

        kwargs['ctxt'] = f"in {other} != {function}"
        self.assertApprox(function.constant_coefficient, other.constant_coefficient, **kwargs)
        self.assertEqual(len(function.compound_terms), len(other.compound_terms))
        if isinstance(function, MultiParameterFunction):
            function_pairs = {tuple(p for p, _ in t.parameter_term_pairs): t for t in function.compound_terms}
            other_pairs = {tuple(p for p, _ in t.parameter_term_pairs): t for t in other.compound_terms}
            self.assertEqual(len(function_pairs), len(other_pairs))
            for p in function_pairs:
                self.assertApproxTerm(function_pairs[p], other_pairs[p], **kwargs)
        else:
            for tt, to in zip(function.compound_terms, other.compound_terms):
                self.assertApproxTerm(tt, to, **kwargs)

    def assertApproxTerm(self, tt: CompoundTerm, to: CompoundTerm, **kwargs):
        ctxt = kwargs.get('ctxt', '')
        if isinstance(tt, CompoundTerm):
            self.assertEqual(len(tt.simple_terms), len(to.simple_terms))
            for stt, sto in zip(tt.simple_terms, to.simple_terms):
                self.assertApproxSimpleTerm(stt, sto, **kwargs)
        elif isinstance(tt, MultiParameterTerm):
            self.assertEqual(len(tt.parameter_term_pairs), len(to.parameter_term_pairs))
            for stt, sto in zip(sorted(tt.parameter_term_pairs, key=itemgetter(0)),
                                sorted(to.parameter_term_pairs, key=itemgetter(0))):
                self.assertEqual(stt[0], sto[0], msg=f"Parameters are not identical {sto[0]} != {stt[0]} {ctxt}")
                self.assertApproxTerm(stt[1], sto[1], **kwargs)
        self.assertApprox(tt.coefficient, to.coefficient, **kwargs)

    def assertApproxSimpleTerm(self, stt: SimpleTerm, sto: SimpleTerm, **kwargs):
        self.assertEqual(stt.term_type, sto.term_type)
        self.assertApprox(stt.exponent, sto.exponent, **kwargs)
