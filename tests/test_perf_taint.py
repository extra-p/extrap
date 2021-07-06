# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest
from typing import cast

from extrap.entities.parameter import Parameter
from extrap.entities.terms import MultiParameterTerm
from extrap.fileio.file_reader.perf_taint_reader import PerfTaintReader
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.exceptions import FileFormatError
from tests.modelling_testcase import TestCaseWithFunctionAssertions


class PerfTaintTest(TestCaseWithFunctionAssertions):
    all_params_check = [
        'lulesh2.0',
        'lulesh2.0->int main(int, char**)',
        'lulesh2.0->int main(int, char**)->void CommSyncPosVel(Domain&)',
        'lulesh2.0->int main(int, char**)->void CommSend(Domain&, int, Index_t, Real_t& (Domain::**)(Index_t), Index_t, Index_t, Index_t, bool, bool)'
    ]
    size_params_check = [
        'lulesh2.0->int main(int, char**)->void CalcAccelerationForNodes(Domain&, Index_t)',
        'lulesh2.0->int main(int, char**)->void CalcTimeConstraintsForElems(Domain&)->void CalcHydroConstraintForElems(Domain&, Index_t, Index_t*, Real_t, Real_t&)'
    ]
    p_params_check = [
        'lulesh2.0->int main(int, char**)->void TimeIncrement(Domain&)',
        'lulesh2.0->int main(int, char**)->void TimeIncrement(Domain&)->MPI_Allreduce',
        'lulesh2.0->int main(int, char**)->void CommSend(Domain&, int, Index_t, Real_t& (Domain::**)(Index_t), Index_t, Index_t, Index_t, bool, bool)->MPI_Waitall'
    ]
    no_params_check = [
        'lulesh2.0->int main(int, char**)->MPI_Init',
        'lulesh2.0->int main(int, char**)->MPI_Finalize'
    ]

    @classmethod
    def setUpClass(cls) -> None:
        reader = PerfTaintReader()
        reader.scaling_type = 'weak'
        cls.experiment = reader.read_experiment('data/perf_taint/lulesh/lulesh.ll.json')

    def test_loading_fails(self):
        reader = PerfTaintReader()
        reader.scaling_type = 'weak'
        self.assertRaises(FileFormatError, reader.read_experiment, 'data/perf_taint/lulesh')

    def test_loading(self):
        experiment = self.experiment
        self.assertListEqual([Parameter('p'), Parameter('size')], experiment.parameters)
        all_params_counter, size_params_counter, p_params_counter, no_params_counter = 0, 0, 0, 0
        for callpath in experiment.callpaths:
            self.assertIn('perf_taint__depends_on_params', callpath.tags)
            depends_on_params = callpath.tags['perf_taint__depends_on_params']
            self.assertLessEqual(len(depends_on_params), len(experiment.parameters))
            if depends_on_params:
                self.assertLess(max(depends_on_params), len(experiment.parameters))
                self.assertGreaterEqual(min(depends_on_params), 0)
                self.assertListEqual(sorted(set(depends_on_params)), depends_on_params)
            if callpath.name in self.all_params_check:
                self.assertListEqual([0, 1], depends_on_params, callpath)
                all_params_counter += 1
            elif callpath.name in self.size_params_check:
                self.assertListEqual([1], depends_on_params, callpath)
                size_params_counter += 1
            elif callpath.name in self.p_params_check:
                self.assertListEqual([0], depends_on_params, callpath)
                p_params_counter += 1
            elif callpath.name in self.no_params_check:
                self.assertListEqual([], depends_on_params)
                no_params_counter += 1
        self.assertEqual(len(self.all_params_check), all_params_counter, 'Not all checks for "all_params" were done')
        self.assertEqual(len(self.size_params_check), size_params_counter, 'Not all checks for "size_params" were done')
        self.assertEqual(len(self.p_params_check), p_params_counter, 'Not all checks for "p_params" were done')
        self.assertEqual(len(self.no_params_check), no_params_counter, 'Not all checks for "no_params" were done')

    def test_model(self):
        experiment = self.experiment
        ModelGenerator(experiment).model_all()

        models = experiment.modelers[0].models

        for (callpath, _), model in models.items():
            self.assertIn('perf_taint__depends_on_params', callpath.tags)
            depends_on_params = callpath.tags['perf_taint__depends_on_params']
            used_params = [p for t in model.hypothesis.function.compound_terms
                           for p, _ in cast(MultiParameterTerm, t).parameter_term_pairs]
            for p in used_params:
                self.assertIn(p, depends_on_params)


if __name__ == '__main__':
    unittest.main()
