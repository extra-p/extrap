# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import numpy as np

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import Node, CallTree
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment, ExperimentSchema
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.functions import MultiParameterFunction, SingleParameterFunction
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.entities.terms import SimpleTerm, MultiParameterTerm, CompoundTerm
from extrap.fileio import io_helper
from extrap.modelers import aggregation
from extrap.modelers.aggregation.max_aggregation import MaxAggregation, MaxAggregationFunction
from extrap.modelers.aggregation.sum_aggregation import SumAggregationFunction, SumAggregation
from extrap.modelers.model_generator import ModelGenerator
from tests.modelling_testcase import TestCaseWithFunctionAssertions


class TestAggregation(TestCaseWithFunctionAssertions):
    def testSumDefault(self):
        metric = Metric('time')
        experiment1, callpaths = self.prepare_experiment(metric, agg__disabled=True, agg__usage_disabled=True)
        ca, cb, evt_sync, main, overlap, start, sync, wait, work = callpaths
        mg = ModelGenerator(experiment1)
        mg.model_all()
        mg.aggregate(SumAggregation())

        self.check_same(experiment1, metric, [cb.path, ca.path, evt_sync.path, sync.path, work.path, wait.path])
        self.check_changed(experiment1, metric, [overlap.path, main.path, start.path])

        correct = [experiment1.modelers[0].models[overlap.path, metric].hypothesis.function,
                   experiment1.modelers[0].models[cb.path, metric].hypothesis.function,
                   experiment1.modelers[0].models[ca.path, metric].hypothesis.function]

        test_value = experiment1.modelers[1].models[overlap.path, metric].hypothesis.function
        self.assertIsInstance(test_value, ComputationFunction)

        coeff_sum = 0
        for x in correct:
            coeff_sum += x.constant_coefficient

        self.assertApprox(coeff_sum,
                          test_value.sympy_function.as_coeff_Add()[0],
                          15)
        self.assertEqual(3, len(test_value.sympy_function.args))
        self.assertEqual(0, len(test_value.compound_terms))
        self.assertEqual(0, test_value.constant_coefficient)

        # check model
        term1 = correct[0].compound_terms[0]
        term2 = CompoundTerm(SimpleTerm('logarithm', 1))
        term2.coefficient = correct[1].compound_terms[0].coefficient + correct[2].compound_terms[0].coefficient
        correct_function = SingleParameterFunction(term1, term2)
        correct_function.constant_coefficient = coeff_sum
        correct_function = ComputationFunction(correct_function)
        self.assertEqual(correct_function,
                         experiment1.modelers[1].models[overlap.path, metric].hypothesis.function)

        correct_sum = 0
        for c in correct:
            correct_sum += c.evaluate(3.5)

        test_sum = experiment1.modelers[1].models[overlap.path, metric].hypothesis.function.evaluate(3.5)

        self.assertAlmostEqual(correct_sum, float(test_sum), 9)

        experiment2, _ = self.prepare_experiment(metric, agg__disabled__sum=True, agg__usage_disabled__sum=True)
        mg = ModelGenerator(experiment2)
        mg.model_all()
        mg.aggregate(SumAggregation())
        self.assertSetEqual(set(experiment1.modelers[1].models.keys()), set(experiment2.modelers[1].models.keys()))
        for k in experiment1.modelers[1].models:
            self.assertEqual(experiment1.modelers[1].models[k], experiment2.modelers[1].models[k], msg=str(k))

        experiment3, callpaths = self.prepare_experiment(metric, agg__disabled__max=True, agg__usage_disabled__max=True)
        ca, cb, evt_sync, main, overlap, start, sync, wait, work = callpaths
        mg = ModelGenerator(experiment3)
        mg.model_all()
        mg.aggregate(SumAggregation())

        self.check_same(experiment3, metric, [cb.path, ca.path, work.path, wait.path])
        self.check_changed(experiment3, metric, [overlap.path, evt_sync.path, sync.path, main.path, start.path])

    def testSumMultiParam(self):
        x = [2, 3]

        mp_term = MultiParameterTerm()  # 2 * p^2 * q^2
        smpl_term = SimpleTerm("polynomial", 2)
        mp_term.add_parameter_term_pair((0, smpl_term))
        mp_term.add_parameter_term_pair((1, smpl_term))
        mp_term.coefficient = 2
        fkt1 = MultiParameterFunction(mp_term)

        res1 = fkt1.evaluate(x)
        self.assertEqual(res1, 72)

        mp_term = MultiParameterTerm()  # 4 * p^2 * q^2
        smpl_term = SimpleTerm("polynomial", 2)
        mp_term.add_parameter_term_pair((0, smpl_term))
        mp_term.add_parameter_term_pair((1, smpl_term))
        mp_term.coefficient = 4
        fkt2 = MultiParameterFunction(mp_term)

        res2 = fkt2.evaluate(x)
        self.assertEqual(res2, 144)

        mp_term_mixed = MultiParameterTerm()  # 1.4 * log2(p)^3 * q^2 + 4 * p^2 * q^2
        smpl_term_log = SimpleTerm("logarithm", 3)
        mp_term_mixed.add_parameter_term_pair((0, smpl_term_log))
        mp_term_mixed.add_parameter_term_pair((1, smpl_term))
        mp_term_mixed.coefficient = 1.4
        fkt3 = MultiParameterFunction(mp_term_mixed)
        fkt3.add_compound_term(mp_term)

        res3 = fkt3.evaluate(x)
        self.assertEqual(res3, 156.6)

        s = SumAggregationFunction([fkt1, fkt2, fkt3])
        self.assertEqual(len(s.raw_terms), 3)
        self.assertIsInstance(s, ComputationFunction)
        self.assertEqual(len(s.sympy_function.args), 2)
        self.assertEqual(len(s.compound_terms), 0)

        res_sum = s.evaluate(x)
        self.assertEqual(res_sum, res1 + res2 + res3)

        fkt1.constant_coefficient = 1
        fkt2.constant_coefficient = 2
        fkt3.constant_coefficient = 3
        s = SumAggregationFunction([fkt1, fkt2, fkt3])
        self.assertEqual(len(s.raw_terms), 3)
        self.assertIsInstance(s, ComputationFunction)
        self.assertEqual(len(s.sympy_function.args), 3)
        self.assertEqual(len(s.compound_terms), 0)
        self.assertEqual(s.constant_coefficient, 0)
        self.assertEqual(s.sympy_function.as_coeff_Add()[0], 6)

    def testMaxMultiParam(self):
        small_x = [2, 3]
        large_x = [5, 6]

        mp_term = MultiParameterTerm()  # 2 * p^2 * q^2 +8
        smpl_term = SimpleTerm("polynomial", 2)
        mp_term.add_parameter_term_pair((0, smpl_term))
        mp_term.add_parameter_term_pair((1, smpl_term))
        mp_term.coefficient = 2
        fkt1 = MultiParameterFunction(mp_term)
        fkt1.constant_coefficient = 113

        small_res1 = fkt1.evaluate(small_x)
        self.assertEqual(small_res1, 185)
        large_res1 = fkt1.evaluate(large_x)
        self.assertEqual(large_res1, 1913)

        mp_term = MultiParameterTerm()  # 4 * p^2 * q^2
        smpl_term = SimpleTerm("polynomial", 2)
        mp_term.add_parameter_term_pair((0, smpl_term))
        mp_term.add_parameter_term_pair((1, smpl_term))
        mp_term.coefficient = 4
        fkt2 = MultiParameterFunction(mp_term)

        small_res2 = fkt2.evaluate(small_x)
        self.assertEqual(small_res2, 144)
        large_res2 = fkt2.evaluate(large_x)
        self.assertEqual(large_res2, 3600)

        s = MaxAggregationFunction([fkt1, fkt2])
        self.assertEqual(len(s.raw_terms), 2)
        self.assertIsInstance(s, ComputationFunction)
        self.assertEqual(len(s.sympy_function.args), 2)
        self.assertEqual(len(s.compound_terms), 0)

        res_sum = s.evaluate(small_x)
        self.assertEqual(res_sum, 185)
        res_sum = s.evaluate(large_x)
        self.assertEqual(res_sum, 3600)

    def testSum2(self):
        metric = Metric('time')
        experiment1 = Experiment()
        experiment1.parameters = [Parameter('n')]
        experiment1.metrics = [metric]
        experiment1.coordinates = [Coordinate(c) for c in range(1, 6)]
        neB = Node("neB", Callpath("main->emptyA->emptyB->neA->neB"), [])
        neA = Node("neA", Callpath("main->emptyA->emptyB->neA"), [neB])
        emptyB = Node("emptyB", Callpath.EMPTY, [neA])
        emptyA = Node("emptyA", Callpath.EMPTY, [emptyB])
        main = Node("main", Callpath("main"), [emptyA])
        experiment1.callpaths = [main.path, neA.path, neB.path]
        experiment1.call_tree = io_helper.create_call_tree(experiment1.callpaths)
        experiment1.measurements = {
            (main.path, metric): [Measurement(Coordinate(c), None, None, 3 * c) for c in range(1, 6)],
            (neB.path, metric): [Measurement(Coordinate(c), None, None, 2 * c) for c in range(1, 6)],
            (neA.path, metric): [Measurement(Coordinate(c), None, None, 1 * c) for c in range(1, 6)],
        }

        mg = ModelGenerator(experiment1)
        mg.model_all()
        mg.aggregate(SumAggregation())

        self.check_same(experiment1, metric, [neB.path])
        self.check_changed(experiment1, metric,
                           [main.path, Callpath("main->emptyA"), Callpath("main->emptyA->emptyB"), neA.path])

        correct = [experiment1.modelers[0].models[main.path, metric].hypothesis.function,
                   experiment1.modelers[0].models[neB.path, metric].hypothesis.function,
                   experiment1.modelers[0].models[neA.path, metric].hypothesis.function]

        test_value = experiment1.modelers[1].models[main.path, metric].hypothesis.function
        self.assertIsInstance(test_value, ComputationFunction)

        coeff_sum = 0
        for x in correct:
            coeff_sum += x.constant_coefficient

        self.assertApprox(coeff_sum,
                          test_value.sympy_function.as_coeff_Add()[0],
                          15)
        self.assertEqual(2, len(test_value.sympy_function.args))
        self.assertEqual(0, len(test_value.compound_terms))
        self.assertEqual(0, test_value.constant_coefficient)

    def testSumCategory(self):
        metric = Metric('time')
        experiment1 = Experiment()
        experiment1.parameters = [Parameter('n')]
        experiment1.metrics = [metric]
        experiment1.coordinates = [Coordinate(c) for c in range(1, 6)]
        categoryB = Node("cat", Callpath("main->emptyA->emptyB->neA->neB->cat", agg__category='cat'), [])
        neB = Node("neB", Callpath("main->emptyA->emptyB->neA->neB"), [categoryB])
        neA = Node("neA", Callpath("main->emptyA->emptyB->neA"), [neB])
        emptyB = Node("emptyB", Callpath("main->emptyA->emptyB"), [neA])
        emptyA = Node("emptyA", Callpath("main->emptyA"), [emptyB])
        categoryC = Node("cat", Callpath("main->neC->cat", agg__category='cat'), [])
        neC = Node("neC", Callpath("main->neC"), [categoryC])
        main = Node("main", Callpath("main"), [emptyA, neC])
        experiment1.callpaths = [main.path, neC.path, emptyA.path, emptyB.path, neA.path, neB.path, categoryC.path,
                                 categoryB.path]
        experiment1.call_tree = io_helper.create_call_tree(experiment1.callpaths)
        experiment1.measurements = {
            (neB.path, metric): [Measurement(Coordinate(c), None, None, 2 * c) for c in range(1, 6)],
            (neA.path, metric): [Measurement(Coordinate(c), None, None, 1 * c) for c in range(1, 6)],
            (neC.path, metric): [Measurement(Coordinate(c), None, None, 3 * c) for c in range(1, 6)],

            (categoryC.path, metric): [Measurement(Coordinate(c), None, None, 3 * c ** (3 / 2)) for c in range(1, 6)],
            (categoryB.path, metric): [Measurement(Coordinate(c), None, None, 3 * c ** (3 / 2)) for c in range(1, 6)],
        }

        mg = ModelGenerator(experiment1)
        mg.model_all()
        mg.aggregate(SumAggregation())

        self.check_same(experiment1, metric, [neB.path, neC.path, categoryB.path, categoryC.path])
        self.check_changed(experiment1, metric, [main.path, emptyA.path, emptyB.path, neA.path])

        correct = [experiment1.modelers[0].models[neB.path, metric].hypothesis.function,
                   experiment1.modelers[0].models[neA.path, metric].hypothesis.function,
                   experiment1.modelers[0].models[neC.path, metric].hypothesis.function]

        test_value = experiment1.modelers[1].models[main.path, metric].hypothesis.function
        self.assertIsInstance(test_value, ComputationFunction)

        coeff_sum = 0
        for x in correct:
            coeff_sum += x.constant_coefficient

        self.assertApprox(coeff_sum,
                          test_value.sympy_function.as_coeff_Add()[0],
                          15)
        self.assertEqual(2, len(test_value.sympy_function.args))
        self.assertEqual(0, len(test_value.compound_terms))
        self.assertEqual(0, test_value.constant_coefficient)

        correct = [experiment1.modelers[0].models[neB.path, metric].hypothesis.function,
                   experiment1.modelers[0].models[neA.path, metric].hypothesis.function]

        test_value = experiment1.modelers[1].models[emptyA.path, metric].hypothesis.function
        self.assertIsInstance(test_value, ComputationFunction)

        coeff_sum = 0
        for x in correct:
            coeff_sum += x.constant_coefficient

        self.assertApprox(coeff_sum,
                          test_value.sympy_function.as_coeff_Add()[0],
                          15)
        self.assertEqual(2, len(test_value.sympy_function.args))
        self.assertEqual(0, len(test_value.compound_terms))
        self.assertEqual(0, test_value.constant_coefficient)

        self.assertIn((main.path.concat('cat'), metric), experiment1.modelers[1].models)
        self.assertIn((Callpath('cat'), metric), experiment1.modelers[1].models)

    def testSum3(self):
        metric = Metric('time')
        experiment1 = Experiment()
        experiment1.parameters = [Parameter('n')]
        experiment1.metrics = [metric]
        experiment1.coordinates = [Coordinate(c) for c in range(1, 6)]
        neB = Node("neB", Callpath("main->emptyA->emptyB->neA->neB"), [])
        neA = Node("neA", Callpath("main->emptyA->emptyB->neA"), [neB])
        emptyB = Node("emptyB", Callpath("main->emptyA->emptyB"), [neA])
        emptyA = Node("emptyA", Callpath("main->emptyA"), [emptyB])
        neC = Node("neC", Callpath("main->neC"), [])
        main = Node("main", Callpath("main"), [emptyA, neC])
        experiment1.callpaths = [main.path, neC.path, emptyA.path, emptyB.path, neA.path, neB.path]
        experiment1.call_tree = io_helper.create_call_tree(experiment1.callpaths)
        experiment1.measurements = {
            (neB.path, metric): [Measurement(Coordinate(c), None, None, 2 * c) for c in range(1, 6)],
            (neA.path, metric): [Measurement(Coordinate(c), None, None, 1 * c) for c in range(1, 6)],
            (neC.path, metric): [Measurement(Coordinate(c), None, None, 3 * c) for c in range(1, 6)],
        }

        mg = ModelGenerator(experiment1)
        mg.model_all()
        mg.aggregate(SumAggregation())

        self.check_same(experiment1, metric, [neB.path, neC.path])
        self.check_changed(experiment1, metric, [main.path, emptyA.path, emptyB.path, neA.path])

        correct = [experiment1.modelers[0].models[neB.path, metric].hypothesis.function,
                   experiment1.modelers[0].models[neA.path, metric].hypothesis.function,
                   experiment1.modelers[0].models[neC.path, metric].hypothesis.function]

        test_value = experiment1.modelers[1].models[main.path, metric].hypothesis.function
        self.assertIsInstance(test_value, ComputationFunction)

        coeff_sum = 0
        for x in correct:
            coeff_sum += x.constant_coefficient

        self.assertApprox(coeff_sum,
                          test_value.sympy_function.as_coeff_Add()[0],
                          15)
        self.assertEqual(2, len(test_value.sympy_function.args))
        self.assertEqual(0, len(test_value.compound_terms))
        self.assertEqual(0, test_value.constant_coefficient)

        correct = [experiment1.modelers[0].models[neB.path, metric].hypothesis.function,
                   experiment1.modelers[0].models[neA.path, metric].hypothesis.function]

        test_value = experiment1.modelers[1].models[emptyA.path, metric].hypothesis.function
        self.assertIsInstance(test_value, ComputationFunction)

        coeff_sum = 0
        for x in correct:
            coeff_sum += x.constant_coefficient

        self.assertApprox(coeff_sum,
                          test_value.sympy_function.as_coeff_Add()[0],
                          15)

        self.assertEqual(2, len(test_value.sympy_function.args))
        self.assertEqual(0, len(test_value.compound_terms))
        self.assertEqual(0, test_value.constant_coefficient)

    def check_changed(self, experiment1, metric, paths):
        for cp in paths:
            model = experiment1.modelers[0].models.get((cp, metric))
            model1 = experiment1.modelers[1].models[cp, metric]
            if model is None:
                self.assertIsNotNone(model1)
                continue
            self.assertNotEqual(type(model.hypothesis.function), type(model1.hypothesis.function), msg=str(cp))
            self.assertRaises(AssertionError, self.assertApproxFunction, model.hypothesis.function,
                              model1.hypothesis.function, msg=str(cp))

    def check_same(self, experiment1, metric, paths):
        for cp in paths:
            model = experiment1.modelers[0].models[cp, metric]
            model1 = experiment1.modelers[1].models[cp, metric]
            self.assertEqual(type(model.hypothesis.function), type(model1.hypothesis.function), msg=str(cp))
            self.assertApproxFunction(model.hypothesis.function, model1.hypothesis.function, msg=str(cp))

    def testMax(self):
        metric = Metric('time')
        experiment1, callpaths = self.prepare_experiment(metric, agg__disabled__max=True)
        ca, cb, evt_sync, main, overlap, start, sync, wait, work = callpaths
        mg = ModelGenerator(experiment1)
        mg.model_all()
        mg.aggregate(MaxAggregation())

        self.check_same(experiment1, metric, [cb.path, ca.path, evt_sync.path, work.path, wait.path])
        self.check_changed(experiment1, metric, [overlap.path, sync.path, main.path, start.path])

    # def test_tag_suffix_update(self):
    #     agg = SumAggregation()
    #     self.assertEqual(Aggregation.TAG_CATEGORY, agg.TAG_CATEGORY)
    #     self.assertEqual(SumAggregation.TAG_CATEGORY, agg.TAG_CATEGORY)
    #     self.assertNotEqual(Aggregation.TAG_DISABLED, agg.TAG_DISABLED)
    #     self.assertEqual(SumAggregation.TAG_DISABLED, agg.TAG_DISABLED)
    #     self.assertNotEqual(Aggregation.TAG_USAGE_DISABLED, agg.TAG_USAGE_DISABLED)
    #     self.assertEqual(SumAggregation.TAG_USAGE_DISABLED, agg.TAG_USAGE_DISABLED)
    #
    #     expected_class_tag_category = Aggregation.TAG_CATEGORY
    #     expected_class_tag_disabled = SumAggregation.TAG_DISABLED
    #     expected_class_tag_usage_disabled = SumAggregation.TAG_USAGE_DISABLED
    #     expected_class_tag_usage_disabled_agg_model = Aggregation.TAG_USAGE_DISABLED_agg_model
    #     agg.tag_suffix = "test"
    #     self.assertEqual(expected_class_tag_category, Aggregation.TAG_CATEGORY)
    #     self.assertNotEqual(expected_class_tag_category, agg.TAG_CATEGORY)
    #     self.assertEqual(expected_class_tag_category + "__test", agg.TAG_CATEGORY)
    #
    #     self.assertEqual(expected_class_tag_disabled, SumAggregation.TAG_DISABLED)
    #     self.assertNotEqual(expected_class_tag_disabled, agg.TAG_DISABLED)
    #     self.assertEqual(expected_class_tag_disabled + "__test", agg.TAG_DISABLED)
    #
    #     self.assertEqual(expected_class_tag_usage_disabled, SumAggregation.TAG_USAGE_DISABLED)
    #     self.assertNotEqual(expected_class_tag_usage_disabled, agg.TAG_USAGE_DISABLED)
    #     self.assertEqual(expected_class_tag_usage_disabled + "__test", agg.TAG_USAGE_DISABLED)
    #
    #     self.assertEqual(expected_class_tag_usage_disabled_agg_model, Aggregation.TAG_USAGE_DISABLED_agg_model)
    #     self.assertEqual(expected_class_tag_usage_disabled_agg_model, SumAggregation.TAG_USAGE_DISABLED_agg_model)
    #     self.assertEqual(expected_class_tag_usage_disabled_agg_model, agg.TAG_USAGE_DISABLED_agg_model)
    #
    #     agg.tag_suffix = "test2"
    #     self.assertEqual(expected_class_tag_category + "__test2", agg.TAG_CATEGORY)
    #     self.assertEqual(expected_class_tag_disabled + "__test2", agg.TAG_DISABLED)
    #     self.assertEqual(expected_class_tag_usage_disabled + "__test2", agg.TAG_USAGE_DISABLED)
    #
    #     agg.tag_suffix = ""
    #     self.assertEqual(expected_class_tag_category, agg.TAG_CATEGORY)
    #     self.assertEqual(expected_class_tag_disabled, agg.TAG_DISABLED)
    #     self.assertEqual(expected_class_tag_usage_disabled, agg.TAG_USAGE_DISABLED)
    #
    #     agg.tag_suffix = "test"
    #     agg.tag_suffix = None
    #     self.assertEqual(expected_class_tag_category, agg.TAG_CATEGORY)
    #     self.assertEqual(expected_class_tag_disabled, agg.TAG_DISABLED)
    #     self.assertEqual(expected_class_tag_usage_disabled, agg.TAG_USAGE_DISABLED)

    @staticmethod
    def prepare_experiment(metric, **evt_sync_tags):
        experiment1 = Experiment()
        experiment1.parameters = [Parameter('n')]
        experiment1.metrics = [metric]
        experiment1.coordinates = [Coordinate(c) for c in range(1, 6)]
        cb = Node("cb", Callpath("_start->main->sync->EVT_SYNC->OVERLAP->cb"), [])
        ca = Node("ca", Callpath("_start->main->sync->EVT_SYNC->OVERLAP->ca"), [])
        overlap = Node("OVERLAP", Callpath("_start->main->sync->EVT_SYNC->OVERLAP"), [ca, cb])
        wait = Node("WAIT", Callpath("_start->main->sync->EVT_SYNC->WAIT"), [])
        evt_sync = Node("EVT_SYNC", Callpath("_start->main->sync->EVT_SYNC", **evt_sync_tags), [wait, overlap])
        sync = Node("sync", Callpath("_start->main->sync"), [evt_sync])
        work = Node("work", Callpath("_start->main->work"))
        main = Node("main", Callpath("_start->main"), [sync, work])
        start = Node("_start", Callpath("_start"), [main])
        root = CallTree()
        root.add_child_node(start)
        experiment1.callpaths = [cb.path, ca.path, evt_sync.path, work.path, wait.path, overlap.path, sync.path,
                                 main.path, start.path]
        experiment1.call_tree = io_helper.create_call_tree(experiment1.callpaths)
        experiment1.measurements = {
            (evt_sync.path, metric): [Measurement(Coordinate(c), evt_sync.path, metric, 1 * c ** 2) for c in
                                      range(1, 6)],
            (wait.path, metric): [Measurement(Coordinate(c), wait.path, metric, 2 * c) for c in range(1, 6)],
            (overlap.path, metric): [Measurement(Coordinate(c), overlap.path, metric, 7 * c) for c in range(1, 6)],
            (cb.path, metric): [Measurement(Coordinate(c), cb.path, metric, 10 * np.log2(c)) for c in range(1, 6)],
            (ca.path, metric): [Measurement(Coordinate(c), ca.path, metric, 20 * np.log2(c)) for c in range(1, 6)],
            (sync.path, metric): [Measurement(Coordinate(c), sync.path, metric, 5 * c) for c in range(1, 6)],
            (work.path, metric): [Measurement(Coordinate(c), work.path, metric, 3 * c ** 2) for c in range(1, 6)],
            (main.path, metric): [Measurement(Coordinate(c), main.path, metric, 2 * c) for c in range(1, 6)],
            (start.path, metric): [Measurement(Coordinate(c), start.path, metric, 1 * c) for c in range(1, 6)],
        }
        return experiment1, (ca, cb, evt_sync, main, overlap, start, sync, wait, work)

    def test_serialization(self):
        metric = Metric('time')
        experiment1, _ = self.prepare_experiment(metric, agg__disabled=True, agg__usage_disabled=True)
        mg = ModelGenerator(experiment1)
        mg.model_all()
        for aggregation_cls in aggregation.all_aggregations.values():
            mg.aggregate(aggregation_cls())

        schema = ExperimentSchema()
        ser_exp = schema.dump(experiment1)

        reconstructed = schema.load(ser_exp)
        self.assertListEqual(experiment1.parameters, reconstructed.parameters)
        self.assertListEqual(experiment1.coordinates, reconstructed.coordinates)
        self.assertListEqual(experiment1.metrics, reconstructed.metrics)
        self.assertSetEqual(set(experiment1.callpaths), set(reconstructed.callpaths))
        self.assertDictEqual(experiment1.measurements, reconstructed.measurements)
        for modeler, modeler_reconstructed in zip(experiment1.modelers, reconstructed.modelers):
            self.assertEqual(modeler, modeler_reconstructed,
                             f"Model Generator {modeler.name} was not reconstructed correctly .")
        self.assertListEqual(experiment1.modelers, reconstructed.modelers)
