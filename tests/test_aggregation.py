import numpy as np

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import Node, CallTree
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.functions import SingleParameterFunction, MultiParameterFunction
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.entities.terms import CompoundTerm, SimpleTerm, MultiParameterTerm, SingleParameterTerm
from extrap.fileio import io_helper
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.modelers.aggregation.basic_aggregations import MaxAggregation, SumAggregation
from extrap.modelers.aggregation.sum_aggregation import SumAggregationFunction
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler
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

        test_value = experiment1.modelers[1].models[overlap.path, metric].hypothesis.function.compound_terms

        coeff_sum = 0
        for x in correct:
            coeff_sum += x.constant_coefficient

        self.assertApprox(coeff_sum,
                          experiment1.modelers[1].models[overlap.path, metric].hypothesis.function.constant_coefficient,
                          5)

        self.assertEqual(2, len(test_value))

        # check model
        term1 = correct[0].compound_terms[0]
        term2 = CompoundTerm(SimpleTerm('logarithm', 1))
        term2.coefficient = correct[1].compound_terms[0].coefficient + correct[2].compound_terms[0].coefficient
        correct_function = SingleParameterFunction(term1, term2)
        correct_function.constant_coefficient = coeff_sum
        self.assertApproxFunction(correct_function,
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
        mp_term = MultiParameterTerm()  # 2 * p^2 * q^2
        smpl_term = SimpleTerm("polynomial", 2)
        mp_term.add_parameter_term_pair((0, smpl_term))
        mp_term.add_parameter_term_pair((1, smpl_term))
        mp_term.coefficient = 2
        fkt1 = MultiParameterFunction(mp_term)

        res1 = fkt1.evaluate([2, 3])
        self.assertEqual(res1, 72)

        mp_term = MultiParameterTerm()  # 4 * p^2 * q^2
        smpl_term = SimpleTerm("polynomial", 2)
        mp_term.add_parameter_term_pair((0, smpl_term))
        mp_term.add_parameter_term_pair((1, smpl_term))
        mp_term.coefficient = 4
        fkt2 = MultiParameterFunction(mp_term)

        res2 = fkt2.evaluate([2, 3])
        self.assertEqual(res2, 144)

        mp_term_mixed = MultiParameterTerm()  # 1.4 * log2(p)^3 * q^2 + 4 * p^2 * q^2
        smpl_term_log = SimpleTerm("logarithm", 3)
        mp_term_mixed.add_parameter_term_pair((0, smpl_term_log))
        mp_term_mixed.add_parameter_term_pair((1, smpl_term))
        mp_term_mixed.coefficient = 1.4
        fkt3 = MultiParameterFunction(mp_term_mixed)
        fkt3.add_compound_term(mp_term)

        res3 = fkt3.evaluate([2, 3])
        self.assertEqual(res3, 156.6)

        s = SumAggregationFunction([fkt1, fkt2, fkt3])
        self.assertEqual(len(s.raw_terms), 3)
        self.assertEqual(len(s.compound_terms), 2)

        res_sum = s.evaluate([2, 3])
        self.assertEqual(res_sum, res1 + res2 + res3)

    def check_changed(self, experiment1, metric, paths):
        for cp in paths:
            model = experiment1.modelers[0].models[cp, metric]
            model1 = experiment1.modelers[1].models[cp, metric]
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
            (evt_sync.path, metric): [Measurement(Coordinate(c), None, None, 1 * c ** 2) for c in range(1, 6)],
            (wait.path, metric): [Measurement(Coordinate(c), None, None, 2 * c) for c in range(1, 6)],
            (overlap.path, metric): [Measurement(Coordinate(c), None, None, 7 * c) for c in range(1, 6)],
            (cb.path, metric): [Measurement(Coordinate(c), None, None, 10 * np.log2(c)) for c in range(1, 6)],
            (ca.path, metric): [Measurement(Coordinate(c), None, None, 20 * np.log2(c)) for c in range(1, 6)],
            (sync.path, metric): [Measurement(Coordinate(c), None, None, 5 * c) for c in range(1, 6)],
            (work.path, metric): [Measurement(Coordinate(c), None, None, 3 * c ** 2) for c in range(1, 6)],
            (main.path, metric): [Measurement(Coordinate(c), None, None, 2 * c) for c in range(1, 6)],
            (start.path, metric): [Measurement(Coordinate(c), None, None, 1 * c) for c in range(1, 6)],
        }
        return experiment1, (ca, cb, evt_sync, main, overlap, start, sync, wait, work)
