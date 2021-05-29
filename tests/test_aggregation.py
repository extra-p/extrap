import numpy as np

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import Node, CallTree
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.modelers.aggregation.basic_aggregations import SumAggregation, MaxAggregation
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

        test_value = experiment1.modelers[1].models[overlap.path, metric].hypothesis.function.compound_terms

        coeff_sum = 0
        for x in correct:
            coeff_sum += x.constant_coefficient

        self.assertApprox(coeff_sum,
                          experiment1.modelers[1].models[overlap.path, metric].hypothesis.function.constant_coefficient,
                          5)

        self.assertEqual(2, len(test_value))

        # self.assertCountEqual(correct, test_value)
        # for c in correct:
        #     self.assertIn(c, test_value)

        correct_sum = 0
        for c in correct:
            correct_sum += c.evaluate(3.5)

        test_sum = experiment1.modelers[1].models[overlap.path, metric].hypothesis.function.evaluate(3.5)

        self.assertAlmostEqual(correct_sum, test_sum, 9)

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
