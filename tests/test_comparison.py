# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
from unittest import TestCase

import numpy as np

from extrap.comparison.experiment_comparison import ComparisonExperiment, ComparisonModel, ComparisonFunction
from extrap.comparison.matchers import all_matchers
from extrap.comparison.matchers.minimum_matcher import MinimumMatcher
from extrap.comparison.matchers.smart_matcher import SmartMatcher
from extrap.entities.callpath import Callpath
from extrap.entities.calltree import Node, CallTree
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment, ExperimentSchema
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.fileio.experiment_io import read_experiment
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.modelers.aggregation.sum_aggregation import SumAggregation
from extrap.modelers.model_generator import ModelGenerator


class TestComparison(TestCase):
    def test_identity_comparison(self):
        experiment1 = TextFileReader().read_experiment('data/text/one_parameter_6.txt')
        ModelGenerator(experiment1).model_all()
        experiment2 = TextFileReader().read_experiment('data/text/one_parameter_6.txt')
        ModelGenerator(experiment2).model_all()
        experiment = ComparisonExperiment(experiment1, experiment2, MinimumMatcher())
        experiment.do_comparison()
        model = next(iter(experiment.modelers[0].models.values()))
        model.hypothesis.function.evaluate(np.array([12, 22, 32, 42]))

    def test_smart_comparison_basic_output(self):
        experiment1 = read_experiment('data/comparison/lulesh_with_tags.extra-p')
        experiment2 = read_experiment('data/comparison/lulesh-cpu_demangled.extra-p')
        experiment = ComparisonExperiment(experiment1, experiment2, SmartMatcher())
        experiment.do_comparison()
        self.check_comparison_against_source(experiment, experiment1)
        self.check_comparison_against_source(experiment, experiment2)

    def test_differing_coordinate_sets(self):
        experiment1 = TextFileReader().read_experiment('data/text/one_parameter_1.txt')
        experiment2 = TextFileReader().read_experiment('data/comparison/one_parameter_1_other_coordinates.txt')
        experiment = ComparisonExperiment(experiment1, experiment2, SmartMatcher())
        experiment.do_comparison()
        # self.check_comparison_against_source(experiment, experiment1)
        # self.check_comparison_against_source(experiment, experiment2)

    def test_minimal_comparison_basic_output(self):
        experiment1 = TextFileReader().read_experiment('data/text/two_parameter_5.txt')
        ModelGenerator(experiment1).model_all()
        experiment2 = TextFileReader().read_experiment('data/text/two_parameter_6.txt')
        ModelGenerator(experiment2).model_all()
        experiment = ComparisonExperiment(experiment1, experiment2, SmartMatcher())
        experiment.do_comparison()
        self.check_comparison_against_source(experiment, experiment1)
        self.check_comparison_against_source(experiment, experiment2)
        self.assertNotEqual([], experiment.modelers)
        self.assertNotEqual([], experiment.modelers[0].models)

    def test_smart_comparison_add_subtree_and_merge_measurements(self):
        ct_parent = Node('[exp1] main', Callpath('->[Comparison]->[exp1] main', agg__disabled=True))

        cB = Node("cB", Callpath("_start->main->sync->EVT_SYNC->OVERLAP->cB", gpu__overlap=True, gpu__kernel=True), [])
        cA = Node("cA", Callpath("_start->main->sync->EVT_SYNC->OVERLAP->cA", gpu__overlap=True, gpu__kernel=True), [])
        overlap = Node("OVERLAP", Callpath("_start->main->sync->EVT_SYNC->OVERLAP", agg__usage__disabled=True,
                                           gpu__overlap='agg'), [cA, cB])
        wait = Node("WAIT",
                    Callpath("_start->main->sync->EVT_SYNC->WAIT", agg__usage__disabled=True, agg__disabled=True), [])
        evt_sync = Node("EVT_SYNC", Callpath("_start->main->sync->EVT_SYNC"), [wait, overlap])
        source_nodes = Node("sync", Callpath("_start->main->sync"), [evt_sync])

        measurement_out = {Coordinate(c): Measurement(Coordinate(c), None, None, v) for c, v in
                           zip([64, 512, 4096, 32768, 262144], range(5, 10, 1))}
        new_matches = {}
        measurements = {}
        s_measurements = {}
        metric = Metric('time')
        # TestMatcher()._add_subtree_and_merge_measurements(ct_parent, source_nodes, s_measurements, 0, 2, metric,
        #                                                  measurement_out, new_matches, measurements)
        r_cB = Node("cB", Callpath("->[Comparison]->[exp1] main->sync->EVT_SYNC->OVERLAP->cB", gpu__overlap=True,
                                   gpu__kernel=True), [])
        r_cA = Node("cA", Callpath("->[Comparison]->[exp1] main->sync->EVT_SYNC->OVERLAP->cA", gpu__overlap=True,
                                   gpu__kernel=True), [])
        r_overlap = Node("OVERLAP",
                         Callpath("->[Comparison]->[exp1] main->sync->EVT_SYNC->OVERLAP", agg__usage__disabled=True,
                                  gpu__overlap='agg'), [r_cA, r_cB])
        r_wait = Node("WAIT",
                      Callpath("->[Comparison]->[exp1] main->sync->EVT_SYNC->WAIT", agg__usage__disabled=True,
                               agg__disabled=True), [])
        r_evt_sync = Node("EVT_SYNC", Callpath("->[Comparison]->[exp1] main->sync->EVT_SYNC"), [r_wait, r_overlap])
        r_sync = Node("sync", Callpath("->[Comparison]->[exp1] main->sync"), [r_evt_sync])
        expected_parent = Node('[exp1] main', Callpath('->[Comparison]->[exp1] main', agg__disabled=True),
                               [r_sync])
        expected_measurement_out = {Coordinate(c): Measurement(Coordinate(c), None, None, v) for c, v in
                                    zip([64, 512, 4096, 32768, 262144], range(5, 10, 1))}
        # self.assertTrue(expected_parent.exactly_equal(ct_parent))
        # self.assertDictEqual(expected_measurement_out, measurement_out)
        # self.assertDictEqual({}, measurements)
        # self.assertDictEqual({}, s_measurements)
        # self.assertDictEqual({r_cB: [cB, None], r_cA: [cA, None], r_overlap: [overlap, None], r_wait: [wait, None],
        #                       r_evt_sync: [evt_sync, None], r_sync: [source_nodes, None]}, new_matches)

        new_matches = {}
        measurements = {}
        s_measurements = {(evt_sync.path, metric): [Measurement(Coordinate(c), None, None, v) for c, v in
                                                    zip([64, 512, 4096, 32768, 262144], range(20, 101, 20))],
                          (wait.path, metric): [Measurement(Coordinate(c), None, None, v) for c, v in
                                                zip([64, 512, 4096, 32768, 262144], range(10, 51, 10))],
                          (overlap.path, metric): [Measurement(Coordinate(c), None, None, v) for c, v in
                                                   zip([64, 512, 4096, 32768, 262144], range(10, 51, 10))],
                          (cB.path, metric): [Measurement(Coordinate(c), None, None, v) for c, v in
                                              zip([64, 512, 4096, 32768, 262144], range(10, 51, 10))]
                          }
        expected_s_measurements = copy.deepcopy(s_measurements)
        SmartMatcher()._add_subtree_and_merge_measurements(ct_parent, source_nodes, s_measurements, 0, 2, metric,
                                                           measurement_out, new_matches, measurements)
        expected_measurement_out = {Coordinate(c): Measurement(Coordinate(c), None, None, v) for c, v in
                                    zip([64, 512, 4096, 32768, 262144], [25, 46, 67, 88, 109])}

        self.assertTrue(expected_parent.exactly_equal(ct_parent))
        self.assertDictEqual(expected_measurement_out, measurement_out)
        self.assertDictEqual({
            (r_evt_sync.path, metric): [Measurement(Coordinate(c), None, None, v) for c, v in
                                        zip([64, 512, 4096, 32768, 262144], range(20, 101, 20))],
            (r_wait.path, metric): [Measurement(Coordinate(c), None, None, v) for c, v in
                                    zip([64, 512, 4096, 32768, 262144], range(10, 51, 10))],
            (r_overlap.path, metric): [Measurement(Coordinate(c), None, None, v) for c, v in
                                       zip([64, 512, 4096, 32768, 262144], range(10, 51, 10))],
            (r_cB.path, metric): [Measurement(Coordinate(c), None, None, v) for c, v in
                                  zip([64, 512, 4096, 32768, 262144], range(10, 51, 10))]
        }, measurements)
        self.assertDictEqual(expected_s_measurements, s_measurements)
        self.assertDictEqual({r_cB: [cB, None], r_cA: [cA, None], r_overlap: [overlap, None], r_wait: [wait, None],
                              r_evt_sync: [evt_sync, None], r_sync: [source_nodes, None]}, new_matches)

    def test_smart_comparison2(self):
        metric = Metric('time')
        experiment1 = Experiment()
        experiment1.parameters = [Parameter('n')]
        experiment1.metrics = [metric]
        experiment1.coordinates = [Coordinate(c) for c in range(1, 6)]
        cB = Node("cB", Callpath("_start->main->sync->EVT_SYNC->OVERLAP->cB"), [])
        cA = Node("cA", Callpath("_start->main->sync->EVT_SYNC->OVERLAP->cA"), [])
        overlap = Node("OVERLAP", Callpath("_start->main->sync->EVT_SYNC->OVERLAP"), [cA, cB])
        wait = Node("WAIT", Callpath("_start->main->sync->EVT_SYNC->WAIT"), [])
        evt_sync = Node("EVT_SYNC", Callpath("_start->main->sync->EVT_SYNC"), [wait, overlap])
        sync = Node("sync", Callpath("_start->main->sync"), [evt_sync])
        main = Node("main", Callpath("_start->main"), [sync])
        start = Node("_start", Callpath("_start"), [main])
        root = CallTree()
        root.add_child_node(start)
        experiment1.callpaths = [evt_sync.path, wait.path, overlap.path, cB.path]
        experiment1.call_tree = io_helper.create_call_tree(experiment1.callpaths)
        experiment1.measurements = {
            (evt_sync.path, metric): [Measurement(Coordinate(c), None, None, 10 * c ** 2) for c in range(1, 6)],
            (wait.path, metric): [Measurement(Coordinate(c), None, None, 10 * c) for c in range(1, 6)],
            (overlap.path, metric): [Measurement(Coordinate(c), None, None, 5 * c) for c in range(1, 6)],
            (cB.path, metric): [Measurement(Coordinate(c), None, None, 100 * np.log2(c)) for c in range(1, 6)]
        }
        experiment2 = copy.deepcopy(experiment1)
        experiment2.measurements = {
            (evt_sync.path, metric): [Measurement(Coordinate(c), None, None, 10 * c ** 3) for c in range(1, 6)],
            (wait.path, metric): [Measurement(Coordinate(c), None, None, 10 * c ** 2) for c in range(1, 6)],
            (overlap.path, metric): [Measurement(Coordinate(c), None, None, 5 * c ** 2) for c in range(1, 6)]
        }
        experiment2.callpaths = [evt_sync.path, wait.path, overlap.path]
        experiment2.call_tree = io_helper.create_call_tree(experiment2.callpaths)
        ModelGenerator(experiment1).model_all()
        ModelGenerator(experiment2).model_all()
        experiment = ComparisonExperiment(experiment1, experiment2, SmartMatcher())
        experiment.do_comparison()
        self.check_comparison_against_source(experiment, experiment1)
        self.check_comparison_against_source(experiment, experiment2)
        for cp, s_cp in [(Callpath("main->sync->EVT_SYNC"), evt_sync.path),
                         (Callpath("main->sync->EVT_SYNC->WAIT"), wait.path)]:
            print(cp)
            model = experiment.modelers[0].models[(cp, metric)]
            model1 = experiment1.modelers[0].models[s_cp, metric]
            model2 = experiment2.modelers[0].models[s_cp, metric]
            self.assertEqual(ComparisonModel, type(model))
            self.assertEqual(ComparisonFunction, type(model.hypothesis.function))
            self.assertEqual(model.hypothesis.function.functions[0],
                             model1.hypothesis.function)
            self.assertEqual(model.hypothesis.function.functions[1],
                             model2.hypothesis.function)
            self.assertEqual(model.hypothesis.function.to_string(),
                             '(' + model1.hypothesis.function.to_string() + ', ' + model2.hypothesis.function.to_string() + ')')
        print(Callpath("main->sync->EVT_SYNC->OVERLAP"))
        model = experiment.modelers[0].models[(Callpath("main->sync->EVT_SYNC->OVERLAP"), metric)]
        model1 = experiment1.modelers[0].models[overlap.path, metric]
        model2 = experiment2.modelers[0].models[overlap.path, metric]
        self.assertEqual(ComparisonModel, type(model))
        self.assertEqual(ComparisonFunction, type(model.hypothesis.function))
        self.assertNotEqual(model.hypothesis.function.functions[0],
                            model1.hypothesis.function)
        self.assertEqual(model.hypothesis.function.functions[1],
                         model2.hypothesis.function)
        self.assertNotEqual(model.hypothesis.function.to_string(),
                            '(' + model1.hypothesis.function.to_string() + ', ' + model2.hypothesis.function.to_string() + ')')

    def test_smart_comparison_multi_level(self):
        metric = Metric('time')
        experiment1 = Experiment()
        experiment1.parameters = [Parameter('n')]
        experiment1.metrics = [metric]
        experiment1.coordinates = [Coordinate(c) for c in range(1, 6)]
        cB = Node("cB", Callpath("_start->main->sync->EVT_SYNC->OVERLAP->cB"), [])
        cA = Node("cA", Callpath("_start->main->sync->EVT_SYNC->OVERLAP->cA"), [])
        overlap = Node("OVERLAP", Callpath("_start->main->sync->EVT_SYNC->OVERLAP"), [cA, cB])
        wait = Node("WAIT", Callpath("_start->main->sync->EVT_SYNC->WAIT"), [])
        evt_sync = Node("EVT_SYNC", Callpath("_start->main->sync->EVT_SYNC"), [wait, overlap])
        sync = Node("sync", Callpath("_start->main->sync"), [evt_sync])
        wA = Node("wA", Callpath("_start->main->work->wA"), [])
        wBB = Node("wBB", Callpath("_start->main->work->wB->wBB"), [])
        wB = Node("wA", Callpath("_start->main->work->wB"), [wBB])
        work = Node("work", Callpath("_start->main->work"), [wA, wB])
        main = Node("main", Callpath("_start->main"), [sync, work])
        start = Node("_start", Callpath("_start"), [main])
        root = CallTree()
        root.add_child_node(start)
        experiment1.callpaths = [evt_sync.path, wait.path, overlap.path, cB.path, work.path]
        experiment1.call_tree = io_helper.create_call_tree(experiment1.callpaths)
        experiment1.measurements = {
            (evt_sync.path, metric): [Measurement(Coordinate(c), None, None, 10 * c ** 2) for c in range(1, 6)],
            (wait.path, metric): [Measurement(Coordinate(c), None, None, 10 * c) for c in range(1, 6)],
            (overlap.path, metric): [Measurement(Coordinate(c), None, None, 5 * c) for c in range(1, 6)],
            (cB.path, metric): [Measurement(Coordinate(c), None, None, 100 * np.log2(c)) for c in range(1, 6)]
        }
        experiment2 = copy.deepcopy(experiment1)
        experiment2.measurements = {
            (evt_sync.path, metric): [Measurement(Coordinate(c), None, None, 10 * c ** 3) for c in range(1, 6)],
            (wait.path, metric): [Measurement(Coordinate(c), None, None, 10 * c ** 2) for c in range(1, 6)],
            (overlap.path, metric): [Measurement(Coordinate(c), None, None, 5 * c ** 2) for c in range(1, 6)],
            (wBB.path, metric): [Measurement(Coordinate(c), None, None, 5 * c ** 2) for c in range(1, 6)]
        }
        experiment2.callpaths = [evt_sync.path, wait.path, overlap.path, wBB.path]
        experiment2.call_tree = io_helper.create_call_tree(experiment2.callpaths)
        ModelGenerator(experiment1).model_all()
        ModelGenerator(experiment2).model_all()
        experiment = ComparisonExperiment(experiment1, experiment2, SmartMatcher())
        experiment.do_comparison()
        self.check_comparison_against_source(experiment, experiment1)
        self.check_comparison_against_source(experiment, experiment2)
        for cp, s_cp in [(Callpath("main->sync->EVT_SYNC"), evt_sync.path),
                         (Callpath("main->sync->EVT_SYNC->WAIT"), wait.path)]:
            print(cp)
            model = experiment.modelers[0].models[(cp, metric)]
            model1 = experiment1.modelers[0].models[s_cp, metric]
            model2 = experiment2.modelers[0].models[s_cp, metric]
            self.assertEqual(ComparisonModel, type(model))
            self.assertEqual(ComparisonFunction, type(model.hypothesis.function))
            self.assertEqual(model.hypothesis.function.functions[0],
                             model1.hypothesis.function)
            self.assertEqual(model.hypothesis.function.functions[1],
                             model2.hypothesis.function)
            self.assertEqual(model.hypothesis.function.to_string(),
                             '(' + model1.hypothesis.function.to_string() + ', ' + model2.hypothesis.function.to_string() + ')')
        print(Callpath("main->sync->EVT_SYNC->OVERLAP"))
        model = experiment.modelers[0].models[(Callpath("main->sync->EVT_SYNC->OVERLAP"), metric)]
        model1 = experiment1.modelers[0].models[overlap.path, metric]
        model2 = experiment2.modelers[0].models[overlap.path, metric]
        self.assertEqual(ComparisonModel, type(model))
        self.assertEqual(ComparisonFunction, type(model.hypothesis.function))
        self.assertNotEqual(model.hypothesis.function.functions[0],
                            model1.hypothesis.function)
        self.assertEqual(model.hypothesis.function.functions[1],
                         model2.hypothesis.function)
        self.assertNotEqual(model.hypothesis.function.to_string(),
                            '(' + model1.hypothesis.function.to_string() + ', ' + model2.hypothesis.function.to_string() + ')')

    def test_smart_comparison_measurements_format(self):
        metric = Metric('time')
        experiment1 = Experiment()
        experiment1.parameters = [Parameter('n')]
        experiment1.metrics = [metric]
        experiment1.coordinates = [Coordinate(c) for c in range(1, 6)]
        wA = Node("wA", Callpath("_start->main->work->wA"), [])
        wB = Node("wA", Callpath("_start->main->work->wB"), [])
        work = Node("work", Callpath("_start->main->work"), [wA, wB])
        main = Node("main", Callpath("_start->main"), [work])
        start = Node("_start", Callpath("_start"), [main])
        root = CallTree()
        root.add_child_node(start)
        experiment1.callpaths = [work.path, wA.path, start.path]
        experiment1.call_tree = io_helper.create_call_tree(experiment1.callpaths)
        experiment1.measurements = {
            (work.path, metric): [Measurement(Coordinate(c), None, None, 10 * c ** 2) for c in range(1, 6)],
            (wA.path, metric): [Measurement(Coordinate(c), None, None, 10 * c) for c in range(1, 6)],
            (start.path, metric): [Measurement(Coordinate(c), None, None, 5 * c) for c in range(1, 6)]
        }
        experiment2 = copy.deepcopy(experiment1)
        experiment2.measurements = {
            (work.path, metric): [Measurement(Coordinate(c), None, None, 10 * c ** 3) for c in range(1, 6)],
            (wB.path, metric): [Measurement(Coordinate(c), None, None, 10 * c ** 2) for c in range(1, 6)],
            (start.path, metric): [Measurement(Coordinate(c), None, None, 5 * c ** 2) for c in range(1, 6)]
        }
        experiment2.callpaths = [work.path, wB.path, start.path]
        experiment2.call_tree = io_helper.create_call_tree(experiment2.callpaths)
        mg1 = ModelGenerator(experiment1)
        mg1.model_all()
        mg2 = ModelGenerator(experiment2)
        mg2.model_all()

        for name, matcher_class in all_matchers.items():
            print("Testing:", name)
            experiment = ComparisonExperiment(experiment1, experiment2, matcher_class())
            experiment.do_comparison()
            self.check_comparison_against_source(experiment, experiment1)
            self.check_comparison_against_source(experiment, experiment2)

            for c in experiment.callpaths:
                self.assertIsInstance(c, Callpath)

            for m in experiment.metrics:
                self.assertIsInstance(m, Metric)

            for k, ms in experiment.measurements.items():
                self.assertIsInstance(ms, list)
                for m in ms:
                    self.assertIsInstance(m, Measurement)

    def test_smart_comparison_of_aggregated_models_serialization(self):
        metric = Metric('time')
        experiment1 = Experiment()
        experiment1.parameters = [Parameter('n')]
        experiment1.metrics = [metric]
        experiment1.coordinates = [Coordinate(c) for c in range(1, 6)]
        wA = Node("wA", Callpath("_start->main->work->wA"), [])
        wB = Node("wA", Callpath("_start->main->work->wB"), [])
        work = Node("work", Callpath("_start->main->work"), [wA, wB])
        main = Node("main", Callpath("_start->main"), [work])
        start = Node("_start", Callpath("_start"), [main])
        root = CallTree()
        root.add_child_node(start)
        experiment1.callpaths = [work.path, wA.path, start.path]
        experiment1.call_tree = io_helper.create_call_tree(experiment1.callpaths)
        experiment1.measurements = {
            (work.path, metric): [Measurement(Coordinate(c), work.path, metric, 10 * c ** 2) for c in range(1, 6)],
            (wA.path, metric): [Measurement(Coordinate(c), wA.path, metric, 10 * c) for c in range(1, 6)],
            (start.path, metric): [Measurement(Coordinate(c), start.path, metric, 5 * c) for c in range(1, 6)]
        }
        experiment2 = copy.deepcopy(experiment1)
        experiment2.measurements = {
            (work.path, metric): [Measurement(Coordinate(c), work.path, metric, 10 * c ** 3) for c in range(1, 6)],
            (wB.path, metric): [Measurement(Coordinate(c), wB.path, metric, 10 * c ** 2) for c in range(1, 6)],
            (start.path, metric): [Measurement(Coordinate(c), start.path, metric, 5 * c ** 2) for c in range(1, 6)]
        }
        experiment2.callpaths = [work.path, wB.path, start.path]
        experiment2.call_tree = io_helper.create_call_tree(experiment2.callpaths)
        mg1 = ModelGenerator(experiment1)
        mg1.model_all()
        mg1.aggregate(SumAggregation())
        mg2 = ModelGenerator(experiment2)
        mg2.model_all()
        mg2.aggregate(SumAggregation())
        experiment = ComparisonExperiment(experiment1, experiment2, SmartMatcher())
        experiment.do_comparison()
        self.check_comparison_against_source(experiment, experiment1)
        self.check_comparison_against_source(experiment, experiment2)

        schema = ExperimentSchema()
        data = schema.dump(experiment)
        reconstructed = schema.load(data)

        self.assertListEqual(experiment.parameters, reconstructed.parameters)
        self.assertEqual(len(experiment.measurements), len(reconstructed.measurements))
        # for key in experiment.measurements:
        #     print(key)
        #     self.assertEqual(experiment.measurements[key], reconstructed.measurements[key])
        # self.assertDictEqual(experiment.measurements, reconstructed.measurements)
        self.assertListEqual(experiment.coordinates, reconstructed.coordinates)
        self.assertListEqual(experiment.callpaths, reconstructed.callpaths)
        self.assertListEqual(experiment.metrics, reconstructed.metrics)
        self.assertEqual(experiment.call_tree, reconstructed.call_tree)
        self.assertListEqual(experiment.modelers, reconstructed.modelers)
        self.assertEqual(experiment.scaling, reconstructed.scaling)

    def check_comparison_against_source(self, experiment, experiment1):
        self.assertSetEqual(set(experiment.coordinates), set(experiment1.coordinates))
        self.assertListEqual(experiment.parameters, experiment1.parameters)
        self.assertTrue(set(experiment.metrics) <= set(experiment1.metrics))
