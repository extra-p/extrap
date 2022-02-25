# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools
import unittest

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree, Node
from extrap.entities.coordinate import Coordinate
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter


class TestLoadCompatLayer(unittest.TestCase):

    def test_read_text(self):
        # noinspection PyUnresolvedReferences
        from extrap.fileio.text_file_reader import read_text_file
        experiment = read_text_file("data/text/one_parameter_7.txt")
        self.assertEqual(len(experiment.metrics), 1)
        self.assertEqual(len(experiment.parameters), 1)
        self.assertListEqual(experiment.callpaths, [Callpath('met1')])
        p = Parameter('p')
        self.assertListEqual(experiment.parameters, [p])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(1000),
            Coordinate(2000),
            Coordinate(4000),
            Coordinate(8000),
            Coordinate(16000)
        ])

    def test_read_talpas(self):
        # noinspection PyUnresolvedReferences
        from extrap.fileio.talpas_file_reader import read_talpas_file
        experiment = read_talpas_file("data/talpas/talpas_1.txt")
        x = Parameter('x')
        self.assertListEqual(experiment.parameters, [x])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(20),
            Coordinate(30),
            Coordinate(40),
            Coordinate(50),
            Coordinate(60)
        ])
        self.assertListEqual(experiment.metrics, [
            Metric('time')
        ])
        self.assertListEqual(experiment.callpaths, [
            Callpath('compute')
        ])

    def test_extrap3_multiparameter_experiment(self):
        # noinspection PyUnresolvedReferences
        from extrap.fileio.extrap3_experiment_reader import read_extrap3_experiment
        experiment = read_extrap3_experiment('data/input/experiment_3_mp')
        self.assertListEqual([Parameter('x'), Parameter('y'), Parameter('z')], experiment.parameters)
        self.assertSetEqual({Coordinate(1, 1, 1), Coordinate(1, 1, 10), Coordinate(1, 1, 25),
                             Coordinate(1, 10, 1), Coordinate(1, 10, 10), Coordinate(1, 10, 25),
                             Coordinate(1, 25, 1), Coordinate(1, 25, 10), Coordinate(1, 25, 25),
                             Coordinate(10, 1, 1), Coordinate(10, 1, 10), Coordinate(10, 1, 25),
                             Coordinate(10, 10, 1), Coordinate(10, 10, 10), Coordinate(10, 10, 25),
                             Coordinate(10, 25, 1), Coordinate(10, 25, 10), Coordinate(10, 25, 25),
                             Coordinate(25, 1, 1), Coordinate(25, 1, 10), Coordinate(25, 1, 25),
                             Coordinate(25, 10, 1), Coordinate(25, 10, 10), Coordinate(25, 10, 25),
                             Coordinate(25, 25, 1), Coordinate(25, 25, 10), Coordinate(25, 25, 25)
                             }, set(experiment.coordinates))
        self.assertSetEqual({Callpath('main'), Callpath('main->init_mat'), Callpath('main->zero_mat'),
                             Callpath('main->mat_mul')}, set(experiment.callpaths))
        self.assertSetEqual({Callpath('main'), Callpath('main->init_mat'), Callpath('main->zero_mat'),
                             Callpath('main->mat_mul')}, set(experiment.callpaths))
        call_tree = CallTree()
        main = Node('main', Callpath('main'))
        call_tree.add_child_node(main)
        init_mat = Node('init_mat', Callpath('main->init_mat'))
        main.add_child_node(init_mat)
        zero_mat = Node('zero_mat', Callpath('main->zero_mat'))
        main.add_child_node(zero_mat)
        mat_mul = Node('mat_mul', Callpath('main->mat_mul'))
        main.add_child_node(mat_mul)
        self.assertEqual(call_tree, experiment.call_tree)

    def test_read_json(self):
        Parameter.ID_COUNTER = itertools.count()
        # noinspection PyUnresolvedReferences
        from extrap.fileio.json_file_reader import read_json_file
        experiment = read_json_file("data/json/input_1.JSON")
        self.assertListEqual(experiment.parameters, [Parameter('x')])
        self.assertListEqual([p.id for p in experiment.parameters], [0])
        self.assertListEqual(experiment.coordinates, [
            Coordinate(4),
            Coordinate(8),
            Coordinate(16),
            Coordinate(32),
            Coordinate(64)
        ])
        self.assertListEqual(experiment.metrics, [
            Metric('time')
        ])
        self.assertListEqual(experiment.callpaths, [
            Callpath('sweep')
        ])

    def test_read_jsonlines(self):
        Parameter.ID_COUNTER = itertools.count()
        # noinspection PyUnresolvedReferences
        from extrap.fileio.jsonlines_file_reader import read_jsonlines_file
        experiment = read_jsonlines_file("data/jsonlines/test1.jsonl")
        self.assertListEqual([Parameter('x'), Parameter('y')], experiment.parameters)
        self.assertListEqual([0, 1], [p.id for p in experiment.parameters])
        self.assertListEqual([Coordinate(x, y) for x in range(1, 5 + 1) for y in range(1, 5 + 1)],
                             experiment.coordinates)
        self.assertListEqual([
            Metric('metr')
        ], experiment.metrics)
        self.assertListEqual([
            Callpath('<root>')
        ], experiment.callpaths)

    def test_read_cube(self):
        # noinspection PyUnresolvedReferences
        from extrap.fileio.cube_file_reader2 import read_cube_file
        experiment = read_cube_file('data/cubeset/single_parameter', 'weak')
        self.assertListEqual([Parameter('x')], experiment.parameters)
        self.assertSetEqual({Coordinate(1), Coordinate(10), Coordinate(25), Coordinate(50), Coordinate(100),
                             Coordinate(250), Coordinate(500), Coordinate(1000), Coordinate(2000)
                             }, set(experiment.coordinates))
        self.assertSetEqual({Callpath('main'), Callpath('main->init_mat'), Callpath('main->zero_mat'),
                             Callpath('main->mat_mul')}, set(experiment.callpaths))
        self.assertSetEqual({Metric('visits'), Metric('time'), Metric('min_time'), Metric('max_time'),
                             Metric('PAPI_FP_OPS'), Metric('PAPI_L3_TCM'), Metric('PAPI_L2_TCM')},
                            set(experiment.metrics))