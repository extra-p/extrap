# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree, Node
from extrap.entities.coordinate import Coordinate
from extrap.entities.parameter import Parameter
from extrap.fileio.file_reader.extrap3_experiment_reader import Extrap3ExperimentFileReader


class TestLoadExtraP3Experiment(unittest.TestCase):
    def test_extrap3_experiment(self):
        experiment = Extrap3ExperimentFileReader().read_experiment('data/input/experiment_3')

    def test_extrap3_multiparameter_experiment(self):
        experiment = Extrap3ExperimentFileReader().read_experiment('data/input/experiment_3_mp')
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

    def test_sparse_experiment(self):
        experiment = Extrap3ExperimentFileReader().read_experiment('data/input/experiment_3_sparse')
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
        call_tree = CallTree()
        main = Node('main', Callpath('main'))
        call_tree.add_child_node(main)
        init_mat = Node('init_mat', Callpath('main->init_mat'))
        main.add_child_node(init_mat)
        zero_mat = Node('zero_mat', Callpath('main->zero_mat'))
        main.add_child_node(zero_mat)
        mat_mul = Node('mat_mul', Callpath('main->mat_mul'))
        main.add_child_node(mat_mul)


if __name__ == '__main__':
    unittest.main()
