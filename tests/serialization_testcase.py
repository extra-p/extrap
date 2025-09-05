# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest
from typing import TYPE_CHECKING

from extrap.entities.experiment import Experiment
from extrap.modelers.modeler_options import modeler_options

if TYPE_CHECKING:
    class TestCaseProtocol(unittest.TestCase):
        ...
else:
    class TestCaseProtocol:
        ...


class BasicExperimentSerializationTest(TestCaseProtocol):
    experiment: Experiment
    reconstructed: Experiment

    def test_setup(self):
        self.setUpClass()

    def test_parameters(self):
        self.assertListEqual(self.experiment.parameters, self.reconstructed.parameters)

    def test_measurements(self):
        self.assertDictEqual(self.experiment.measurements, self.reconstructed.measurements)

    def test_coordinates(self):
        self.assertListEqual(self.experiment.coordinates, self.reconstructed.coordinates)

    def test_callpaths(self):
        self.assertListEqual(self.experiment.callpaths, self.reconstructed.callpaths)

    def test_metrics(self):
        self.assertListEqual(self.experiment.metrics, self.reconstructed.metrics)

    def test_call_tree(self):
        self.assertEqual(self.experiment.call_tree, self.reconstructed.call_tree)

    def test_modelers(self):
        if self.experiment.modelers != self.reconstructed.modelers:
            self.assertEqual(len(self.experiment.modelers), len(self.reconstructed.modelers))
            for (e_mg, r_mg) in zip(self.experiment.modelers, self.reconstructed.modelers):
                self.assertEqual(e_mg._modeler.NAME, r_mg._modeler.NAME)
                self.assertEqual(e_mg._modeler.use_measure, r_mg._modeler.use_measure)
                self.assertTrue(modeler_options.equal(e_mg, r_mg))
                self.assertEqual(len(e_mg.models), len(r_mg.models))

    def test_hypothesis(self):
        for (e_mg, r_mg) in zip(self.experiment.modelers, self.reconstructed.modelers):
            for k in e_mg.models:
                self.assertListEqual(e_mg.models[k].measurements, r_mg.models[k].measurements)
                self.assertEqual(e_mg.models[k].callpath, r_mg.models[k].callpath)
                self.assertEqual(e_mg.models[k].metric, r_mg.models[k].metric)

                e_hypothesis_val = {k: v for k, v in e_mg.models[k].hypothesis.__dict__.items() if k != 'function'}
                r_hypothesis_val = {k: v for k, v in r_mg.models[k].hypothesis.__dict__.items() if k != 'function'}
                self.assertDictEqual(e_hypothesis_val, r_hypothesis_val)

    def test_scaling(self):
        self.assertEqual(self.experiment.scaling, self.reconstructed.scaling)


class BasicMPExperimentSerializationTest(BasicExperimentSerializationTest):
    def test_functions(self):
        for (e_mg, r_mg) in zip(self.experiment.modelers, self.reconstructed.modelers):
            for k in e_mg.models:
                e_mg_models_k_ = e_mg.models[k].hypothesis.function
                r_mg_models_k_ = r_mg.models[k].hypothesis.function
                self.assertAlmostEqual(e_mg_models_k_.evaluate([10, 10]), r_mg_models_k_.evaluate([10, 10]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([1000, 1000]), r_mg_models_k_.evaluate([1000, 1000]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([10000, 10000]), r_mg_models_k_.evaluate([10000, 10000]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([100000, 100000]),
                                       r_mg_models_k_.evaluate([100000, 100000]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([10, 10]), r_mg_models_k_.evaluate([10, 10]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([1000, 10]), r_mg_models_k_.evaluate([1000, 10]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([10000, 10]), r_mg_models_k_.evaluate([10000, 10]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([100000, 10]), r_mg_models_k_.evaluate([100000, 10]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([10, 10]), r_mg_models_k_.evaluate([10, 10]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([10, 1000]), r_mg_models_k_.evaluate([10, 1000]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([10, 10000]), r_mg_models_k_.evaluate([10, 10000]))
                self.assertAlmostEqual(e_mg_models_k_.evaluate([10, 100000]),
                                       r_mg_models_k_.evaluate([10, 100000]))
