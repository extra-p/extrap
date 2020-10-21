# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

from marshmallow import ValidationError

from extrap.entities.experiment import ExperimentSchema, Experiment
from extrap.fileio.text_file_reader import read_text_file
from extrap.modelers.model_generator import ModelGenerator


class TestSingleParameter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = read_text_file("data/text/one_parameter_1.txt")
        schema = ExperimentSchema()
        # print(json.dumps(schema.dump(cls.experiment), indent=1))
        exp_str = schema.dumps(cls.experiment)
        cls.reconstructed: Experiment = schema.loads(exp_str)

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
        self.assertListEqual(self.experiment.modelers, self.reconstructed.modelers)

    def test_scaling(self):
        self.assertEqual(self.experiment.scaling, self.reconstructed.scaling)


class TestSingleParameterAfterModeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = read_text_file("data/text/one_parameter_1.txt")
        ModelGenerator(cls.experiment).model_all()
        schema = ExperimentSchema()
        # print(json.dumps(schema.dump(cls.experiment), indent=1))
        exp_str = schema.dumps(cls.experiment)
        cls.reconstructed: Experiment = schema.loads(exp_str)

    def test_setup(self):
        self.setUpClass()

    def test_parameters(self):
        self.assertListEqual(self.experiment.parameters, self.reconstructed.parameters)
        pass

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
        self.assertListEqual(self.experiment.modelers, self.reconstructed.modelers)

    def test_scaling(self):
        self.assertEqual(self.experiment.scaling, self.reconstructed.scaling)


class TestSerialization(unittest.TestCase):

    def test_validation(self):
        experiment = read_text_file("data/text/one_parameter_1.txt")
        schema = ExperimentSchema()
        exp_data = schema.dump(experiment)
        val_erros = schema.validate(exp_data)
        self.assertDictEqual({}, val_erros)

    def test_additional_keys_in_experiment(self):
        experiment = read_text_file("data/text/one_parameter_1.txt")
        schema = ExperimentSchema()
        exp_data = schema.dump(experiment)
        # print(json.dumps(exp_data, indent=1))
        exp_data['TEST_ATTRIBUTE'] = 'TEST_ATTRIBUTE'
        reconstructed: Experiment = schema.load(exp_data)
        self.assertFalse(hasattr(reconstructed, 'TEST_ATTRIBUTE'))

    def test_additional_keys_in_measurements(self):
        experiment = read_text_file("data/text/one_parameter_1.txt")
        schema = ExperimentSchema()
        exp_data = schema.dump(experiment)
        # print(json.dumps(exp_data, indent=1))
        exp_data['measurements']['TEST_ATTRIBUTE'] = 'TEST_ATTRIBUTE'
        self.assertRaises(ValidationError, schema.load, exp_data)

    def test_additional_keys_in_measurement_obj(self):
        experiment = read_text_file("data/text/one_parameter_1.txt")
        schema = ExperimentSchema()
        exp_data = schema.dump(experiment)
        # print(json.dumps(exp_data, indent=1))
        exp_data['measurements']['compute']['time'][0]['TEST_ATTRIBUTE'] = 'TEST_ATTRIBUTE'
        reconstructed: Experiment = schema.load(exp_data)
        self.assertFalse(hasattr(reconstructed, 'TEST_ATTRIBUTE'))


class TestMultiParameter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = read_text_file("data/text/two_parameter_3.txt")
        schema = ExperimentSchema()
        # print(json.dumps(schema.dump(cls.experiment), indent=1))
        exp_str = schema.dumps(cls.experiment)
        cls.reconstructed: Experiment = schema.loads(exp_str)

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
        self.assertListEqual(self.experiment.modelers, self.reconstructed.modelers)

    def test_scaling(self):
        self.assertEqual(self.experiment.scaling, self.reconstructed.scaling)


class TestMultiParameterAfterModeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = read_text_file("data/text/two_parameter_3.txt")
        ModelGenerator(cls.experiment).model_all()
        schema = ExperimentSchema()
        # print(json.dumps(schema.dump(cls.experiment), indent=1))
        exp_str = schema.dumps(cls.experiment)
        cls.reconstructed: Experiment = schema.loads(exp_str)

    def test_setup(self):
        self.setUpClass()

    def test_parameters(self):
        self.assertListEqual(self.experiment.parameters, self.reconstructed.parameters)
        pass

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
        self.assertListEqual(self.experiment.modelers, self.reconstructed.modelers)

    def test_scaling(self):
        self.assertEqual(self.experiment.scaling, self.reconstructed.scaling)


if __name__ == '__main__':
    unittest.main()
