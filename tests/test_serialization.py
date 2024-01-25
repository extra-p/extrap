# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

from marshmallow import ValidationError

from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import ExperimentSchema, Experiment
from extrap.entities.functions import ConstantFunction
from extrap.entities.hypotheses import ConstantHypothesisSchema, ConstantHypothesis
from extrap.entities.measurement import Measurement
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.modelers.model_generator import ModelGenerator
from tests.serialization_testcase import BasicExperimentSerializationTest, BasicMPExperimentSerializationTest


class TestSingleParameter(BasicExperimentSerializationTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = TextFileReader().read_experiment("data/text/one_parameter_1.txt")
        schema = ExperimentSchema()
        # print(json.dumps(schema.dump(cls.experiment), indent=1))
        exp_str = schema.dumps(cls.experiment)
        cls.reconstructed: Experiment = schema.loads(exp_str)


class TestSingleParameterAfterModeling(BasicExperimentSerializationTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = TextFileReader().read_experiment("data/text/one_parameter_1.txt")
        ModelGenerator(cls.experiment).model_all()
        schema = ExperimentSchema()
        # print(json.dumps(schema.dump(cls.experiment), indent=1))
        exp_str = schema.dumps(cls.experiment)
        cls.reconstructed: Experiment = schema.loads(exp_str)


class TestSerialization(unittest.TestCase):

    def test_validation(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_1.txt")
        schema = ExperimentSchema()
        exp_data = schema.dump(experiment)
        val_erros = schema.validate(exp_data)
        self.assertDictEqual({}, val_erros)

    def test_additional_keys_in_experiment(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_1.txt")
        schema = ExperimentSchema()
        exp_data = schema.dump(experiment)
        # print(json.dumps(exp_data, indent=1))
        exp_data['TEST_ATTRIBUTE'] = 'TEST_ATTRIBUTE'
        reconstructed: Experiment = schema.load(exp_data)
        self.assertFalse(hasattr(reconstructed, 'TEST_ATTRIBUTE'))

    def test_additional_keys_in_measurements(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_1.txt")
        schema = ExperimentSchema()
        exp_data = schema.dump(experiment)
        # print(json.dumps(exp_data, indent=1))
        exp_data['measurements']['TEST_ATTRIBUTE'] = 'TEST_ATTRIBUTE'
        self.assertRaises(ValidationError, schema.load, exp_data)

    def test_additional_keys_in_measurement_obj(self):
        experiment = TextFileReader().read_experiment("data/text/one_parameter_1.txt")
        schema = ExperimentSchema()
        exp_data = schema.dump(experiment)
        # print(json.dumps(exp_data, indent=1))
        exp_data['measurements']['compute']['time'][0]['TEST_ATTRIBUTE'] = 'TEST_ATTRIBUTE'
        reconstructed: Experiment = schema.load(exp_data)
        self.assertFalse(hasattr(reconstructed, 'TEST_ATTRIBUTE'))

    def test_serialize_constant_hypothesis(self):
        schema = ConstantHypothesisSchema()
        hyp = ConstantHypothesis(ConstantFunction(12.0), False)
        hyp.compute_cost(
            [Measurement(Coordinate(1), None, None, [11.9]), Measurement(Coordinate(2), None, None, [11.9])])
        hyp_data = schema.dump(hyp)
        reconstructed: ConstantHypothesis = schema.load(hyp_data)
        self.assertEqual(hyp, reconstructed)


class TestMultiParameter(BasicMPExperimentSerializationTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = TextFileReader().read_experiment("data/text/two_parameter_3.txt")
        schema = ExperimentSchema()
        # print(json.dumps(schema.dump(cls.experiment), indent=1))
        exp_str = schema.dumps(cls.experiment)
        cls.reconstructed: Experiment = schema.loads(exp_str)


class TestMultiParameterAfterModeling(BasicMPExperimentSerializationTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = TextFileReader().read_experiment("data/text/two_parameter_3.txt")
        ModelGenerator(cls.experiment).model_all()
        schema = ExperimentSchema()
        # print(json.dumps(schema.dump(cls.experiment), indent=1))
        exp_str = schema.dumps(cls.experiment)
        cls.reconstructed: Experiment = schema.loads(exp_str)


if __name__ == '__main__':
    unittest.main()
