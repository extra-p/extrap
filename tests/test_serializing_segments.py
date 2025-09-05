# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

from extrap.entities.experiment import ExperimentSchema, Experiment
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler
from extrap.modelers.single_parameter.segmented import SegmentedModeler
from tests.serialization_testcase import BasicExperimentSerializationTest


class TestSerializingSegments(BasicExperimentSerializationTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = TextFileReader().read_experiment("data/text/one_parameter_segmented_5.txt")
        ModelGenerator(cls.experiment, SegmentedModeler()).model_all()
        schema = ExperimentSchema()
        # print(json.dumps(schema.dump(cls.experiment), indent=1))
        exp_str = schema.dumps(cls.experiment)
        cls.reconstructed: Experiment = schema.loads(exp_str)


class TestSerializingMultiParameterSegments(BasicExperimentSerializationTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = TextFileReader().read_experiment("data/text/two_parameter_segmented_1.txt")
        mp = MultiParameterModeler()
        mp.single_parameter_modeler = SegmentedModeler()
        ModelGenerator(cls.experiment, mp).model_all()
        schema = ExperimentSchema()
        # print(json.dumps(schema.dump(cls.experiment), indent=1))
        exp_str = schema.dumps(cls.experiment)
        cls.reconstructed: Experiment = schema.loads(exp_str)


if __name__ == '__main__':
    unittest.main()
