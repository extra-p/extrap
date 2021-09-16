# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import json
import tempfile
import unittest
import warnings
import zipfile

import extrap
from extrap.fileio.experiment_io import write_experiment, read_experiment, EXPERIMENT_DATA_FILE
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.modelers.model_generator import ModelGenerator


class TestMultiParameterAfterModeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = TextFileReader().read_experiment("data/text/two_parameter_3.txt")
        ModelGenerator(cls.experiment).model_all()
        with tempfile.TemporaryFile() as tmp:
            write_experiment(cls.experiment, tmp)
            cls.reconstructed = read_experiment(tmp)

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


class TestVersionCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = TextFileReader().read_experiment("data/text/two_parameter_3.txt")
        ModelGenerator(cls.experiment).model_all()

    def test_setup(self):
        self.setUpClass()

    def test_no_warning_on_same_version(self):
        with tempfile.TemporaryFile() as tmp:
            write_experiment(self.experiment, tmp)
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter('ignore', DeprecationWarning)
                read_experiment(tmp)
            self.assertFalse(record)

    def test_no_warning_on_earlier_version(self):
        with tempfile.TemporaryFile() as tmp:
            write_experiment(self.experiment, tmp)
            with zipfile.ZipFile(tmp, 'r', allowZip64=True) as file:
                data = json.loads(file.read(EXPERIMENT_DATA_FILE).decode("utf-8"))
                data[extrap.__title__] = '4.0.0'
            with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as file:
                file.writestr(EXPERIMENT_DATA_FILE, json.dumps(data))
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter('ignore', DeprecationWarning)
                read_experiment(tmp)
            self.assertFalse(record)

    def test_no_warning_on_developer_version(self):
        with tempfile.TemporaryFile() as tmp:
            write_experiment(self.experiment, tmp)
            with zipfile.ZipFile(tmp, 'r', allowZip64=True) as file:
                data = json.loads(file.read(EXPERIMENT_DATA_FILE).decode("utf-8"))
                data[extrap.__title__] = data[extrap.__title__] + '-alpha1'
            with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as file:
                file.writestr(EXPERIMENT_DATA_FILE, json.dumps(data))
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter('ignore', DeprecationWarning)
                read_experiment(tmp)
            self.assertFalse(record)

    def test_no_warning_on_newer_bugfix_version(self):
        with tempfile.TemporaryFile() as tmp:
            write_experiment(self.experiment, tmp)
            with zipfile.ZipFile(tmp, 'r', allowZip64=True) as file:
                data = json.loads(file.read(EXPERIMENT_DATA_FILE).decode("utf-8"))
                data[extrap.__title__] = data[extrap.__title__] + '1'
            with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as file:
                file.writestr(EXPERIMENT_DATA_FILE, json.dumps(data))
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter('ignore', DeprecationWarning)
                read_experiment(tmp)
            self.assertFalse(record)

    def test_warning_on_newer_version(self):
        with tempfile.TemporaryFile() as tmp:
            write_experiment(self.experiment, tmp)
            with zipfile.ZipFile(tmp, 'r', allowZip64=True) as file:
                data = json.loads(file.read(EXPERIMENT_DATA_FILE).decode("utf-8"))
                data[extrap.__title__] = '1' + data[extrap.__title__]
            with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as file:
                file.writestr(EXPERIMENT_DATA_FILE, json.dumps(data))
            self.assertWarnsRegex(UserWarning, 'newer version', read_experiment, tmp)


if __name__ == '__main__':
    unittest.main()
