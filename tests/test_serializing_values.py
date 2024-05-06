# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import tempfile
import unittest
import zipfile

import numpy as np

from extrap.fileio.experiment_io import write_experiment, read_experiment
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.fileio.values_io import ValueWriter, ValueReader


class TestSerializingValues(unittest.TestCase):
    def test_load_store_values(self):
        with tempfile.TemporaryFile() as path:
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as file:
                val_writer = ValueWriter(file)
                id1 = val_writer.write_values(np.array([1, 2, 3, 4, 5, 6]))
                val_writer.write_values(np.array([2, 2]))
                id3 = val_writer.write_values(np.array([3, 3]))
                val_writer.flush()
                print('before close')

            with zipfile.ZipFile(path, 'r', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as file:
                val_reader = ValueReader(file)
                self.assertTrue(np.all(np.array([1, 2, 3, 4, 5, 6]) == val_reader.read_values(*id1)))
                self.assertTrue(np.all(np.array([3, 3]) == val_reader.read_values(*id3)))
                print('done')

                if __name__ == '__main__':
                    unittest.main()

    def test_experiment_with_values(self):
        reader = TextFileReader()
        reader.keep_values = True
        experiment_org = reader.read_experiment("data/text/two_parameter_3.txt")
        with tempfile.TemporaryFile() as path:
            write_experiment(experiment_org, path)
            experiment = read_experiment(path)
        self.assertDictEqual(experiment_org.measurements, experiment.measurements)
        for k in experiment_org.measurements:
            ms1 = experiment.measurements[k]
            ms2 = experiment_org.measurements[k]
            for m1, m2 in zip(ms1, ms2):
                self.assertTrue(np.all(m1.values == m2.values))

    def test_experiment_without_values(self):
        reader = TextFileReader()
        reader.keep_values = False
        experiment_org = reader.read_experiment("data/text/two_parameter_3.txt")
        with tempfile.TemporaryFile() as path:
            write_experiment(experiment_org, path)
            experiment = read_experiment(path)
        self.assertDictEqual(experiment_org.measurements, experiment.measurements)
        for k in experiment_org.measurements:
            ms1 = experiment.measurements[k]
            ms2 = experiment_org.measurements[k]
            for m1, m2 in zip(ms1, ms2):
                self.assertIsNone(m1.values)
                self.assertIsNone(m2.values)
