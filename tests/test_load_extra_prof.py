# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import unittest

from extrap.entities.metric import Metric
from extrap.fileio.file_reader.extra_prof import ExtraProf2Reader


class TestExtraProfFileLoader(unittest.TestCase):
    def test_load_basic(self):
        experiment = ExtraProf2Reader().read_experiment('data/extra_prof/test1/')

    def test_load_basic2(self):
        experiment = ExtraProf2Reader().read_experiment('data/extra_prof/test2/')

    def test_load_basic3(self):
        experiment = ExtraProf2Reader().read_experiment('data/extra_prof/test3/')

    def test_load_basic4(self):
        experiment = ExtraProf2Reader().read_experiment('data/extra_prof/test4/')

    def test_load_with_energy(self):
        experiment = ExtraProf2Reader().read_experiment('data/extra_prof/test_with_energy/')
        self.assertIn(Metric('energy_cpu'), experiment.metrics)
        self.assertIn(Metric('energy_gpu'), experiment.metrics)
        for key, value in experiment.measurements.items():
            self.assertGreaterEqual(value[0].mean, 0)


if __name__ == '__main__':
    unittest.main()
