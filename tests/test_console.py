# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
import shutil
import tempfile
import unittest

from extrap.extrap import extrapcmd as extrap
from extrap.fileio.experiment_io import read_experiment


class TestConsole(unittest.TestCase):
    def test_nothing(self):
        self.assertRaisesRegex(SystemExit, '[^0]', extrap.main)

    def test_help(self):
        self.assertRaisesRegex(SystemExit, '0', extrap.main, ['--help'])

    def test_help_modeler_options(self):
        self.assertRaisesRegex(SystemExit, '0', extrap.main, ['--help-modeler', 'Default'])

    def test_text(self):
        extrap.main(['--text', 'data/text/one_parameter_1.txt'])
        self.assertRaisesRegex(SystemExit, '[^0]', extrap.main, ['--text', 'data/text/does_not_exist.txt'])
        self.assertRaises(Exception, extrap.main, ['--text', 'data/talpas/talpas_1.txt'])

    def test_modeler(self):
        extrap.main(['--modeler', 'Default', '--text', 'data/text/one_parameter_1.txt'])
        self.assertRaisesRegex(SystemExit, '[^0]', extrap.main,
                               ['--modeler', 'does_not_exist', '--text', 'data/text/one_parameter_1.txt'])

    def test_save_experiment(self):
        temp_dir = tempfile.mkdtemp()
        try:
            extrap.main(['--save-experiment', temp_dir + '/test1.extra-p', '--text', 'data/text/one_parameter_1.txt'])
            read_experiment(temp_dir + '/test1.extra-p')
            extrap.main(['--save-experiment', temp_dir + '/test2', '--text', 'data/text/one_parameter_1.txt'])
            read_experiment(temp_dir + '/test2.extra-p')
            extrap.main(['--save-experiment', temp_dir + '/test3.other-ext', '--text', 'data/text/one_parameter_1.txt'])
            read_experiment(temp_dir + '/test3.other-ext')
            self.assertRaisesRegex(SystemExit, '[^0]', extrap.main,
                                   ['--save-experiment', temp_dir + '/does_not_exist/test1.extra-p',
                                    '--text', 'data/text/one_parameter_1.txt'])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
