# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import contextlib
import shutil
import tempfile
import unittest
from io import StringIO

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

    def test_cube(self):
        extrap.main(['--cube', 'data/cubeset/single_parameter/'])
        self.assertRaisesRegex(SystemExit, '[^0]', extrap.main, ['--cube', 'data/cubeset/does_not_exist'])
        self.assertRaisesRegex(SystemExit, '[^0]', extrap.main, ['--cube', 'data/text/one_parameter_1.txt'])

    def test_modeler(self):
        extrap.main(['--modeler', 'Default', '--text', 'data/text/one_parameter_1.txt'])
        self.assertRaisesRegex(SystemExit, '[^0]', extrap.main,
                               ['--modeler', 'does_not_exist', '--text', 'data/text/one_parameter_1.txt'])

    def test_print(self):
        extrap.main(['--text', 'data/text/one_parameter_1.txt'])
        self.assertOutputRegex(
            "Callpath\:\s+compute\s+"
            "Metric\:\s+time\s+"
            "Measurement\s+point\:\s+\(2\.00E\+01\)\s+Mean\:\s+8\.19E\+01\s+Median\:\s+8\.20E\+01\s+"
            "Measurement\s+point\:\s+\(3\.00E\+01\)\s+Mean\:\s+1\.79E\+02\s+Median\:\s+1\.78E\+02\s+"
            "Measurement\s+point\:\s+\(4\.00E\+01\)\s+Mean\:\s+3\.19E\+02\s+Median\:\s+3\.19E\+02\s+"
            "Measurement\s+point\:\s+\(5\.00E\+01\)\s+Mean\:\s+5\.05E\+02\s+Median\:\s+5\.06E\+02\s+"
            "Measurement\s+point\:\s+\(6\.00E\+01\)\s+Mean\:\s+7\.25E\+02\s+Median\:\s+7\.26E\+02\s+"
            "Model\:\s+\-0\.88979340\d+\s+\+\s+0\.20168243\d+\s+\*\s+x\^\(2\)\s+"
            "RSS\:\s+3\.43E\+01\s+"
            "Adjusted\s+R\^2\:\s+1\.00E\+00",
            extrap.main, ['--print', 'all', '--text', 'data/text/one_parameter_1.txt'])  # noqa
        # noqa
        self.assertOutputRegex(r'-0\.88979340\d+ \+ 0\.20168243\d+ \* x\^\(2\)', extrap.main,
                               ['--print', 'functions', '--text', 'data/text/one_parameter_1.txt'])
        extrap.main(['--print', 'callpaths', '--text', 'data/text/one_parameter_1.txt'])
        extrap.main(['--print', 'metrics', '--text', 'data/text/one_parameter_1.txt'])
        extrap.main(['--print', 'parameters', '--text', 'data/text/one_parameter_1.txt'])

        extrap.main(['--print', 'all', '--text', 'data/text/two_parameter_1.txt'])
        self.assertOutputRegex(
            r'1\.37420831\d+ \+ 6\.69808039\d+ \* log2\(y\)\^\(1\) \+ 0\.04384165\d+ \* x\^\(3\/2\) \* '
            r'log2\(x\)\^\(2\) \* log2\(y\)\^\(1\)',
            extrap.main,
            ['--print', 'functions', '--text', 'data/text/two_parameter_1.txt'])
        self.assertOutput('reg', extrap.main, ['--print', 'callpaths', '--text', 'data/text/two_parameter_1.txt'])
        extrap.main(['--print', 'metrics', '--text', 'data/text/two_parameter_1.txt'])
        extrap.main(['--print', 'parameters', '--text', 'data/text/two_parameter_1.txt'])

    def test_print_python(self):
        self.assertOutputRegex(
            "Callpath\:\s+compute\s+"
            "Metric\:\s+time\s+"
            "Measurement\s+point\:\s+\(2\.00E\+01\)\s+Mean\:\s+8\.19E\+01\s+Median\:\s+8\.20E\+01\s+"
            "Measurement\s+point\:\s+\(3\.00E\+01\)\s+Mean\:\s+1\.79E\+02\s+Median\:\s+1\.78E\+02\s+"
            "Measurement\s+point\:\s+\(4\.00E\+01\)\s+Mean\:\s+3\.19E\+02\s+Median\:\s+3\.19E\+02\s+"
            "Measurement\s+point\:\s+\(5\.00E\+01\)\s+Mean\:\s+5\.05E\+02\s+Median\:\s+5\.06E\+02\s+"
            "Measurement\s+point\:\s+\(6\.00E\+01\)\s+Mean\:\s+7\.25E\+02\s+Median\:\s+7\.26E\+02\s+"
            "Model\:\s+\-0\.88979\d+\+0\.20168\d+\*x\*\*\(2\)\s+"
            "RSS\:\s+3\.43E\+01\s+"
            "Adjusted\s+R\^2\:\s+1\.00E\+00",
            extrap.main, ['--print', 'all-python', '--text', 'data/text/one_parameter_1.txt'])  # noqa
        # noqa
        self.assertOutputRegex(r"-0\.88979\d+\+0\.20168\d+\*x\*\*\(2\)", extrap.main,
                               ['--print', 'functions-python', '--text', 'data/text/one_parameter_1.txt'])

        extrap.main(['--print', 'all-python', '--text', 'data/text/two_parameter_1.txt'])
        self.assertOutputRegex(r"1\.37420\d+\+6\.69808\d+\*log2\(y\)\*\*\(1\)\+0\.043841\d+\*x\*\*\(3/2\)"
                               r"\*log2\(x\)\*\*\(2\)\*log2\(y\)\*\*\(1\)",
                               extrap.main,
                               ['--print', 'functions-python', '--text', 'data/text/two_parameter_1.txt'])

    def test_print_latex(self):
        self.assertOutputRegex(
            "Callpath\:\s+compute\s+"
            "Metric\:\s+time\s+"
            "Measurement\s+point\:\s+\(2\.00E\+01\)\s+Mean\:\s+8\.19E\+01\s+Median\:\s+8\.20E\+01\s+"
            "Measurement\s+point\:\s+\(3\.00E\+01\)\s+Mean\:\s+1\.79E\+02\s+Median\:\s+1\.78E\+02\s+"
            "Measurement\s+point\:\s+\(4\.00E\+01\)\s+Mean\:\s+3\.19E\+02\s+Median\:\s+3\.19E\+02\s+"
            "Measurement\s+point\:\s+\(5\.00E\+01\)\s+Mean\:\s+5\.05E\+02\s+Median\:\s+5\.06E\+02\s+"
            "Measurement\s+point\:\s+\(6\.00E\+01\)\s+Mean\:\s+7\.25E\+02\s+Median\:\s+7\.26E\+02\s+"
            r"Model\:\s+\$\-8\.898\\times10\^\{−1\}\+2\.017\\times10\^\{−1\}\\cdot\ x\^\{2\}\$\s+"
            "RSS\:\s+3\.43E\+01\s+"
            "Adjusted\s+R\^2\:\s+1\.00E\+00",
            extrap.main, ['--print', 'all-latex', '--text', 'data/text/one_parameter_1.txt'])  # noqa
        # noqa
        self.assertOutputRegex(r"\$\-8\.898\\times10\^\{−1\}\+2\.017\\times10\^\{−1\}\\cdot\ x\^\{2\}\$", extrap.main,
                               ['--print', 'functions-latex', '--text', 'data/text/one_parameter_1.txt'])

        extrap.main(['--print', 'all-python', '--text', 'data/text/two_parameter_1.txt'])
        self.assertOutputRegex(
            r"\$1\.374\+6\.698\\cdot\ \\log_2\{y\}\^\{1\}\+4\.384\\times10\^\{−2\}\\cdot\ x\^\{3/2\}\\cdot\ "
            r"\\log_2\{x\}\^\{2\}\\cdot\ \\log_2\{y\}\^\{1\}\$",
            extrap.main,
            ['--print', 'functions-latex', '--text', 'data/text/two_parameter_1.txt'])

    def test_segmented_output(self):
        self.assertOutputRegex(
            r"Callpath:\s+main\s+"
            r"\s+Metric:\s+runtime\s+"
            r"\s+Measurement\s+point:\s+\(1\.00E\+00\)\s+Mean:\s+1\.00E\+00\s+Median:\s+1\.00E\+00\s+"
            r"\s+Measurement\s+point:\s+\(2\.00E\+00\)\s+Mean:\s+4\.00E\+00\s+Median:\s+4\.00E\+00\s+"
            r"\s+Measurement\s+point:\s+\(3\.00E\+00\)\s+Mean:\s+9\.00E\+00\s+Median:\s+9\.00E\+00\s+"
            r"\s+Measurement\s+point:\s+\(4\.00E\+00\)\s+Mean:\s+1\.60E\+01\s+Median:\s+1\.60E\+01\s+"
            r"\s+Measurement\s+point:\s+\(5\.00E\+00\)\s+Mean:\s+2\.50E\+01\s+Median:\s+2\.50E\+01\s+"
            r"\s+Measurement\s+point:\s+\(6\.00E\+00\)\s+Mean:\s+3\.60E\+01\s+Median:\s+3\.60E\+01\s+"
            r"\s+Measurement\s+point:\s+\(7\.00E\+00\)\s+Mean:\s+3\.70E\+01\s+Median:\s+3\.70E\+01\s+"
            r"\s+Measurement\s+point:\s+\(8\.00E\+00\)\s+Mean:\s+3\.80E\+01\s+Median:\s+3\.80E\+01\s+"
            r"\s+Measurement\s+point:\s+\(9\.00E\+00\)\s+Mean:\s+3\.90E\+01\s+Median:\s+3\.90E\+01\s+"
            r"\s+Measurement\s+point:\s+\(1\.00E\+01\)\s+Mean:\s+4\.00E\+01\s+Median:\s+4\.00E\+01\s+"
            r"\s+Model\s+1:\s+6\.082977478\d+e\-15\s+\+\s+0\.9999999999\d+\s+\*\s+p\^\(2\)\s+for\s+p<=6\.0\s+"
            r"\s+Model\s+2:\s+29\.999999999\d+\s+\+\s+1\.0000000000\d+\s+\*\s+p\^\(1\)\s+for\s+p>=6\.0\s+"
            r"\s+RSS\s+Model\s+1:\s+\d\.\d\dE\-2\d\s+"
            r"\s+Adjusted\s+R\^2\s+Model\s+1:\s+1\.00E\+00\s+"
            r"\s+RSS\s+Model\s+2:\s+\d\.\d\dE\-2\d\s+"
            r"\s+Adjusted\s+R\^2\s+Model\s+2:\s+1\.00E\+00",
            extrap.main,
            ['--text', 'data/text/one_parameter_segmented_1.txt', '--modeler', 'segmented'])

    def assertOutput(self, text, command, *args, **kwargs):
        temp_stdout = StringIO()
        with contextlib.redirect_stdout(temp_stdout):
            command(*args, **kwargs)
        output = temp_stdout.getvalue().strip()
        self.assertEqual(text, output)

    def assertOutputRegex(self, regex, command, *args, **kwargs):
        temp_stdout = StringIO()
        with contextlib.redirect_stdout(temp_stdout):
            command(*args, **kwargs)
        output = temp_stdout.getvalue().strip()
        try:
            self.assertRegex(output, regex)
        except Exception:
            print()
            print("####### ORIGINAL OUTPUT #######")
            print(output)
            print("####### END OF ORIGINAL OUTPUT #######")
            raise

    def test_experiment(self):
        temp_dir = tempfile.mkdtemp()
        try:
            extrap.main(['--save-experiment', temp_dir + '/test1.extra-p', '--model-set-name', 'test', '--text',
                         'data/text/one_parameter_1.txt'])
            experiment = read_experiment(temp_dir + '/test1.extra-p')
            self.assertEqual('test', experiment.modelers[0].name)
            extrap.main(['--save-experiment', temp_dir + '/test2', '--text', 'data/text/one_parameter_1.txt'])
            read_experiment(temp_dir + '/test2.extra-p')
            extrap.main(['--save-experiment', temp_dir + '/test3.other-ext', '--text', 'data/text/one_parameter_1.txt'])
            read_experiment(temp_dir + '/test3.other-ext')
            self.assertRaisesRegex(SystemExit, '[^0]', extrap.main,
                                   ['--save-experiment', temp_dir + '/does_not_exist/test1.extra-p',
                                    '--text', 'data/text/one_parameter_1.txt'])
            extrap.main(['--save-experiment', temp_dir + '/test4.extra-p', '--model-set-name', 'test2', '--experiment',
                         temp_dir + '/test1.extra-p'])
            experiment = read_experiment(temp_dir + '/test4.extra-p')
            self.assertEqual('test', experiment.modelers[0].name)
            self.assertEqual('test2', experiment.modelers[1].name)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
