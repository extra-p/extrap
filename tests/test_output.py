import contextlib
import unittest
from io import StringIO

from extrap.extrap import extrapcmd as extrap
from extrap.fileio import output
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.fileio.output import OutputFormatError
from extrap.modelers.model_generator import ModelGenerator


class TestOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.exp = TextFileReader().read_experiment('data/text/two_parameter_3.txt')
        ModelGenerator(cls.exp).model_all()

    def test_print(self):
        self.assertOutputRegex(
            "time,\s+merge\:\s+errors\:\s+3\.12E\+02,\s+7\.83E\-02\s+"
            "time,\s+sort\:\s+errors\:\s+3\.12E\+02,\s+7\.83E\-02\s+"
            "flops,\s+merge\:\s+errors\:\s+3\.12E\+02,\s+7\.83E\-02\s+"
            "flops,\s+sort\:\s+errors\:\s+3\.12E\+02,\s+7\.83E\-02",
            extrap.main,
            ['--print', '{metric}, {callpath}: Errors: {rss}, {re}', '--text', 'data/text/two_parameter_3.txt'])

        self.assertOutputRegex(
            "\(2\.00E\+01\)\s+Mean\:\s+8\.19E\+01\s+Median\:\s+8\.20E\+01\s+"
            "\(3\.00E\+01\)\s+Mean\:\s+1\.79E\+02\s+Median\:\s+1\.78E\+02\s+"
            "\(4\.00E\+01\)\s+Mean\:\s+3\.19E\+02\s+Median\:\s+3\.19E\+02\s+"
            "\(5\.00E\+01\)\s+Mean\:\s+5\.05E\+02\s+Median\:\s+5\.06E\+02\s+"
            "\(6\.00E\+01\)\s+Mean\:\s+7\.25E\+02\s+Median\:\s+7\.26E\+02",
            extrap.main, ['--print', '{measurements}', '--text', 'data/text/one_parameter_1.txt'])

        self.assertOutputRegex(
            "time,\s+merge\:\s+3\.12E\+02\/6\.64E\+00\s+1\.3742083125359215\s+\+\s+6\.698080399955742\s+\*\s+log2\(y\)\^\(1\)\s+\+\s+0\.04384165529030426\s+\*\s+x\^\(3\/2\)\s+\*\s+log2\(x\)\^\(2\)\s+\*\s+log2\(y\)\^\(1\)\s+"
            "time,\s+sort\:\s+3\.12E\+02\/6\.64E\+00\s+1\.3742083125359215\s+\+\s+6\.698080399955742\s+\*\s+log2\(y\)\^\(1\)\s+\+\s+0\.04384165529030426\s+\*\s+x\^\(3\/2\)\s+\*\s+log2\(x\)\^\(2\)\s+\*\s+log2\(y\)\^\(1\)\s+"
            "flops,\s+merge\:\s+3\.12E\+02\/6\.64E\+00\s+1\.3742083125359215\s+\+\s+6\.698080399955742\s+\*\s+log2\(y\)\^\(1\)\s+\+\s+0\.04384165529030426\s+\*\s+x\^\(3\/2\)\s+\*\s+log2\(x\)\^\(2\)\s+\*\s+log2\(y\)\^\(1\)\s+"
            "flops,\s+sort\:\s+3\.12E\+02\/6\.64E\+00\s+1\.3742083125359215\s+\+\s+6\.698080399955742\s+\*\s+log2\(y\)\^\(1\)\s+\+\s+0\.04384165529030426\s+\*\s+x\^\(3\/2\)\s+\*\s+log2\(x\)\^\(2\)\s+\*\s+log2\(y\)\^\(1\)",
            extrap.main,
            ['--print', '{metric}, {callpath}: {rss}/{smape} {model}', '--text', 'data/text/two_parameter_3.txt'])

    def test_invalid(self):
        with self.assertRaises(OutputFormatError):
            extrap.main(['--print', '{metric}, {a}', '--text', 'data/text/two_parameter_3.txt'])

    def test_basic_output(self):
        self.assertOutputRegex(
            r"time\s+flops",
            extrap.main,
            ['--print', '{metric}', '--text', 'data/text/two_parameter_3.txt'])
        self.assertOutputRegex(
            r"merge\s+sort",
            extrap.main,
            ['--print', '{callpath}', '--text', 'data/text/two_parameter_3.txt'])

    @staticmethod
    def _make_expected(point_string):
        truth = ""
        for name in ("merge,time:", "sort,time:", "merge,flops:", "sort,flops:"):
            truth += name + point_string
        return truth

    def test_point_formatting(self):
        truth = self._make_expected(
            "(2.00E+01, 1.00E+00) | (2.00E+01, 2.00E+00) | (2.00E+01, 3.00E+00) | (2.00E+01, 4.00E+00) | "
            "(2.00E+01, 5.00E+00) | (3.00E+01, 1.00E+00) | (3.00E+01, 2.00E+00) | (3.00E+01, 3.00E+00) | "
            "(3.00E+01, 4.00E+00) | (3.00E+01, 5.00E+00) | (4.00E+01, 1.00E+00) | (4.00E+01, 2.00E+00) | "
            "(4.00E+01, 3.00E+00) | (4.00E+01, 4.00E+00) | (4.00E+01, 5.00E+00) | (5.00E+01, 1.00E+00) | "
            "(5.00E+01, 2.00E+00) | (5.00E+01, 3.00E+00) | (5.00E+01, 4.00E+00) | (5.00E+01, 5.00E+00) | "
            "(6.00E+01, 1.00E+00) | (6.00E+01, 2.00E+00) | (6.00E+01, 3.00E+00) | (6.00E+01, 4.00E+00) | "
            "(6.00E+01, 5.00E+00)\n")
        self.assertEqual(truth, output.fmt_output(self.exp, '{callpath},{metric}:{points}'))

    def test_point_formatting1(self):
        truth = self._make_expected(
            "(2.00E+01, 1.00E+00);(2.00E+01, 2.00E+00);(2.00E+01, 3.00E+00);(2.00E+01, 4.00E+00);"
            "(2.00E+01, 5.00E+00);(3.00E+01, 1.00E+00);(3.00E+01, 2.00E+00);(3.00E+01, 3.00E+00);"
            "(3.00E+01, 4.00E+00);(3.00E+01, 5.00E+00);(4.00E+01, 1.00E+00);(4.00E+01, 2.00E+00);"
            "(4.00E+01, 3.00E+00);(4.00E+01, 4.00E+00);(4.00E+01, 5.00E+00);(5.00E+01, 1.00E+00);"
            "(5.00E+01, 2.00E+00);(5.00E+01, 3.00E+00);(5.00E+01, 4.00E+00);(5.00E+01, 5.00E+00);"
            "(6.00E+01, 1.00E+00);(6.00E+01, 2.00E+00);(6.00E+01, 3.00E+00);(6.00E+01, 4.00E+00);"
            "(6.00E+01, 5.00E+00)\n")
        self.assertEqual(truth, output.fmt_output(self.exp, "{callpath},{metric}:{points:sep:';';}"))

    def test_point_formatting2(self):
        truth = self._make_expected("(2.00E+01, 1.00E+00)(2.00E+01, 2.00E+00)(2.00E+01, 3.00E+00)(2.00E+01, 4.00E+00)"
                                    "(2.00E+01, 5.00E+00)(3.00E+01, 1.00E+00)(3.00E+01, 2.00E+00)(3.00E+01, 3.00E+00)"
                                    "(3.00E+01, 4.00E+00)(3.00E+01, 5.00E+00)(4.00E+01, 1.00E+00)(4.00E+01, 2.00E+00)"
                                    "(4.00E+01, 3.00E+00)(4.00E+01, 4.00E+00)(4.00E+01, 5.00E+00)(5.00E+01, 1.00E+00)"
                                    "(5.00E+01, 2.00E+00)(5.00E+01, 3.00E+00)(5.00E+01, 4.00E+00)(5.00E+01, 5.00E+00)"
                                    "(6.00E+01, 1.00E+00)(6.00E+01, 2.00E+00)(6.00E+01, 3.00E+00)(6.00E+01, 4.00E+00)"
                                    "(6.00E+01, 5.00E+00)\n")
        self.assertEqual(truth, output.fmt_output(self.exp, "{callpath},{metric}:{points:sep:''}"))

    def test_point_formatting3(self):
        truth = self._make_expected("2.00E+01, 1.00E+00;2.00E+01, 2.00E+00;2.00E+01, 3.00E+00;2.00E+01, 4.00E+00;"
                                    "2.00E+01, 5.00E+00;3.00E+01, 1.00E+00;3.00E+01, 2.00E+00;3.00E+01, 3.00E+00;"
                                    "3.00E+01, 4.00E+00;3.00E+01, 5.00E+00;4.00E+01, 1.00E+00;4.00E+01, 2.00E+00;"
                                    "4.00E+01, 3.00E+00;4.00E+01, 4.00E+00;4.00E+01, 5.00E+00;5.00E+01, 1.00E+00;"
                                    "5.00E+01, 2.00E+00;5.00E+01, 3.00E+00;5.00E+01, 4.00E+00;5.00E+01, 5.00E+00;"
                                    "6.00E+01, 1.00E+00;6.00E+01, 2.00E+00;6.00E+01, 3.00E+00;6.00E+01, 4.00E+00;"
                                    "6.00E+01, 5.00E+00\n")
        self.assertEqual(truth, output.fmt_output(self.exp, "{callpath},{metric}:{points:sep:';';format:'{point}'}"))

    def test_point_formatting4(self):
        truth = self._make_expected("2.00E+01, 1.00E+00;2.00E+01, 2.00E+00;2.00E+01, 3.00E+00;2.00E+01, 4.00E+00;"
                                    "2.00E+01, 5.00E+00;3.00E+01, 1.00E+00;3.00E+01, 2.00E+00;3.00E+01, 3.00E+00;"
                                    "3.00E+01, 4.00E+00;3.00E+01, 5.00E+00;4.00E+01, 1.00E+00;4.00E+01, 2.00E+00;"
                                    "4.00E+01, 3.00E+00;4.00E+01, 4.00E+00;4.00E+01, 5.00E+00;5.00E+01, 1.00E+00;"
                                    "5.00E+01, 2.00E+00;5.00E+01, 3.00E+00;5.00E+01, 4.00E+00;5.00E+01, 5.00E+00;"
                                    "6.00E+01, 1.00E+00;6.00E+01, 2.00E+00;6.00E+01, 3.00E+00;6.00E+01, 4.00E+00;"
                                    "6.00E+01, 5.00E+00\n")
        self.assertEqual(truth, output.fmt_output(self.exp, "{callpath},{metric}:{points:format:'{point}';sep:';'}"))

    def test_point_formatting5(self):
        truth = self._make_expected("2.00E+01'1.00E+00 | 2.00E+01'2.00E+00 | 2.00E+01'3.00E+00 | 2.00E+01'4.00E+00 | "
                                    "2.00E+01'5.00E+00 | 3.00E+01'1.00E+00 | 3.00E+01'2.00E+00 | 3.00E+01'3.00E+00 | "
                                    "3.00E+01'4.00E+00 | 3.00E+01'5.00E+00 | 4.00E+01'1.00E+00 | 4.00E+01'2.00E+00 | "
                                    "4.00E+01'3.00E+00 | 4.00E+01'4.00E+00 | 4.00E+01'5.00E+00 | 5.00E+01'1.00E+00 | "
                                    "5.00E+01'2.00E+00 | 5.00E+01'3.00E+00 | 5.00E+01'4.00E+00 | 5.00E+01'5.00E+00 | "
                                    "6.00E+01'1.00E+00 | 6.00E+01'2.00E+00 | 6.00E+01'3.00E+00 | 6.00E+01'4.00E+00 | "
                                    "6.00E+01'5.00E+00\n")
        self.assertEqual(truth, output.fmt_output(self.exp, r"{callpath},{metric}:{points:format:'{point:sep:'\''}'}"))

    def test_point_formatting6(self):
        truth = self._make_expected(
            "x20.000;y1.000 | x20.000;y2.000 | x20.000;y3.000 | x20.000;y4.000 | x20.000;y5.000 | "
            "x30.000;y1.000 | x30.000;y2.000 | x30.000;y3.000 | x30.000;y4.000 | x30.000;y5.000 | x40.000;y1.000 | "
            "x40.000;y2.000 | x40.000;y3.000 | x40.000;y4.000 | x40.000;y5.000 | x50.000;y1.000 | x50.000;y2.000 | "
            "x50.000;y3.000 | x50.000;y4.000 | x50.000;y5.000 | x60.000;y1.000 | x60.000;y2.000 | x60.000;y3.000 | "
            "x60.000;y4.000 | x60.000;y5.000\n")
        self.assertEqual(truth, output.fmt_output(self.exp, "{callpath},{metric}:{points:format:'{point:sep:';';"
                                                            "format:'{parameter}{coordinate:.3f}'}'}"))

    def test_measurement_formatting(self):
        truth = self._make_expected(
            "x20.000;y1.000:9.99E-01|x20.000;y2.000:8.11E+01|x20.000;y3.000:1.28E+02|x20.000;y4.000:1.62E+02|"
            "x20.000;y5.000:1.87E+02|x30.000;y1.000:1.00E+00|x30.000;y2.000:1.82E+02|x30.000;y3.000:2.84E+02|"
            "x30.000;y4.000:3.64E+02|x30.000;y5.000:4.21E+02|x40.000;y1.000:9.89E-01|x40.000;y2.000:3.18E+02|"
            "x40.000;y3.000:5.09E+02|x40.000;y4.000:6.43E+02|x40.000;y5.000:7.44E+02|x50.000;y1.000:1.00E+00|"
            "x50.000;y2.000:5.02E+02|x50.000;y3.000:7.95E+02|x50.000;y4.000:1.00E+03|x50.000;y5.000:1.17E+03|"
            "x60.000;y1.000:1.00E+00|x60.000;y2.000:7.26E+02|x60.000;y3.000:1.14E+03|x60.000;y4.000:1.44E+03|"
            "x60.000;y5.000:1.66E+03\n")

        self.assertEqual(truth, output.fmt_output(self.exp,
                                                  "{callpath},{metric}:"
                                                  "{measurements: format: "
                                                  "'{point:sep:';'; format:'{parameter}{coordinate:.3f}'}:{mean}';"
                                                  "sep:'|'}"))

    def test_brace_escape_parameter(self):
        self.assertEqual("{p}x {p}y\n{p}x {p}y\n{p}x {p}y\n{p}x {p}y\n",
                         output.fmt_output(self.exp, "{parameters: format:'{{p}}{parameter}'}"))
        self.assertEqual("{x} {y}\n{x} {y}\n{x} {y}\n{x} {y}\n",
                         output.fmt_output(self.exp, "{parameters: format:'{{{parameter}}}'}"))

    def test_brace_escape_measurement(self):
        self.assertEqual(
            ("{9.99E-01}{q},{8.11E+01}{q},{1.28E+02}{q},{1.62E+02}{q},{1.87E+02}{q},{1.00E+00}{q},{1.82E+02}{q},"
             "{2.84E+02}{q},{3.64E+02}{q},{4.21E+02}{q},{9.89E-01}{q},{3.18E+02}{q},{5.09E+02}{q},{6.43E+02}{q},"
             "{7.44E+02}{q},{1.00E+00}{q},{5.02E+02}{q},{7.95E+02}{q},{1.00E+03}{q},{1.17E+03}{q},{1.00E+00}{q},"
             "{7.26E+02}{q},{1.14E+03}{q},{1.44E+03}{q},{1.66E+03}{q} 312.47721968273464\n") * 4,
            output.fmt_output(self.exp, "{measurements: sep:',' format:'{{{mean}}}{{q}}'} {rss}"))
        self.assertEqual(
            ("{9.99E-01},{8.11E+01},{1.28E+02},{1.62E+02},{1.87E+02},{1.00E+00},{1.82E+02},{2.84E+02},{3.64E+02},"
             "{4.21E+02},{9.89E-01},{3.18E+02},{5.09E+02},{6.43E+02},{7.44E+02},{1.00E+00},{5.02E+02},{7.95E+02},"
             "{1.00E+03},{1.17E+03},{1.00E+00},{7.26E+02},{1.14E+03},{1.44E+03},{1.66E+03}\n") * 4,
            output.fmt_output(self.exp, "{measurements: sep:',' format:'{{{mean}}}'}"))

    def test_brace_escape(self):
        self.assertEqual("mergea\nsorta\nmergea\nsorta\n", output.fmt_output(self.exp, "{callpath}a"))
        self.assertEqual("merge{a}\nsort{a}\nmerge{a}\nsort{a}\n", output.fmt_output(self.exp, "{callpath}{{a}}"))
        self.assertEqual("{merge}\n{sort}\n{merge}\n{sort}\n", output.fmt_output(self.exp, "{{{callpath}}}"))

    def test_error_message(self):
        self.assertRaises(OutputFormatError,
                          output.fmt_output, self.exp, "{invalid_name}")
        self.assertRaises(OutputFormatError,
                          output.fmt_output, self.exp, "{{{callpath}}")
        self.assertRaises(OutputFormatError,
                          output.fmt_output, self.exp, "{callpath")

    def assertOutputRegex(self, regex, command, *args, **kwargs):
        temp_stdout = StringIO()
        with contextlib.redirect_stdout(temp_stdout):
            command(*args, **kwargs)
        output = temp_stdout.getvalue().strip()
        self.assertRegex(output, regex)

    def assertOutput(self, expected, command, *args, **kwargs):
        temp_stdout = StringIO()
        with contextlib.redirect_stdout(temp_stdout):
            command(*args, **kwargs)
        output = temp_stdout.getvalue().strip()
        self.assertEqual(expected, output)


if __name__ == '__main__':
    unittest.main()
