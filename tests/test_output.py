import unittest
import contextlib
from io import StringIO

from extrap.extrap import extrapcmd as extrap


class TestOutput(unittest.TestCase):
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
        with self.assertRaises(ValueError):
            extrap.main(['--print', '{metric}, {a}', '--text', 'data/text/two_parameter_3.txt'])

    def assertOutputRegex(self, regex, command, *args, **kwargs):
        temp_stdout = StringIO()
        with contextlib.redirect_stdout(temp_stdout):
            command(*args, **kwargs)
        output = temp_stdout.getvalue().strip()
        self.assertRegex(output, regex)


if __name__ == '__main__':
    unittest.main()
