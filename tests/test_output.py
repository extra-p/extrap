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

    def assertOutputRegex(self, regex, command, *args, **kwargs):
        temp_stdout = StringIO()
        with contextlib.redirect_stdout(temp_stdout):
            command(*args, **kwargs)
        output = temp_stdout.getvalue().strip()
        self.assertRegex(output, regex)

if __name__ == '__main__':
    unittest.main()
