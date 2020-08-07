import unittest

from extrap import extrap


class TestConsole(unittest.TestCase):
    def test_nothing(self):
        self.assertRaisesRegex(SystemExit, '[^0]', extrap.main)

    def test_help(self):
        self.assertRaisesRegex(SystemExit, '0', extrap.main, ['--help'])

    def test_help_modeler_options(self):
        self.assertRaisesRegex(SystemExit, '0', extrap.main, ['--help-options', 'Default'])

    def test_text(self):
        extrap.main(['--text', 'data/text/one_parameter_1.txt'])
        self.assertRaisesRegex(SystemExit, '[^0]', extrap.main, ['--text', 'data/text/does_not_exist.txt'])
        self.assertRaises(Exception, extrap.main, ['--text', 'data/talpas/talpas_1.txt'])

    def test_modeler(self):
        extrap.main(['--modeler', 'Default', '--text', 'data/text/one_parameter_1.txt'])
        self.assertRaisesRegex(SystemExit, '[^0]', extrap.main,
                               ['--modeler', 'does_not_exist', '--text', 'data/text/one_parameter_1.txt'])


if __name__ == '__main__':
    unittest.main()
