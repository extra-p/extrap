import unittest

from extrap.fileio.file_reader.extra_prof import ExtraProf2Reader


class TestExtraProfFileLoader(unittest.TestCase):
    def test_load_basic(self):
        experiment = ExtraProf2Reader().read_experiment('data/extra_prof/test1/')


if __name__ == '__main__':
    unittest.main()
