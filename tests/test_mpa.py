import unittest

from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.mpa.util import identify_selection_mode


class TestMpaOneParam(unittest.TestCase):
    def test_add(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/1_param_add.txt')
        self.assertEqual('gpr', identify_selection_mode(experiment, 5))

    def test_gpr(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/1_param_gpr.txt')
        self.assertEqual('gpr', identify_selection_mode(experiment, 5))

    def test_base(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/1_param_base.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))

    def test_base2(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/1_param_base2.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))

    def test_base3(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/1_param_base3.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))

    def test_base4(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/1_param_base4.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))

    def test_base5(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/1_param_base5.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))

    def test_base6(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/1_param_base6.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))

    def test_base7(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/1_param_base7.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))

    def test_base8(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/1_param_base8.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))


class TestMpaTwoParam(unittest.TestCase):
    def test_add(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/2_param_add.txt')
        self.assertEqual('add', identify_selection_mode(experiment, 5))

    def test_gpr(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/2_param_gpr.txt')
        self.assertEqual('gpr', identify_selection_mode(experiment, 5))

    def test_base(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/2_param_base.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))

    def test_add2(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/2_param_add2.txt')
        self.assertEqual('add', identify_selection_mode(experiment, 5))

    def test_gpr2(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/2_param_gpr2.txt')
        self.assertEqual('gpr', identify_selection_mode(experiment, 5))

    def test_base2(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/2_param_base2.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))

    def test_base3(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/2_param_base3.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))


class TestMpaThreeParam(unittest.TestCase):
    def test_add(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/3_param_add.txt')
        self.assertEqual('add', identify_selection_mode(experiment, 5))

    def test_gpr(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/3_param_gpr.txt')
        self.assertEqual('gpr', identify_selection_mode(experiment, 5))

    def test_base(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/3_param_base.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))


class TestMpaFourParam(unittest.TestCase):
    def test_add(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/4_param_add.txt')
        self.assertEqual('add', identify_selection_mode(experiment, 5))

    def test_gpr(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/4_param_gpr.txt')
        self.assertEqual('gpr', identify_selection_mode(experiment, 5))

    def test_base(self):
        experiment = TextFileReader().read_experiment('data/text/mpa/4_param_base.txt')
        self.assertEqual('base', identify_selection_mode(experiment, 5))
