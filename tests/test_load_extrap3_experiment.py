import unittest

from fileio.extrap3_experiment_reader import read_extrap3_experiment


class TestLoadExtraP3Experiment(unittest.TestCase):
    def test_extrap3_experiment(self):
        experiment = read_extrap3_experiment('data/input/experiment_3')

    def test_extrap3_multiparameter_experiment(self):
        experiment = read_extrap3_experiment('data/input/experiment_3_mp')

    def test_sparse_experiment(self):
        experiment = read_extrap3_experiment('data/input/experiment_3_sparse')
        pass


if __name__ == '__main__':
    unittest.main()
