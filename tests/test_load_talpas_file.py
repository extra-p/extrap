import unittest
from fileio.talpas_file_reader import read_talpas_file
from entities.callpath import Callpath
from entities.parameter import Parameter
from entities.coordinate import Coordinate
from entities.metric import Metric
import timeit


class Test_TestFiles(unittest.TestCase):

    def test_read_1(self):
        experiment = read_talpas_file("data/talpas/talpas_1.txt")
        x = Parameter('x')
        self.assertListEqual(experiment.parameters, [x])
        self.assertListEqual(experiment.coordinates, [
            Coordinate([(x, 20)]),
            Coordinate([(x, 30)]),
            Coordinate([(x, 40)]),
            Coordinate([(x, 50)]),
            Coordinate([(x, 60)])
        ])
        self.assertListEqual(experiment.metrics, [
            Metric('time')
        ])
        self.assertListEqual(experiment.callpaths, [
            Callpath('compute')
        ])

    def test_read_2(self):
        experiment = read_talpas_file("data/talpas/talpas_2.txt")
        x, y = Parameter('x'), Parameter('y')
        self.assertListEqual(experiment.parameters, [x, y])

    def test_read_3(self):
        experiment = read_talpas_file("data/talpas/talpas_3.txt")

    def test_read_4(self):
        experiment = read_talpas_file("data/talpas/talpas_4.txt")

    def test_read_5(self):
        experiment = read_talpas_file("data/talpas/talpas_5.txt")

    def test_read_6(self):
        experiment = read_talpas_file("data/talpas/talpas_6.txt")

    def test_read_7(self):
        experiment = read_talpas_file("data/talpas/talpas_7.txt")

    def test_read_8(self):
        experiment = read_talpas_file("data/talpas/talpas_8.txt")

    def test_read_9(self):
        experiment = read_talpas_file("data/talpas/talpas_9.txt")

    def test_read_10(self):
        experiment = read_talpas_file("data/talpas/talpas_10.txt")
