import unittest
from fileio.text_file_reader import read_text_file
from modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler
from entities.coordinate import Coordinate
from entities.hypotheses import SingleParameterHypothesis
from entities.experiment import Experiment
from entities.callpath import Callpath
from entities.metric import Metric
from random import shuffle


class TestFindFirstMeasurements(unittest.TestCase):

    def test_2parameters_basic(self):
        experiment = read_text_file('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]

        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 2)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate((20, 1)),
            Coordinate((30, 1)),
            Coordinate((40, 1)),
            Coordinate((50, 1)),
            Coordinate((60, 1))
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate((20, 1)),
            Coordinate((20, 2)),
            Coordinate((20, 3)),
            Coordinate((20, 4)),
            Coordinate((20, 5))
        ])

    def test_2parameters_reversed(self):
        experiment = read_text_file('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        measurements = list(reversed(measurements))

        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 2)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate((60, 1)),
            Coordinate((50, 1)),
            Coordinate((40, 1)),
            Coordinate((30, 1)),
            Coordinate((20, 1))
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate((20, 5)),
            Coordinate((20, 4)),
            Coordinate((20, 3)),
            Coordinate((20, 2)),
            Coordinate((20, 1))
        ])

    def test_2parameters_random(self):
        experiment = read_text_file('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        for _ in range(len(measurements)):
            shuffle(measurements)

            f_msm = modeler.find_first_measurement_points(measurements)

            self.assertEqual(len(f_msm), 2)
            self.assertSetEqual(set(m.coordinate for m in f_msm[0]), set([
                Coordinate((20, 1)),
                Coordinate((30, 1)),
                Coordinate((40, 1)),
                Coordinate((50, 1)),
                Coordinate((60, 1))
            ]))
            self.assertSetEqual(set(m.coordinate for m in f_msm[1]), set([
                Coordinate((20, 1)),
                Coordinate((20, 2)),
                Coordinate((20, 3)),
                Coordinate((20, 4)),
                Coordinate((20, 5))
            ]))

    def test_3parameters_basic(self):
        experiment = read_text_file('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate((20, 1, 100)),
            Coordinate((30, 1, 100)),
            Coordinate((40, 1, 100)),
            Coordinate((50, 1, 100)),
            Coordinate((60, 1, 100))
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate((20, 1, 100)),
            Coordinate((20, 2, 100)),
            Coordinate((20, 3, 100)),
            Coordinate((20, 4, 100)),
            Coordinate((20, 5, 100))
        ])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate((20, 1, 100)),
            Coordinate((20, 1, 200)),
            Coordinate((20, 1, 300)),
            Coordinate((20, 1, 400)),
            Coordinate((20, 1, 500))
        ])

    def test_3parameters_reversed(self):
        experiment = read_text_file('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        measurements = list(reversed(measurements))

        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate((60, 1, 100)),
            Coordinate((50, 1, 100)),
            Coordinate((40, 1, 100)),
            Coordinate((30, 1, 100)),
            Coordinate((20, 1, 100))
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate((20, 5, 100)),
            Coordinate((20, 4, 100)),
            Coordinate((20, 3, 100)),
            Coordinate((20, 2, 100)),
            Coordinate((20, 1, 100))
        ])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate((20, 1, 500)),
            Coordinate((20, 1, 400)),
            Coordinate((20, 1, 300)),
            Coordinate((20, 1, 200)),
            Coordinate((20, 1, 100))
        ])

    def test_3parameters_random(self):
        experiment = read_text_file('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        for _ in range(len(measurements)):
            shuffle(measurements)

            f_msm = modeler.find_first_measurement_points(measurements)

            self.assertEqual(len(f_msm), 3)
            self.assertSetEqual(set(m.coordinate for m in f_msm[0]), set([
                Coordinate((20, 1, 100)),
                Coordinate((30, 1, 100)),
                Coordinate((40, 1, 100)),
                Coordinate((50, 1, 100)),
                Coordinate((60, 1, 100))
            ]))
            self.assertSetEqual(set(m.coordinate for m in f_msm[1]), set([
                Coordinate((20, 1, 100)),
                Coordinate((20, 2, 100)),
                Coordinate((20, 3, 100)),
                Coordinate((20, 4, 100)),
                Coordinate((20, 5, 100))
            ]))
            self.assertSetEqual(set(m.coordinate for m in f_msm[2]), set([
                Coordinate((20, 1, 100)),
                Coordinate((20, 1, 200)),
                Coordinate((20, 1, 300)),
                Coordinate((20, 1, 400)),
                Coordinate((20, 1, 500))
            ]))
