import unittest
from random import shuffle

from entities.callpath import Callpath
from entities.coordinate import Coordinate
from entities.metric import Metric
from fileio.jsonlines_file_reader import read_jsonlines_file
from fileio.text_file_reader import read_text_file
from modelers.model_generator import ModelGenerator
from modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler


class TestFindFirstMeasurements(unittest.TestCase):

    def test_2parameters_basic(self):
        experiment = read_text_file('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]

        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 2)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate((20,)),
            Coordinate((30,)),
            Coordinate((40,)),
            Coordinate((50,)),
            Coordinate((60,))
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate((1,)),
            Coordinate((2,)),
            Coordinate((3,)),
            Coordinate((4,)),
            Coordinate((5,))
        ])

    def test_2parameters_reversed(self):
        experiment = read_text_file('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        measurements = list(reversed(measurements))

        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 2)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate((60,)),
            Coordinate((50,)),
            Coordinate((40,)),
            Coordinate((30,)),
            Coordinate((20,))
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate((5,)),
            Coordinate((4,)),
            Coordinate((3,)),
            Coordinate((2,)),
            Coordinate((1,))
        ])

    def test_2parameters_random(self):
        experiment = read_text_file('data/text/two_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        for _ in range(len(measurements)):
            shuffle(measurements)

            f_msm = modeler.find_first_measurement_points(measurements)

            self.assertEqual(len(f_msm), 2)
            self.assertSetEqual(set(m.coordinate for m in f_msm[0]),
                                {Coordinate(20), Coordinate(30), Coordinate(40), Coordinate(50), Coordinate(60)})
            self.assertSetEqual(set(m.coordinate for m in f_msm[1]),
                                {Coordinate(1), Coordinate(2), Coordinate(3), Coordinate(4), Coordinate(5)})

    def test_3parameters_basic(self):
        experiment = read_text_file('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate(20),
            Coordinate(30),
            Coordinate(40),
            Coordinate(50),
            Coordinate(60)
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate(1),
            Coordinate(2),
            Coordinate(3),
            Coordinate(4),
            Coordinate(5)
        ])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate(100),
            Coordinate(200),
            Coordinate(300),
            Coordinate(400),
            Coordinate(500)
        ])

    def test_3parameters_reversed(self):
        experiment = read_text_file('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        measurements = list(reversed(measurements))

        f_msm = modeler.find_first_measurement_points(measurements)

        self.assertEqual(len(f_msm), 3)
        self.assertListEqual([m.coordinate for m in f_msm[0]], [
            Coordinate(60),
            Coordinate(50),
            Coordinate(40),
            Coordinate(30),
            Coordinate(20)
        ])
        self.assertListEqual([m.coordinate for m in f_msm[1]], [
            Coordinate(5),
            Coordinate(4),
            Coordinate(3),
            Coordinate(2),
            Coordinate(1)
        ])
        self.assertListEqual([m.coordinate for m in f_msm[2]], [
            Coordinate(500),
            Coordinate(400),
            Coordinate(300),
            Coordinate(200),
            Coordinate(100)
        ])

    def test_3parameters_random(self):
        experiment = read_text_file('data/text/three_parameter_1.txt')

        modeler = MultiParameterModeler()
        measurements = experiment.measurements[(Callpath('reg'), Metric('metr'))]
        for _ in range(len(measurements)):
            shuffle(measurements)

            f_msm = modeler.find_first_measurement_points(measurements)

            self.assertEqual(len(f_msm), 3)
            self.assertSetEqual(set(m.coordinate for m in f_msm[0]), {
                Coordinate(20),
                Coordinate(30),
                Coordinate(40),
                Coordinate(50),
                Coordinate(60)
            })
            self.assertSetEqual(set(m.coordinate for m in f_msm[1]), {
                Coordinate(1),
                Coordinate(2),
                Coordinate(3),
                Coordinate(4),
                Coordinate(5)
            })
            self.assertSetEqual(set(m.coordinate for m in f_msm[2]), {
                Coordinate(100),
                Coordinate(200),
                Coordinate(300),
                Coordinate(400),
                Coordinate(500)
            })


class TestSparseModeling(unittest.TestCase):
    def test_1(self):
        experiment = read_jsonlines_file('data/sparsemodeler/test1.jsonl')
        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        model_generator.model_all()

    def test_2(self):
        experiment = read_jsonlines_file('data/sparsemodeler/test2.jsonl')
        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        model_generator.model_all()

    def test_input_1(self):
        experiment = read_jsonlines_file('data/sparsemodeler/input_1.jsonl')
        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        model_generator.model_all()

    def test_complete_matrix_2p(self):
        experiment = read_jsonlines_file('data/sparsemodeler/complete_matrix_2p.jsonl')
        modeler = MultiParameterModeler()
        modeler.single_parameter_point_selection = 'all'
        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        model_generator.model_all()

    def test_cross_matrix_2p(self):
        experiment = read_jsonlines_file('data/sparsemodeler/cross_matrix_2p.jsonl')
        modeler = MultiParameterModeler()
        modeler.single_parameter_point_selection = 'all'
        # initialize model generator
        model_generator = ModelGenerator(experiment)
        # create models from data
        self.assertWarns(UserWarning, model_generator.model_all)
