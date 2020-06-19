import unittest
from fileio.text_file_reader import read_text_file
from entities.callpath import Callpath
from entities.parameter import Parameter
from entities.coordinate import Coordinate


class Test_TestOneParameterFiles(unittest.TestCase):

    def test_read_1(self):
        experiment = read_text_file("data/text/one_parameter_1.txt")
        self.assertEqual(len(experiment.parameters), 1)
        self.assertListEqual(experiment.parameters, [Parameter('x')])

    def test_read_2(self):
        experiment = read_text_file("data/text/one_parameter_2.txt")
        self.assertEqual(len(experiment.parameters), 1)

    def test_read_3(self):
        experiment = read_text_file("data/text/one_parameter_3.txt")
        self.assertEqual(len(experiment.parameters), 1)

    def test_read_4(self):
        experiment = read_text_file("data/text/one_parameter_4.txt")
        self.assertEqual(len(experiment.parameters), 1)

    def test_read_5(self):
        experiment = read_text_file("data/text/one_parameter_5.txt")
        self.assertEqual(len(experiment.parameters), 1)

    def test_read_6(self):
        experiment = read_text_file("data/text/one_parameter_6.txt")
        self.assertEqual(len(experiment.parameters), 1)
        self.assertListEqual(experiment.callpaths, [
                             Callpath('met1'), Callpath('met2'), Callpath('met3'), Callpath('met4')])
        p = Parameter('p')
        self.assertListEqual(experiment.parameters, [p])
        self.assertListEqual(experiment.coordinates, [
            Coordinate([(p, 1000)]),
            Coordinate([(p, 2000)]),
            Coordinate([(p, 4000)]),
            Coordinate([(p, 8000)]),
            Coordinate([(p, 16000)])
        ])

    def test_read_7(self):
        experiment = read_text_file("data/text/one_parameter_7.txt")
        self.assertEqual(len(experiment.parameters), 1)
        self.assertListEqual(experiment.callpaths, [Callpath('met1')])
        p = Parameter('p')
        self.assertListEqual(experiment.parameters, [p])
        self.assertListEqual(experiment.coordinates, [
            Coordinate([(p, 1000)]),
            Coordinate([(p, 2000)]),
            Coordinate([(p, 4000)]),
            Coordinate([(p, 8000)]),
            Coordinate([(p, 16000)])
        ])


class Test_TestTwoParameterFiles(unittest.TestCase):

    def test_read_1(self):
        experiment = read_text_file("data/text/two_parameter_1.txt")
        self.assertEqual(len(experiment.parameters), 2)

    def test_read_2(self):
        experiment = read_text_file("data/text/two_parameter_2.txt")
        self.assertEqual(len(experiment.parameters), 2)

    def test_read_3(self):
        experiment = read_text_file("data/text/two_parameter_3.txt")
        self.assertEqual(len(experiment.parameters), 2)
        self.assertListEqual(experiment.parameters, [
                             Parameter('x'), Parameter('y')])
        self.assertListEqual(experiment.coordinates, [
            Coordinate([(Parameter('x'), 20),
                        (Parameter('y'), 1)]),
            Coordinate([(Parameter('x'), 20),
                        (Parameter('y'), 2)]),
            Coordinate([(Parameter('x'), 20),
                        (Parameter('y'), 3)]),
            Coordinate([(Parameter('x'), 20),
                        (Parameter('y'), 4)]),
            Coordinate([(Parameter('x'), 20),
                        (Parameter('y'), 5)]),
            #
            Coordinate([(Parameter('x'), 30),
                        (Parameter('y'), 1)]),
            Coordinate([(Parameter('x'), 30),
                        (Parameter('y'), 2)]),
            Coordinate([(Parameter('x'), 30),
                        (Parameter('y'), 3)]),
            Coordinate([(Parameter('x'), 30),
                        (Parameter('y'), 4)]),
            Coordinate([(Parameter('x'), 30),
                        (Parameter('y'), 5)]),

            Coordinate([(Parameter('x'), 40),
                        (Parameter('y'), 1)]),
            Coordinate([(Parameter('x'), 40),
                        (Parameter('y'), 2)]),
            Coordinate([(Parameter('x'), 40),
                        (Parameter('y'), 3)]),
            Coordinate([(Parameter('x'), 40),
                        (Parameter('y'), 4)]),
            Coordinate([(Parameter('x'), 40),
                        (Parameter('y'), 5)]),

            Coordinate([(Parameter('x'), 50),
                        (Parameter('y'), 1)]),
            Coordinate([(Parameter('x'), 50),
                        (Parameter('y'), 2)]),
            Coordinate([(Parameter('x'), 50),
                        (Parameter('y'), 3)]),
            Coordinate([(Parameter('x'), 50),
                        (Parameter('y'), 4)]),
            Coordinate([(Parameter('x'), 50),
                        (Parameter('y'), 5)]),

            Coordinate([(Parameter('x'), 60),
                        (Parameter('y'), 1)]),
            Coordinate([(Parameter('x'), 60),
                        (Parameter('y'), 2)]),
            Coordinate([(Parameter('x'), 60),
                        (Parameter('y'), 3)]),
            Coordinate([(Parameter('x'), 60),
                        (Parameter('y'), 4)]),
            Coordinate([(Parameter('x'), 60),
                        (Parameter('y'), 5)])
        ])

    def test_read_4(self):
        experiment = read_text_file("data/text/two_parameter_4.txt")
        self.assertEqual(len(experiment.parameters), 2)
        self.assertListEqual(experiment.callpaths, [Callpath(
            'met1'), Callpath('met2'), Callpath('met3'), Callpath('met4')])
        self.assertListEqual(experiment.parameters, [
                             Parameter('p'), Parameter('q')])
        self.assertListEqual(experiment.coordinates, [
                             Coordinate([(Parameter('p'), 1000),
                                         (Parameter('q'), 10)]),
                             Coordinate([(Parameter('p'), 2000),
                                         (Parameter('q'), 20)]),
                             Coordinate([(Parameter('p'), 4000),
                                         (Parameter('q'), 40)]),
                             Coordinate([(Parameter('p'), 8000),
                                         (Parameter('q'), 80)]),
                             Coordinate([(Parameter('p'), 16000),
                                         (Parameter('q'), 160)])])


class Test_TestThreeParameterFiles(unittest.TestCase):

    def test_read_1(self):
        experiment = read_text_file("data/text/three_parameter_1.txt")
        self.assertEqual(len(experiment.parameters), 3)

    def test_read_2(self):
        experiment = read_text_file("data/text/three_parameter_2.txt")
        self.assertEqual(len(experiment.parameters), 3)

    def test_read_3(self):
        experiment = read_text_file("data/text/three_parameter_3.txt")
        self.assertEqual(len(experiment.parameters), 3)
        self.assertListEqual(experiment.parameters, [
                             Parameter('x'), Parameter('y'), Parameter('z')])
