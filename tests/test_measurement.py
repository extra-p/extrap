import unittest

from numpy import ma
from numpy.testing import assert_array_equal

from extrap.entities.coordinate import Coordinate
from extrap.entities.measurement import Measurement


class TestMeasurement(unittest.TestCase):
    def test_basic(self):
        c = Coordinate(1, 2, 3)
        m = Measurement(c, "test", "metric", 1)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)
        self.assertEqual(None, m.values)
        self.assertEqual(1, m.mean)
        self.assertEqual(1, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(1, m.maximum)
        self.assertEqual(0, m.std)
        self.assertEqual(1, m.repetitions)

        m = Measurement(c, "test", "metric", 1, keep_values=True)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)
        self.assertEqual(1, m.values)
        self.assertEqual(1, m.mean)
        self.assertEqual(1, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(1, m.maximum)
        self.assertEqual(0, m.std)
        self.assertEqual(1, m.repetitions)

    def test_basic_list(self):
        c = Coordinate(1, 2, 3)
        m = Measurement(c, "test", "metric", [1, 1, 1, 1, 1])
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)
        self.assertEqual(None, m.values)
        self.assertEqual(1, m.mean)
        self.assertEqual(1, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(1, m.maximum)
        self.assertEqual(0, m.std)
        self.assertEqual(5, m.repetitions)

        m = Measurement(c, "test", "metric", [1, 1, 1, 1, 1], keep_values=True)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)
        assert_array_equal([1, 1, 1, 1, 1], m.values)
        self.assertEqual(1, m.mean)
        self.assertEqual(1, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(1, m.maximum)
        self.assertEqual(0, m.std)
        self.assertEqual(5, m.repetitions)

    def test_list(self):
        c = Coordinate(1, 2, 3)
        values = [1, 1, 1, 2, 2]
        m = Measurement(c, "test", "metric", values)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)
        self.assertEqual(None, m.values)
        self.assertEqual(1.4, m.mean)
        self.assertEqual(1, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(2, m.maximum)
        self.assertAlmostEqual(0.48989, m.std, places=3)
        self.assertEqual(5, m.repetitions)

        m = Measurement(c, "test", "metric", values, keep_values=True)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)
        assert_array_equal(values, m.values)
        self.assertEqual(1.4, m.mean)
        self.assertEqual(1, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(2, m.maximum)
        self.assertAlmostEqual(0.48989, m.std, places=3)
        self.assertEqual(5, m.repetitions)

    def test_nested_list(self):
        c = Coordinate(1, 2, 3)
        values = [[1, 1, 1], [2, 2, 2]]
        m = Measurement(c, "test", "metric", values)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)
        self.assertEqual(None, m.values)
        self.assertEqual(1.5, m.mean)
        self.assertEqual(1.5, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(2, m.maximum)
        self.assertAlmostEqual(0.5, m.std, places=3)
        self.assertEqual(2, m.repetitions)

        m = Measurement(c, "test", "metric", values, keep_values=True)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)
        assert_array_equal(values, m.values)
        self.assertEqual(1.5, m.mean)
        self.assertEqual(1.5, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(2, m.maximum)
        self.assertAlmostEqual(0.5, m.std, places=3)
        self.assertEqual(2, m.repetitions)

    def test_nested_list2(self):
        c = Coordinate(1, 2, 3)
        values = [[1, 1, 1], [2, 2]]
        m = Measurement(c, "test", "metric", values)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)
        self.assertEqual(None, m.values)
        self.assertEqual(1.4, m.mean)
        self.assertEqual(1, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(2, m.maximum)
        self.assertAlmostEqual(0.48989, m.std, places=3)
        self.assertEqual(2, m.repetitions)

        m = Measurement(c, "test", "metric", values, keep_values=True)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)

        expected_array = ma.array([[1, 1, 1], [2, 2, 0]], mask=[[1, 1, 1], [1, 1, 0]])
        assert_array_equal(expected_array, m.values)
        self.assertEqual(1.4, m.mean)
        self.assertEqual(1, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(2, m.maximum)
        self.assertAlmostEqual(0.48989, m.std, places=3)
        self.assertEqual(2, m.repetitions)

    def test_nested_list3(self):
        c = Coordinate(1, 2, 3)
        values = [[[1, 1, 1], [1, 1, 1]], [[2, 2], [2, 2], [2, 2]]]
        m = Measurement(c, "test", "metric", values)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)
        self.assertEqual(None, m.values)
        self.assertEqual(1.5, m.mean)
        self.assertEqual(1.5, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(2, m.maximum)
        self.assertAlmostEqual(0.5, m.std, places=3)
        self.assertEqual(2, m.repetitions)

        m = Measurement(c, "test", "metric", values, keep_values=True)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)

        expected_array = ma.array([[[1, 1, 1], [1, 1, 1], [0, 0, 0]], [[2, 2, 0], [2, 2, 0], [2, 2, 0]]],
                                  mask=[[[1, 1, 1], [1, 1, 1], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [1, 1, 0]]])
        assert_array_equal(expected_array, m.values)
        self.assertEqual(1.5, m.mean)
        self.assertEqual(1.5, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(2, m.maximum)
        self.assertAlmostEqual(0.5, m.std, places=3)
        self.assertEqual(2, m.repetitions)

    def test_add_repetition_nested_list3(self):
        c = Coordinate(1, 2, 3)
        values = [[[1, 1, 1], [1, 1, 1]], [[2, 2], [2, 2], [2, 2]]]
        m = Measurement(c, "test", "metric", values, keep_values=True)
        self.assertEqual(2, m.repetitions)

        m.add_repetition(3)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)

        expected_array = ma.array([[[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                                   [[2, 2, 0], [2, 2, 0], [2, 2, 0]],
                                   [[3, 3, 3], [3, 3, 3], [0, 0, 0]]],
                                  mask=[[[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                                        [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
                                        [[1, 1, 1], [1, 1, 1], [0, 0, 0]]])
        assert_array_equal(expected_array, m.values)
        self.assertEqual(2, m.mean)
        self.assertEqual(2, m.median)
        self.assertEqual(1, m.minimum)
        self.assertEqual(3, m.maximum)
        self.assertAlmostEqual(0.81649, m.std, places=3)
        self.assertEqual(3, m.repetitions)

        values = [[[1, 1, 1], [1, 1, 1], [1]], [[2, 2], [2, 2]]]
        m = Measurement(c, "test", "metric", values, keep_values=True)
        self.assertEqual(2, m.repetitions)

        m.add_repetition(3)
        self.assertEqual(c, m.coordinate)
        self.assertEqual('test', m.callpath)
        self.assertEqual('metric', m.metric)

        expected_array = ma.array([[[1, 1, 1], [1, 1, 1], [1, 0, 0]],
                                   [[2, 2, 0], [2, 2, 0], [0, 0, 0]],
                                   [[3, 3, 3], [3, 3, 3], [0, 0, 0]]],
                                  mask=[[[1, 1, 1], [1, 1, 1], [1, 0, 0]],
                                        [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                                        [[1, 1, 1], [1, 1, 0], [0, 0, 0]]])
        assert_array_equal(expected_array, m.values)
        self.assertEqual(3, m.repetitions)
