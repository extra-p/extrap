import math

from tests.modelling_testcase import TestCaseWithFunctionAssertions


class TestApproxAssertion(TestCaseWithFunctionAssertions):
    def test_assertApprox(self):
        self.assertRaises(AssertionError, self.assertApprox, 200, 200.001)
        self.assertApprox(200, 200.0001)
        self.assertApprox(200.0002, 200.0001)
        self.assertApprox(200.00049, 200.000)
        self.assertApprox(200.0000, 199.99951)
        self.assertApprox(200, 200.4999, places=3)
        self.assertRaises(AssertionError, self.assertApprox, 200.5, 200.49, places=5)
        self.assertRaises(AssertionError, self.assertApprox, 0.200, 0.200001)
        self.assertApprox(0.200, 0.2000001)
        self.assertApprox(0.2000002, 0.2000001)
        self.assertApprox(0.20000049, 0.2000000)
        self.assertApprox(0.2000000, 0.19999951)
        self.assertApprox(0.2005, 0.2004999, places=5)
        self.assertRaises(AssertionError, self.assertApprox, 0.2005, 0.20049, places=5)
        self.assertRaises(AssertionError, self.assertApprox, 0.000200, 0.000201)
        self.assertApprox(0.000200, 0.0002001)
        self.assertApprox(0.000200, 0.0002000001)
        self.assertApprox(0.0002000002, 0.0002000001)
        self.assertApprox(0.00020000049, 0.0002000000)
        self.assertApprox(0.0002000000, 0.00019999951)
        self.assertApprox(0.0002005, 0.0002004999, places=5)
        self.assertApprox(200.0000, 200.0001)
        self.assertApprox(200.0002, 200.0001)
        self.assertApprox(200.00049, 200.0000)
        self.assertApprox(200.0000, 199.99951)
        self.assertApprox(200, 200.4999, places=3)
        was_successful = False
        try:
            self.assertApprox(200, 200.001)
            was_successful = True
        except self.failureException:
            ...
        self.assertFalse(was_successful)

        was_successful = False
        try:
            self.assertApprox(200.002, 200.001)
            was_successful = True
        except self.failureException:
            ...
        self.assertFalse(was_successful)

        self.assertApprox(1, 0.9999, places=3)
        self.assertApprox(0.999, 0.9991, places=3)
        self.assertApprox(0.004, 0.0044, places=3)
        self.assertApprox(1.004, 1.0044, places=3)
        was_successful = False
        try:
            self.assertApprox(0.004, 0.0049, places=3)
            was_successful = True
        except self.failureException:
            ...
        self.assertFalse(was_successful)

        was_successful = False
        try:
            self.assertApprox(1.000, 1.0009, places=3)
            was_successful = True
        except self.failureException:
            ...
        self.assertFalse(was_successful)

        was_successful = False
        try:
            self.assertApprox(1.000, 0.999, places=3)
            was_successful = True
        except self.failureException:
            ...
        self.assertFalse(was_successful)

        self.assertApprox(math.nan, math.nan)
        self.assertApprox(math.inf, math.inf)
        self.assertApprox(-math.inf, -math.inf)
