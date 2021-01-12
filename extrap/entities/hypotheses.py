# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import abc
from typing import Sequence

import numpy
from marshmallow import fields

from extrap.entities.functions import Function, MultiParameterFunction, FunctionSchema
from extrap.entities.measurement import Measurement
from extrap.util.serialization_schema import BaseSchema, NumberField


class Hypothesis:
    def __init__(self, function: Function, use_median):
        """
        Initialize Hypothesis object.
        """
        self.function: Function = function
        self._RSS = 0
        self._rRSS = 0
        self._SMAPE = 0
        self._AR2 = 0
        self._RE = 0
        self._use_median = use_median
        self._costs_are_calculated = False

    @property
    def RSS(self):
        """
        Return the RSS.
        """
        if not self._costs_are_calculated:
            raise RuntimeError("Costs are not calculated.")
        return self._RSS

    @property
    def rRSS(self):
        """
        Return the rRSS.
        """
        if not self._costs_are_calculated:
            raise RuntimeError("Costs are not calculated.")
        return self._rRSS

    @property
    def AR2(self):
        """
        Return the AR2.
        """
        if not self._costs_are_calculated:
            raise RuntimeError("Costs are not calculated.")
        return self._AR2

    @property
    def SMAPE(self):
        """
        Return the SMAPE.
        """
        if not self._costs_are_calculated:
            raise RuntimeError("Costs are not calculated.")
        return self._SMAPE

    @property
    def RE(self):
        """
        Return the relative error.
        """
        if not self._costs_are_calculated:
            raise RuntimeError("Costs are not calculated.")
        return self._RE

    def compute_coefficients(self, measurements: Sequence[Measurement]):
        raise NotImplementedError()

    def compute_cost(self, measurements: Sequence[Measurement]):
        raise NotImplementedError()

    def is_valid(self):
        """
        Checks if there is a numeric imprecision. If this is the case the hypothesis will be ignored.
        """
        valid = not (self.RSS != self.RSS or abs(self.RSS) == float('inf'))
        return valid

    def clean_constant_coefficient(self, phi, training_measurements):
        """
        This function is used to correct numerical imprecision in the caculations,
        when the constant coefficient should be zero but is instead very small.
        We take into account the minimum data value to make sure that we don't "nullify"
        actually relevant numbers.
        """
        if self._use_median:
            minimum = min(m.median for m in training_measurements)
        else:
            minimum = min(m.mean for m in training_measurements)
        if minimum == 0:
            if abs(self.function.constant_coefficient - minimum) < phi:
                self.function.constant_coefficient = 0
        else:
            if abs(self.function.constant_coefficient / minimum) < phi:
                self.function.constant_coefficient = 0

    def calc_term_contribution(self, term, measurements: Sequence[Measurement]):
        """
        Calculates the term contribution of the term with the given term id to see if it is smaller than epsilon.
        """
        if self._use_median:
            actual = numpy.array([m.median for m in measurements])
        else:
            actual = numpy.array([m.mean for m in measurements])

        if measurements[0].coordinate.dimensions > 1:
            points = numpy.array([m.coordinate.as_tuple() for m in measurements]).T
        else:
            points = numpy.array([m.coordinate[0] for m in measurements])

        contribution = numpy.abs(term.evaluate(points) / actual)
        maximum_term_contribution = contribution.max()
        return maximum_term_contribution

    def __repr__(self):
        return f"Hypothesis({self.function}, RSS:{self._RSS:5f}, SMAPE:{self._SMAPE:5f})"

    def __eq__(self, other):
        if not isinstance(other, Hypothesis):
            return NotImplemented
        elif self is other:
            return True
        else:
            return self.__dict__ == other.__dict__


MAX_HYPOTHESIS = Hypothesis(Function(), False)
MAX_HYPOTHESIS._RSS = float('inf')
MAX_HYPOTHESIS._rRSS = float('inf')
MAX_HYPOTHESIS._SMAPE = float('inf')
MAX_HYPOTHESIS._AR2 = float('inf')
MAX_HYPOTHESIS._RE = float('inf')
MAX_HYPOTHESIS._costs_are_calculated = True


class ConstantHypothesis(Hypothesis):
    """
    This class represents a constant hypothesis, it is used to represent a performance
    function that is not affected by the input value of a parameter. The modeler calls this
    class first to see if there is a constant model that describes the data best.
    """

    def __init__(self, function, use_median):
        """
        Initialize the ConstantHypothesis.
        """
        super().__init__(function, use_median)

    # TODO: should this be calculated?
    @property
    def AR2(self):
        return 1

    def compute_coefficients(self, measurements: Sequence[Measurement]):
        """
        Computes the constant_coefficients of the function using the mean.
        """
        if self._use_median:
            self.function.constant_coefficient = numpy.mean([m.median for m in measurements])
        else:
            self.function.constant_coefficient = numpy.mean([m.mean for m in measurements])

    def compute_cost(self, measurements: Sequence[Measurement]):
        """
        Computes the cost of the constant hypothesis using all data points.
        """
        smape = 0
        for measurement in measurements:
            predicted = self.function.constant_coefficient
            if self._use_median:
                actual = measurement.median
            else:
                actual = measurement.mean

            difference = predicted - actual
            self._RSS += difference * difference
            if actual != 0:
                relative_difference = difference / actual
                self._rRSS += relative_difference * relative_difference

                absolute_error = numpy.abs(difference)
                relative_error = absolute_error / actual
                self._RE = numpy.mean(relative_error)

            abssum = abs(actual) + abs(predicted)
            if abssum != 0:
                smape += abs(difference) / abssum * 2

        self._SMAPE = smape / len(measurements) * 100
        self._costs_are_calculated = True


class SingleParameterHypothesis(Hypothesis):
    """
    This class represents a single parameter hypothesis, it is used to represent
    a performance function for one parameter. The modeler calls many of these objects
    to find the best model that fits the data.
    """

    def __init__(self, function, use_median):
        """
        Initialize SingleParameterHypothesis object.
        """
        super().__init__(function, use_median)

    def compute_cost_leave_one_out(self, training_measurements: Sequence[Measurement],
                                   validation_measurement: Measurement):
        """
        Compute the cost for the single-parameter model using leave one out crossvalidation.
        """
        value = validation_measurement.coordinate[0]
        predicted = self.function.evaluate(value)
        actual = validation_measurement.value(self._use_median)

        difference = predicted - actual
        self._RSS += difference * difference
        if actual != 0:
            relative_difference = difference / actual
            self._RE += numpy.abs(relative_difference) / (len(training_measurements) + 1)
            self._rRSS += relative_difference * relative_difference
        abssum = abs(actual) + abs(predicted)
        if abssum != 0:
            self._SMAPE += (abs(difference) / abssum * 2) / \
                           len(training_measurements) * 100
        self._costs_are_calculated = True

    def compute_cost(self, measurements: Sequence[Measurement]):
        points = numpy.array([m.coordinate[0] for m in measurements])
        predicted = self.function.evaluate(points)

        if self._use_median:
            actual = numpy.array([m.median for m in measurements])
        else:
            actual = numpy.array([m.mean for m in measurements])

        difference = predicted - actual
        self._RSS = numpy.sum(difference * difference)

        relativeDifference = difference / actual
        self._rRSS = numpy.sum(relativeDifference * relativeDifference)

        absolute_error = numpy.abs(difference)
        relative_error = absolute_error / actual
        self._RE = numpy.mean(relative_error)

        abssum = numpy.abs(actual) + numpy.abs(predicted)
        # This condition prevents a division by zero, but it is correct: if sum is 0, both `actual` and `predicted`
        # must have been 0, and in that case the error at this point is 0, so we don't need to add anything.
        smape = numpy.abs(difference[abssum != 0.0]) / abssum[abssum != 0.0] * 2
        self._SMAPE = numpy.mean(smape) * 100

        self._costs_are_calculated = True

    def compute_adjusted_rsquared(self, TSS, measurements):
        """
        Compute the adjusted R^2 for the hypothesis.
        """
        adjR = 1.0 - (self._RSS / TSS)
        degrees_freedom = len(measurements) - len(self.function.compound_terms) - 1
        self._AR2 = (1.0 - (1.0 - adjR) *
                     (len(measurements) - 1.0) / degrees_freedom)

    def compute_coefficients(self, measurements: Sequence[Measurement]):
        """
        Computes the coefficients of the function using the least squares solution.
        """
        if self._use_median:
            b_list = numpy.array([m.median for m in measurements])
        else:
            b_list = numpy.array([m.mean for m in measurements])
        points = numpy.array([m.coordinate[0] for m in measurements])

        a_list = [numpy.ones((1, len(points)))]
        for compound_term in self.function.compound_terms:
            compound_term.coefficient = 1
            compound_term_value = compound_term.evaluate(points)
            a_list.append(compound_term_value.reshape(1, -1))

        # solving the lgs for X to get the coefficients
        A = numpy.concatenate(a_list, axis=0).T
        B = b_list

        X, _, _, _ = numpy.linalg.lstsq(A, B, None)
        # logging.debug("Coefficients:"+str(X))

        # setting the coefficients for the hypothesis
        self.function.constant_coefficient = X[0]
        for i, compound_term in enumerate(self.function.compound_terms):
            compound_term.coefficient = X[i + 1]


class MultiParameterHypothesis(Hypothesis):
    """
    This class represents a multi parameter hypothesis, it is used to represent
    a performance function with several parameters. However, it can have also
    only one parameter. The modeler calls many of these objects to find the best
    model that fits the data.
    """

    function: MultiParameterFunction

    def __init__(self, function: MultiParameterFunction, use_median):
        """
        Initialize MultiParameterHypothesis object.
        """
        super().__init__(function, use_median)

    def compute_cost(self, measurements):
        """
        Compute the cost for a multi parameter hypothesis.
        """
        self._RSS = 0
        self._rRSS = 0
        smape = 0
        re_sum = 0

        for measurement in measurements:
            coordinate = measurement.coordinate
            parameter_value_pairs = {}
            for parameter, value in enumerate(coordinate):
                parameter_value_pairs[parameter] = float(value)

            predicted = self.function.evaluate(parameter_value_pairs)
            # print(predicted)

            actual = measurement.value(self._use_median)

            # print(actual)

            difference = predicted - actual
            # absolute_difference = abs(difference)
            abssum = abs(actual) + abs(predicted)

            # calculate relative error
            absolute_error = abs(predicted - actual)
            relative_error = absolute_error / actual
            re_sum = re_sum + relative_error

            self._RSS += difference * difference

            relativeDifference = difference / actual
            self._rRSS += relativeDifference * relativeDifference

            if abssum != 0.0:
                # This `if` condition prevents a division by zero, but it is correct: if sum is 0,
                # both `actual` and `predicted` must have been 0, and in that case the error at this point is 0,
                # so we don't need to add anything.
                smape += abs(difference) / abssum * 2

        # times 100 for percentage error
        self._RE = re_sum / len(measurements)
        self._SMAPE = smape / len(measurements) * 100
        self._costs_are_calculated = True

    def compute_adjusted_rsquared(self, TSS, measurements):
        """
        Compute the adjusted R^2 for the hypothesis.
        """
        self._AR2 = 0.0
        adjR = 1.0 - (self.RSS / TSS)
        counter = 0

        for multi_parameter_term in self.function:
            counter += len(multi_parameter_term.parameter_term_pairs)

        degrees_freedom = len(measurements) - counter - 1
        self._AR2 = (1.0 - (1.0 - adjR) * (len(measurements) - 1.0) / degrees_freedom)

    def compute_coefficients(self, measurements):
        """
        Computes the coefficients of the function using the least squares solution.
        """
        # creating a numpy matrix representation of the lgs
        a_list = []
        b_list = []

        self.function.reset_coefficients()

        for measurement in measurements:
            value = measurement.value(self._use_median)
            list_element = [1]  # 1 for constant coefficient
            for multi_parameter_term in self.function:
                coordinate = measurement.coordinate
                multi_parameter_term_value = multi_parameter_term.evaluate(coordinate)
                list_element.append(multi_parameter_term_value)
            a_list.append(list_element)
            b_list.append(value)
            # print(str(list_element)+"[x]=["+str(value)+"]")
            # logging.debug(str(list_element)+"[x]=["+str(value)+"]")

        # solving the lgs for coeffs to get the coefficients
        A = numpy.array(a_list)
        B = numpy.array(b_list)
        try:
            coeffs, _, _, _ = numpy.linalg.lstsq(A, B, None)
        except numpy.linalg.LinAlgError as e:
            # sometimes first try does not work
            coeffs, _, _, _ = numpy.linalg.lstsq(A, B, None)

        # print("Coefficients:"+str(coeffs))
        # logging.debug("Coefficients:"+str(coeffs[0]))

        # setting the coefficients for the hypothesis
        self.function.constant_coefficient = coeffs[0]
        for multi_parameter_term, coeff in zip(self.function, coeffs[1:]):
            multi_parameter_term.coefficient = coeff


class HypothesisSchema(BaseSchema):
    function = fields.Nested(FunctionSchema)
    _RSS = NumberField(data_key='RSS')
    _rRSS = NumberField(data_key='rRSS')
    _SMAPE = NumberField(data_key='SMAPE')
    _AR2 = NumberField(data_key='AR2')
    _RE = NumberField(data_key='RE')
    _use_median = fields.Bool()
    _costs_are_calculated = fields.Bool()


class DefaultHypothesisSchema(HypothesisSchema):
    def create_object(self):
        return Hypothesis(None, None)


class ConstantHypothesisSchema(HypothesisSchema):
    _AR2 = fields.Constant(1, data_key='AR2', load_only=True)

    def create_object(self):
        return ConstantHypothesis(None, None)


class SingleParameterHypothesisSchema(HypothesisSchema):
    def create_object(self):
        return SingleParameterHypothesis(None, None)


class MultiParameterHypothesisSchema(HypothesisSchema):
    def create_object(self):
        return MultiParameterHypothesis(None, None)
