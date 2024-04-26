# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
from abc import ABC
from typing import Iterable, TypeVar, Union, Tuple, Sequence

import numpy

from extrap.entities.functions import ConstantFunction
from extrap.entities.hypotheses import Hypothesis, SingleParameterHypothesis, MAX_HYPOTHESIS, ConstantHypothesis
from extrap.entities.measurement import Measurement, Measure
from extrap.entities.parameter import Parameter
from extrap.modelers.abstract_modeler import AbstractModeler
from extrap.modelers.modeler_options import modeler_options

_H = TypeVar('_H', bound=Hypothesis)
_SH = TypeVar('_SH', bound=SingleParameterHypothesis)
SH = Union[_SH, SingleParameterHypothesis]
H = Union[_H, Hypothesis]


@modeler_options
class AbstractSingleParameterModeler(AbstractModeler, ABC):
    CLEAN_CONSTANT_EPSILON = 1e-3  # minimum allowed value for a constant coefficient before it is set to 0

    allow_log_terms = modeler_options.add(True, bool, 'Allows models with logarithmic terms')
    use_crossvalidation = modeler_options.add(True, bool, 'Enables cross-validation', name='Cross-validation')
    compare_with_RSS = modeler_options.add(False, bool,
                                           'If enabled the models are compared using their residual sum of squares '
                                           '(RSS) instead of their symmetric mean absolute percentage error (SMAPE)')

    def __init__(self, use_measure: Union[bool, Measure]):
        super().__init__(use_measure)
        self.epsilon = 0.0005  # value for the minimum term contribution

    def compare_hypotheses(self, old: Hypothesis, new: SingleParameterHypothesis, measurements: Sequence[Measurement]):
        """
        Compares the best with the new hypothesis and decides which one is a better fit for the data.
        If the new hypothesis is better than the best one it becomes the best hypothesis.
        The choice is made based on the RSS or SMAPE.
        """
        if old == MAX_HYPOTHESIS:
            return True

        # get the compound terms of the new hypothesis
        compound_terms = new.function.compound_terms

        previous = numpy.seterr(divide='ignore', invalid='ignore')
        # for all compound terms check if they are smaller than minimum allowed contribution
        for term in compound_terms:
            # ignore this hypothesis, since one of the terms contributes less than epsilon to the function
            if term.coefficient == 0 or new.calc_term_contribution(term, measurements) < self.epsilon:
                return False
        numpy.seterr(**previous)

        # print smapes in debug mode
        logging.debug("next hypothesis SMAPE: %g RSS: %g", new.SMAPE, new.RSS)
        logging.debug("best hypothesis SMAPE: %g RSS: %g", old.SMAPE, old.RSS)
        if self.compare_with_RSS:
            return new.RSS < old.RSS
        return new.SMAPE < old.SMAPE

    def create_constant_model(self, measurements: Sequence[Measurement]) -> Tuple[ConstantHypothesis, float]:
        """
        Creates a constant model that fits the data using a ConstantFunction.
        """
        # compute the constant coefficient
        mean_model = numpy.mean(
            numpy.fromiter(Measurement.select_measure(measurements, self.use_measure), float, len(measurements)))

        # create a constant function
        constant_function = ConstantFunction(mean_model)
        constant_hypothesis = ConstantHypothesis(constant_function, self.use_measure)

        # compute cost of the constant model
        constant_hypothesis.compute_cost(measurements)
        constant_cost = constant_hypothesis.RSS

        return constant_hypothesis, constant_cost

    def find_best_hypothesis(self, candidate_hypotheses: Iterable[SH], constant_cost: float,
                             measurements: Sequence[Measurement], current_best: H = MAX_HYPOTHESIS) -> Union[SH, H]:
        """
        Searches for the best single parameter hypothesis and returns it.
        """

        # currently the constant hypothesis is the best hypothesis
        best_hypothesis = current_best

        # search for the best hypothesis over all functions that can be built with the basic building blocks

        for i, next_hypothesis in enumerate(candidate_hypotheses):

            if self.use_crossvalidation:
                # use leave one out cross validation
                # cycle through points and leave one out per iteration
                for element_id in range(len(measurements)):
                    # copy measurements to create the training sets
                    training_measurements = list(measurements)

                    # remove one element the set
                    training_measurements.pop(element_id)

                    # validation set
                    validation_measurement = measurements[element_id]

                    # compute the model coefficients based on the training data
                    next_hypothesis.compute_coefficients(training_measurements)

                    # check if the constant coefficient should actually be 0
                    next_hypothesis.clean_constant_coefficient(self.epsilon, training_measurements)

                    # compute the cost of the single-parameter model for the validation data
                    next_hypothesis.compute_cost_leave_one_out(training_measurements, validation_measurement)

                # compute the model coefficients using all data
                next_hypothesis.compute_coefficients(measurements)
                logging.debug("single-parameter model %i: %s", i, next_hypothesis.function)
            else:
                # compute the model coefficients based on the training data
                next_hypothesis.compute_coefficients(measurements)

                # check if the constant coefficient should actually be 0
                next_hypothesis.clean_constant_coefficient(
                    self.CLEAN_CONSTANT_EPSILON, measurements)

                # compute the cost of the single-parameter model for the validation data
                next_hypothesis.compute_cost(measurements)

            # compute the AR2 for the hypothesis
            next_hypothesis.compute_adjusted_rsquared(constant_cost, measurements)

            # check if hypothesis is valid
            if not next_hypothesis.is_valid():
                logging.info("Numeric imprecision found. Model is invalid and will be ignored.")

            # compare the new hypothesis with the best hypothesis
            elif self.compare_hypotheses(best_hypothesis, next_hypothesis, measurements):
                best_hypothesis = next_hypothesis

        return best_hypothesis

    @staticmethod
    def are_measurements_log_capable(measurements, check_negative_exponents=False):
        """ Checks if logarithmic models can be used to describe the measurements.
            If the parameter values are smaller than 1 log terms are not allowed."""

        if check_negative_exponents:
            for measurement in measurements:
                for value in measurement.coordinate:
                    if value <= 1.0:
                        return False
        else:
            for measurement in measurements:
                for value in measurement.coordinate:
                    if value < 1.0:
                        return False

        return True
