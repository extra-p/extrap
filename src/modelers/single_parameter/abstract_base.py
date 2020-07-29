"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""
import logging
from abc import ABC
from typing import Iterable, TypeVar, Union, Tuple, Sequence

from entities.functions import ConstantFunction
from entities.hypotheses import Hypothesis, SingleParameterHypothesis, MAX_HYPOTHESIS, ConstantHypothesis
from entities.measurement import Measurement
from entities.parameter import Parameter
from modelers.abstract_modeler import AbstractModeler
from modelers.modeler_options import modeler_options

_H = TypeVar('_H', bound=Hypothesis)
_SH = TypeVar('_SH', bound=SingleParameterHypothesis)
SH = Union[_SH, SingleParameterHypothesis]
H = Union[_H, Hypothesis]


@modeler_options
class AbstractSingleParameterModeler(AbstractModeler, ABC):
    allow_log_terms = modeler_options.add(True, bool, 'Allows models with logarithmic terms')
    use_crossvalidation = modeler_options.add(True, bool, 'Enables cross-validation', name='Cross-Validation')
    compare_with_RSS = modeler_options.add(False, bool)

    def __init__(self, use_median: bool):
        super().__init__(use_median)
        # value for the minimum term contribution
        self.phi = 1e-3
        # minimum allowed value for a constant coefficient befor it is set to 0
        self.epsilon = 0.0005

    def compare_hypotheses(self, old: Hypothesis, new: SingleParameterHypothesis, measurements: Sequence[Measurement]):
        """
        Compares the best with the new hypothesis and decides which one is a better fit for the data.
        If the new hypothesis is better than the best one it becomes the best hypothesis.
        The choice is made based on the RSS, since this is the metric optimised by the Regression.
        """

        # get the compound terms of the new hypothesis
        compound_terms = new.function.compound_terms

        # for all compound terms check if they are smaller than minimum allowed contribution
        for term in compound_terms:

            # ignore this hypothesis, since one of the terms contributes less than epsilon to the function
            if term.coefficient == 0 or new.calc_term_contribution(term, measurements) < self.epsilon:
                return False

        # print smapes in debug mode
        logging.debug("next hypothesis SMAPE: " + str(new.SMAPE) + ' RSS:' + str(new.RSS))
        logging.debug("best hypothesis SMAPE: " + str(old.SMAPE) + ' RSS:' + str(old.RSS))
        if self.compare_with_RSS:
            return new.RSS < old.RSS
        return new.SMAPE < old.SMAPE

    def create_constant_model(self, measurements: Sequence[Measurement]) -> Tuple[ConstantHypothesis, float]:
        """
        Creates a constant model that fits the data using a ConstantFunction.
        """
        # compute the constant coefficient
        mean_model = sum(m.value(self.use_median) / len(measurements) for m in measurements)

        # create a constant function
        constant_function = ConstantFunction(mean_model)
        constant_hypothesis = ConstantHypothesis(constant_function, self.use_median)

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

        # search for the best hypothesis over all functions that can be build with the basic building blocks

        for i, next_hypothesis in enumerate(candidate_hypotheses):

            if self.use_crossvalidation:
                # use leave one out crossvalidation
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
                    next_hypothesis.clean_constant_coefficient(self.phi, training_measurements)

                    # compute the cost of the single parameter model for the validation data
                    next_hypothesis.compute_cost(training_measurements, validation_measurement)

                # compute the model coefficients using all data
                next_hypothesis.compute_coefficients(measurements)
                logging.debug(f"Single parameter model {i}: " + next_hypothesis.function.to_string(Parameter('p')))
            else:
                # compute the model coefficients based on the training data
                next_hypothesis.compute_coefficients(measurements)

                # check if the constant coefficient should actually be 0
                next_hypothesis.clean_constant_coefficient(
                    self.phi, measurements)

                # compute the cost of the single parameter model for the validation data
                next_hypothesis.compute_cost_all_points(measurements)

            # compute the AR2 for the hypothesis
            next_hypothesis.compute_adjusted_rsquared(constant_cost, measurements)

            # check if hypothesis is valid
            if not next_hypothesis.is_valid():
                logging.info(
                    "Numeric imprecision found. Model is invalid and will be ignored.")

            # compare the new hypothesis with the best hypothesis
            elif self.compare_hypotheses(best_hypothesis, next_hypothesis, measurements):
                best_hypothesis = next_hypothesis

        return best_hypothesis