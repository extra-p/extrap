"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""

import copy
import itertools
import logging
import warnings
from typing import List

from entities.terms import CompoundTerm
from entities.functions import ConstantFunction, SingleParameterFunction
from entities.hypotheses import ConstantHypothesis, SingleParameterHypothesis
from entities.model import Model
from entities.parameter import Parameter
from entities.measurement import Measurement
from entities.coordinate import Coordinate
from modelers.abstract_modeler import LegacyModeler
from entities.model import Model
from modelers.modeler_options import modeler_options


@modeler_options
class SingleParameterModeler(LegacyModeler):
    """
    This class represents the modeler for single parameter functions.
    In order to create a model measurements at least 5 points are needed.
    The result is either a constant function or one based on the PMNF.
    """

    NAME = 'Basic'
    allow_log_terms = modeler_options.add(True, bool, 'Allows models with logarithmic terms')
    use_crossvalidation = modeler_options.add(True, bool, 'Enables cross-validation', name='Cross-Validation')

    poly_exponents = modeler_options.add('', str, 'Set of polynomial exponents. Use comma separated list.',
                                         name='Polynomial', on_change=lambda self, v: self._exponents_changed())
    log_exponents = modeler_options.add('', str, 'Set of logarithmic exponents. Use comma separated list.',
                                        name='Logarithmic', on_change=lambda self, v: self._exponents_changed())
    retain_default_exponents = modeler_options.add(False, bool,
                                                   'If set the default exponents are added to the given ones.',
                                                   name='Retain Default',
                                                   on_change=lambda self, v: self._exponents_changed())
    force_combination_exponents = modeler_options.add(False, bool,
                                                      'If set the exact combination of exponents is forced.',
                                                      name='Force Combination',
                                                      on_change=lambda self, v: self._exponents_changed())
    modeler_options.group('Exponents', poly_exponents, log_exponents, retain_default_exponents,
                          force_combination_exponents)

    def __init__(self):
        """
        Initialize SingleParameterModeler object.
        """
        super().__init__(use_median=False)

        # value for the minimum term contribution
        self.epsilon = 0.0005

        # minimum allowed value for a constant coefficient befor it is set to 0
        self.phi = 1e-3

        # value for the minimum number of measurement points required for modeling
        self.min_measurement_points = 5

        # create the building blocks for the hypothesis
        self.hypotheses_building_blocks: List[CompoundTerm] = self.create_default_building_blocks(
            self.allow_log_terms)

    def _exponents_changed(self):
        def parse_expos(expos):
            expos = expos.split(',')
            return [float(e) if '.' in e else int(e)
                    for e in expos if e.isnumeric()]

        polyexpos = parse_expos(self.poly_exponents)
        logexpos = parse_expos(self.log_exponents)

        if len(polyexpos) > 0 or len(logexpos) > 0:
            self.hypotheses_building_blocks = self.generate_building_blocks(polyexpos, logexpos,
                                                                            self.force_combination_exponents)
            if self.retain_default_exponents:
                self.hypotheses_building_blocks.extend(self.create_default_building_blocks(self.allow_log_terms))
        else:
            self.hypotheses_building_blocks = self.create_default_building_blocks(self.allow_log_terms)

    def get_matching_hypotheses(self, measurements: List[Measurement]):
        """
        Checkes if the parameter values are smaller than 1.
        In this case log terms are not allowed.
        These are removed from the returned hypotheses_building_blocks.
        """
        values_log_capable = True
        for measurement in measurements:
            for value in measurement.coordinate:
                if value < 1.0:
                    values_log_capable = False
                    break

        if values_log_capable:
            return self.hypotheses_building_blocks

        return [compound_term
                for compound_term in self.hypotheses_building_blocks
                if not any(t.term_type == "logarithm"
                           for t in compound_term.simple_terms)
                ]

    def create_default_building_blocks(self, allow_log_terms):
        """
        Creates the default building blocks for the single parameter hypothesis
        that will be used during the search for the best hypothesis.
        """

        if allow_log_terms:
            exponents = [(0, 1, 1),
                         (0, 1, 2),
                         (1, 4, 0),
                         (1, 3, 0),
                         (1, 4, 1),
                         (1, 3, 1),
                         (1, 4, 2),
                         (1, 3, 2),
                         (1, 2, 0),
                         (1, 2, 1),
                         (1, 2, 2),
                         (2, 3, 0),
                         (3, 4, 0),
                         (2, 3, 1),
                         (3, 4, 1),
                         (4, 5, 0),
                         (2, 3, 2),
                         (3, 4, 2),
                         (1, 1, 0),
                         (1, 1, 1),
                         (1, 1, 2),
                         (5, 4, 0),
                         (5, 4, 1),
                         (4, 3, 0),
                         (4, 3, 1),
                         (3, 2, 0),
                         (3, 2, 1),
                         (3, 2, 2),
                         (5, 3, 0),
                         (7, 4, 0),
                         (2, 1, 0),
                         (2, 1, 1),
                         (2, 1, 2),
                         (9, 4, 0),
                         (7, 3, 0),
                         (5, 2, 0),
                         (5, 2, 1),
                         (5, 2, 2),
                         (8, 3, 0),
                         (11, 4, 0),
                         (3, 1, 0),
                         (3, 1, 1)]
            # These were used for relearn
            """
            exponents += [(-0, 1, -1),
                          (-0, 1, -2),
                          (-1, 4, -1),
                          (-1, 3, -1),
                          (-1, 4, -2),
                          (-1, 3, -2),
                          (-1, 2, -1),
                          (-1, 2, -2),
                          (-2, 3, -1),
                          (-3, 4, -1),
                          (-2, 3, -2),
                          (-3, 4, -2),
                          (-1, 1, -1),
                          (-1, 1, -2),
                          (-5, 4, -1),
                          (-4, 3, -1),
                          (-3, 2, -1),
                          (-3, 2, -2),
                          (-2, 1, -1),
                          (-2, 1, -2),
                          (-5, 2, -1),
                          (-5, 2, -2),
                          (-3, 1, -1)]
            """
        else:
            exponents = [(1, 4, 0),
                         (1, 3, 0),
                         (1, 2, 0),
                         (2, 3, 0),
                         (3, 4, 0),
                         (4, 5, 0),
                         (1, 1, 0),
                         (5, 4, 0),
                         (4, 3, 0),
                         (3, 2, 0),
                         (5, 3, 0),
                         (7, 4, 0),
                         (2, 1, 0),
                         (9, 4, 0),
                         (7, 3, 0),
                         (5, 2, 0),
                         (8, 3, 0),
                         (11, 4, 0),
                         (3, 1, 0)]
            # These were used for relearn
            """
            exponents += [(-1, 4, 0),
                          (-1, 3, 0),
                          (-1, 2, 0),
                          (-2, 3, 0),
                          (-3, 4, 0),
                          (-4, 5, 0),
                          (-1, 1, 0),
                          (-5, 4, 0),
                          (-4, 3, 0),
                          (-3, 2, 0),
                          (-5, 3, 0),
                          (-7, 4, 0),
                          (-2, 1, 0),
                          (-9, 4, 0),
                          (-7, 3, 0),
                          (-5, 2, 0),
                          (-8, 3, 0),
                          (-11, 4, 0),
                          (-3, 1, 0)]
            """
        hypotheses_building_blocks = [CompoundTerm.create(*e) for e in exponents]
        # print the hypothesis building blocks, compound terms in debug mode
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            parameter = Parameter('p')
            for i, compound_term in enumerate(hypotheses_building_blocks):
                logging.debug(
                    f"Compound term {i}: {compound_term.to_string(parameter)}")

        return hypotheses_building_blocks

    def generate_building_blocks(self, poly_exponents, log_exponents, force_combination=False):
        if force_combination:
            exponents = itertools.product(poly_exponents, log_exponents)
        else:
            exponents = itertools.chain(
                itertools.product(poly_exponents, [0]),
                itertools.product([0], log_exponents),
                itertools.product(poly_exponents, log_exponents))

        return [CompoundTerm.create(*e) for e in exponents]

    def create_constant_model(self, measurements):
        """
        Creates a constant model that fits the data using a ConstantFunction.
        """

        # compute the constant coefficient
        mean_model = sum(m.value(self.use_median) / len(measurements)
                         for m in measurements)

        # create a constant function
        return ConstantFunction(mean_model)

    def build_hypothesis(self, compound_term):
        """
        Builds the next hypothesis that should be analysed based on the given compound term.
        """

        # create single parameter function
        return SingleParameterFunction(copy.copy(compound_term))

    def compare_hypotheses(self, old, new, measurements):
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

        return new.SMAPE < old.SMAPE

    def find_best_hypothesis(self, constant_hypothesis, constant_cost, measurements):
        """
        Searches for the best single parameter hypothesis and returns it.
        """

        # currently the constant hypothesis is the best hypothesis
        best_hypothesis = constant_hypothesis

        hypotheses_building_blocks = self.get_matching_hypotheses(measurements)

        # search for the best hypothesis over all functions that can be build with the basic building blocks using leave one out crossvalidation
        for i, compound_term in enumerate(hypotheses_building_blocks):
            compound_term = copy.copy(compound_term)

            # create next function that will be analyzed
            next_function = self.build_hypothesis(compound_term)

            # create single parameter hypothesis from function
            next_hypothesis = SingleParameterHypothesis(
                next_function, self.use_median)

            if self.use_crossvalidation:
                # cycle through points and leave one out per iteration
                for element_id in range(len(measurements)):
                    # copy measurements to create the training sets
                    training_measurements = copy.copy(measurements)

                    # remove one element the set
                    training_measurements.pop(element_id)

                    # validation set
                    validation_measurement = measurements[element_id]

                    # compute the model coefficients based on the training data
                    next_hypothesis.compute_coefficients(training_measurements)

                    # check if the constant coefficient should actually be 0
                    next_hypothesis.clean_constant_coefficient(self.phi, training_measurements)

                    # compute the cost of the single parameter model for the validation data
                    next_hypothesis.compute_cost(
                        training_measurements, validation_measurement)

                # compute the model coefficients using all data
                next_hypothesis.compute_coefficients(measurements)
                logging.debug(
                    f"Single parameter model {i}: " + next_hypothesis.function.to_string(Parameter('p')))
            else:
                # compute the model coefficients based on the training data
                next_hypothesis.compute_coefficients(measurements)

                # check if the constant coefficient should actually be 0
                next_hypothesis.clean_constant_coefficient(
                    self.phi, measurements)

                # compute the cost of the single parameter model for the validation data
                next_hypothesis.compute_cost_all_points(measurements)

            # compute the AR2 for the hypothesis
            next_hypothesis.compute_adjusted_rsquared(
                constant_cost, measurements)

            # check if hypothesis is valid
            if not next_hypothesis.is_valid():
                logging.info(
                    "Numeric imprecision found. Model is invalid and will be ignored.")

            # compare the new hypothesis with the best hypothesis
            elif self.compare_hypotheses(best_hypothesis, next_hypothesis, measurements):
                best_hypothesis = next_hypothesis

        return best_hypothesis

    def create_model(self, measurements: List[Measurement]):
        """
        Create a model for the given callpath and metric using the given data.
        """

        # check if the number of measurements satisfies the reuqirements of the modeler (>=5)
        if len(measurements) < self.min_measurement_points:
            warnings.warn(
                "Number of measurements for a parameter needs to be at least 5 in order to create a performance model.")
            # return None

        # create a constant function
        constant_function = self.create_constant_model(measurements)

        # create a constant hypothesis from the constant function
        constant_hypothesis = ConstantHypothesis(
            constant_function, self.use_median)
        logging.debug("Constant model: " +
                      constant_hypothesis.function.to_string())

        # compute cost of the constant model
        constant_hypothesis.compute_cost(measurements)
        constant_cost = constant_hypothesis.get_RSS()
        logging.debug("Constant model cost: " + str(constant_cost))

        # use constat model when cost is 0
        if constant_cost == 0:
            logging.debug("Using constant model.")
            return Model(constant_hypothesis)

        # otherwise start searching for the best hypothesis based on the pmnf
        else:
            logging.debug("Searching for a single parameter model.")

            # search for the best single parmater hypothesis
            best_hypothesis = self.find_best_hypothesis(
                constant_hypothesis, constant_cost, measurements)
            return Model(best_hypothesis)
