# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
import itertools
import logging
import warnings
from typing import List, Sequence

from extrap.entities.functions import SingleParameterFunction
from extrap.entities.hypotheses import SingleParameterHypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.model import Model
from extrap.entities.parameter import Parameter
from extrap.entities.terms import CompoundTerm
from extrap.modelers.abstract_modeler import SingularModeler
from extrap.modelers.modeler_options import modeler_options
from extrap.modelers.single_parameter.abstract_base import AbstractSingleParameterModeler


@modeler_options
class SingleParameterModeler(AbstractSingleParameterModeler, SingularModeler):
    """
    This class represents the modeler for single parameter functions.
    In order to create a model measurements at least 5 points are needed.
    The result is either a constant function or one based on the PMNF.
    """

    NAME = 'Basic'
    allow_log_terms = modeler_options.add(True, bool, 'Allows models with logarithmic terms',
                                          on_change=lambda self, v: self._exponents_changed())
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

    def get_matching_hypotheses(self, measurements: Sequence[Measurement]):
        """
        Checks if the parameter values are smaller than 1.
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

    @staticmethod
    def create_default_building_blocks(allow_log_terms):
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

    @staticmethod
    def generate_building_blocks(poly_exponents, log_exponents, force_combination=False):
        if force_combination:
            exponents = itertools.product(poly_exponents, log_exponents)
        else:
            exponents = itertools.chain(
                itertools.product(poly_exponents, [0]),
                itertools.product([0], log_exponents),
                itertools.product(poly_exponents, log_exponents))

        return [CompoundTerm.create(*e) for e in exponents]

    def build_hypotheses(self, measurements):
        """
        Builds the next hypothesis that should be analysed based on the given compound term.
        """
        hypotheses_building_blocks = self.get_matching_hypotheses(measurements)

        # search for the best hypothesis over all functions that can be build with the basic building blocks
        # using leave one out crossvalidation
        for i, compound_term in enumerate(hypotheses_building_blocks):
            # create next function that will be analyzed
            next_function = SingleParameterFunction(copy.copy(compound_term))

            # create single parameter hypothesis from function
            yield SingleParameterHypothesis(next_function, self.use_median)

    def create_model(self, measurements: Sequence[Measurement]):
        """
        Create a model for the given callpath and metric using the given data.
        """

        # check if the number of measurements satisfies the requirements of the modeler (>=5)
        if len(measurements) < self.min_measurement_points:
            warnings.warn(
                "Number of measurements for a parameter needs to be at least 5 in order to create a performance model.")
            # return None

        # create a constant model
        constant_hypothesis, constant_cost = self.create_constant_model(measurements)
        logging.debug("Constant model: " + constant_hypothesis.function.to_string())
        logging.debug("Constant model cost: " + str(constant_cost))

        # use constant model when cost is 0
        if constant_cost == 0:
            logging.debug("Using constant model.")
            return Model(constant_hypothesis)

        # otherwise start searching for the best hypothesis based on the PMNF
        else:
            logging.debug("Searching for a single parameter model.")
            # search for the best single parameter hypothesis
            hypotheses_generator = self.build_hypotheses(measurements)
            best_hypothesis = self.find_best_hypothesis(hypotheses_generator, constant_cost, measurements,
                                                        constant_hypothesis)
            return Model(best_hypothesis)
