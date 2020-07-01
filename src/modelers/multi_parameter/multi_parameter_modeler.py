"""
This file is part of the Extra-P software (https://github.com/MeaParvitas/Extra-P)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""

from typing import List, Tuple, Dict
from entities.terms import CompoundTerm
from entities.hypotheses import SingleParameterHypothesis
from entities.functions import SingleParameterFunction
from entities.functions import ConstantFunction
from entities.hypotheses import ConstantHypothesis
from entities.model import Model
from entities.coordinate import Coordinate
from modelers import single_parameter
from entities.functions import MultiParameterFunction
from entities.terms import MultiParameterTerm
from entities.hypotheses import MultiParameterHypothesis
import logging
import copy
from pip._internal.cli.cmdoptions import retries
from modelers.abstract_modeler import MultiParameterModeler as AbstractMultiParameterModeler
from modelers.abstract_modeler import LegacyModeler
from entities.measurement import Measurement
import itertools


class MultiParameterModeler(AbstractMultiParameterModeler, LegacyModeler):
    """
    This class represents the modeler for multi parameter functions.
    In order to create a model measurements at least 5 points are needed.
    The result is either a constant function or one based on the PMNF.
    """

    NAME = 'Multiparameter'

    def __init__(self):
        """
        Initialize SingleParameterModeler object.
        """
        super().__init__(use_median=False, single_parameter_modeler=single_parameter.default())
        # value for the minimum number of measurement points required for modeling
        self.min_measurement_points = 5

    def compare_parameter_values(self, parameter_value_list1, parameter_value_list2):
        """
        This method compares the parameter values of two coordinates with each other
        to see if they are equal and returns a True or False.
        """
        if len(parameter_value_list1) != len(parameter_value_list2):
            return False
        for i in range(len(parameter_value_list1)):
            if parameter_value_list1[i] != parameter_value_list2[i]:
                return False
        return True

    def get_parameter_values(self, coordinate, parameter_id):
        """
        This method returns the parameter values from the coordinate.
        But only the ones necessary for the compare_parameter_values() method.
        """
        parameter_value_list = []
        _, value = coordinate.get_parameter_value(parameter_id)
        parameter_value_list.append(float(value))
        return parameter_value_list

    def find_first_measurement_points(self, measurements: List[Measurement]):
        """
        This method returns the measurments that should be used for creating
        the single parameter models.
        """
        def coordinate_is_mostly_lower(coordinate, other, except_position):
            return all(a <= b
                       for i, (a, b) in enumerate(zip(coordinate, other))
                       if i != except_position)

        def coordinate_is_mostly_equal(coordinate, other, except_position):
            return all(a == b
                       for i, (a, b) in enumerate(zip(coordinate, other))
                       if i != except_position)

        dimensions = measurements[0].coordinate.dimensions
        min_coordinate = [
            Coordinate(float('Inf') for _ in range(dimensions))
            for _ in range(dimensions)
        ]
        candidate_list = [[] for _ in range(dimensions)]
        for m in measurements:
            for p in range(dimensions):
                if coordinate_is_mostly_equal(m.coordinate, min_coordinate[p], p):
                    m_sp = copy.copy(m)
                    m_sp.coordinate = Coordinate(m.coordinate[p])
                    candidate_list[p].append(m_sp)
                elif coordinate_is_mostly_lower(m.coordinate, min_coordinate[p], p):
                    candidate_list[p].clear()
                    m_sp = copy.copy(m)
                    m_sp.coordinate = Coordinate(m.coordinate[p])
                    candidate_list[p].append(m_sp)
                    min_coordinate[p] = m.coordinate

        return candidate_list

    def create_model(self, measurements: List[Measurement]):
        """
        Create a model for the given callpath and metric using the given data.
        """

        # use the first base points found for each parameter for modeling for the single parameter functions
        measurements_sp = self.find_first_measurement_points(measurements)
        # print(coordinates_list)

        # model all single parmaeter experiments using only the selected points from the step before
        parameters = list(range(measurements[0].coordinate.dimensions))

        models = self.single_parameter_modeler.model(measurements_sp)
        functions = [m.hypothesis.function for m in models]

        # check if the number of measurements satisfies the reuqirements of the modeler (>=5)
        if len(measurements) < self.min_measurement_points:
            logging.error("Number of measurements for each parameter needs to be at least 5 in order to create a performance model.")
            return None

        # get the coordinates for modeling
        coordinates = list(dict.fromkeys(m.coordinate for m in measurements).keys())

        # use all available additional points for modeling the multi parameter models
        constantCost = 0
        meanModel = 0

        for m in measurements:
            meanModel += m.value(self.use_median) / float(len(measurements))
        for m in measurements:
            constantCost += (m.value(self.use_median) - meanModel) * (m.value(self.use_median) - meanModel)

        # find out which parameters should be deleted
        compound_term_pairs = []

        for i, function in enumerate(functions):
            terms = function.get_compound_terms()
            if len(terms) > 0:
                compound_term = terms[0]
                compound_term.coefficient = 1
                compound_term_pairs.append((i, compound_term))

        # see if the function is constant
        if len(compound_term_pairs) == 0:
            constant_function = ConstantFunction()
            constant_function.set_constant_coefficient(meanModel)
            constant_hypothesis = ConstantHypothesis(constant_function, self.use_median)
            constant_hypothesis.compute_cost(measurements)
            return Model(constant_hypothesis)

        # in case is only one parameter, make a single parameter function
        elif len(compound_term_pairs) == 1:
            param, compound_term = compound_term_pairs[0]
            multi_parameter_function = MultiParameterFunction()
            multi_parameter_term = MultiParameterTerm(compound_term_pairs[0])
            multi_parameter_term.set_coefficient(compound_term.get_coefficient())
            multi_parameter_function.add_multi_parameter_term(multi_parameter_term)
            constant_coefficient = functions[param].get_constant_coefficient()
            multi_parameter_function.set_constant_coefficient(constant_coefficient)
            multi_parameter_hypothesis = MultiParameterHypothesis(multi_parameter_function, self.use_median)
            multi_parameter_hypothesis.compute_cost(measurements, coordinates)
            return Model(multi_parameter_hypothesis)

        hypotheses = []

        # create multiplicative multi parameter terms
        mult = MultiParameterTerm(*compound_term_pairs)

        # create additive multi parameter terms
        add = [MultiParameterTerm(ctp) for ctp in compound_term_pairs]

        # add Hypotheses for 2 parameter models
        if len(parameters) == 2:

            # create multi parameter functions
            mp_functions = [
                # create f1 function a*b
                MultiParameterFunction(mult),
                # create f2 function a*b+a
                MultiParameterFunction(add[0], mult),
                # create f3 function a*b+b
                MultiParameterFunction(add[1], mult),
                # create f4 function a+b
                MultiParameterFunction(*add)
            ]

            # create the hypotheses from the functions
            mph = [MultiParameterHypothesis(f, self.use_median)
                   for f in mp_functions]

            # add the hypothesis to the list
            hypotheses.extend(mph)

        # add Hypotheses for 3 parameter models
        if len(parameters) == 3:

            # create multiplicative multi parameter terms
            # x*y
            mult_x_y = MultiParameterTerm(compound_term_pairs[0], compound_term_pairs[1])
            # y*z
            mult_y_z = MultiParameterTerm(compound_term_pairs[1], compound_term_pairs[2])
            # x*z
            mult_x_z = MultiParameterTerm(compound_term_pairs[0], compound_term_pairs[2])

            # create multi parameter functions
            mp_functions = [

                # x*y*z
                MultiParameterFunction(mult),
                # x+y+z
                MultiParameterFunction(*add),

                # x*y*z+x
                MultiParameterFunction(mult, add[0]),
                # x*y*z+y
                MultiParameterFunction(mult, add[1]),
                # x*y*z+z
                MultiParameterFunction(mult, add[2]),

                # x*y*z+x*y
                MultiParameterFunction(mult, mult_x_y),
                # x*y*z+y*z
                MultiParameterFunction(mult, mult_y_z),
                # x*y*z+x*z
                MultiParameterFunction(mult, mult_x_z),

                # x*y*z+x*y+z
                MultiParameterFunction(mult, mult_x_y, add[2]),
                # x*y*z+y*z+x
                MultiParameterFunction(mult, mult_y_z, add[0]),
                # x*y*z+x*z+y
                MultiParameterFunction(mult, mult_x_z, add[1]),

                # x*y*z+x+y
                MultiParameterFunction(mult, add[0], add[1]),
                # x*y*z+x+z
                MultiParameterFunction(mult, add[0], add[2]),
                # x*y*z+y+z
                MultiParameterFunction(mult, add[1], add[2]),

                # x*y+z
                MultiParameterFunction(mult_x_y, add[2]),
                # x*y+z+y
                MultiParameterFunction(mult_x_y, add[2], add[1]),
                # x*y+z+x
                MultiParameterFunction(mult_x_y, add[2], add[0]),

                # x*z+y
                MultiParameterFunction(mult_x_z, add[1]),
                # x*z+y+x
                MultiParameterFunction(mult_x_z, add[1], add[0]),
                # x*z+y+z
                MultiParameterFunction(mult_x_z, add[1], add[2]),

                # y*z+x
                MultiParameterFunction(mult_y_z, add[0]),
                # y*z+x+y
                MultiParameterFunction(mult_y_z, add[0], add[1]),
                # y*z+x+z
                MultiParameterFunction(mult_y_z, add[0], add[2])
            ]

            # create the hypotheses from the functions
            mph = [MultiParameterHypothesis(f, self.use_median)
                   for f in mp_functions]

            # add the hypothesis to the list
            hypotheses.extend(mph)

        # select one function as the bestHypothesis for the start
        best_hypothesis = copy.deepcopy(hypotheses[0])
        best_hypothesis.compute_coefficients(measurements, coordinates)
        best_hypothesis.compute_cost(measurements, coordinates)
        best_hypothesis.compute_adjusted_rsquared(constantCost, measurements)

        print("hypothesis 0 : "+str(best_hypothesis.get_function().to_string())+" --- smape: "+str(best_hypothesis.get_SMAPE())+" --- ar2: "+str(best_hypothesis.get_AR2()) +
              " --- rss: "+str(best_hypothesis.get_RSS())+" --- rrss: "+str(best_hypothesis.get_rRSS())+" --- re: "+str(best_hypothesis.get_RE()))

        # find the best hypothesis
        for i in range(1, len(hypotheses)):
            hypotheses[i].compute_coefficients(measurements, coordinates)
            hypotheses[i].compute_cost(measurements, coordinates)
            hypotheses[i].compute_adjusted_rsquared(constantCost, measurements)

            print("hypothesis "+str(i)+" : "+str(hypotheses[i].get_function().to_string())+" --- smape: "+str(hypotheses[i].get_SMAPE())+" --- ar2: " +
                  str(hypotheses[i].get_AR2())+" --- rss: "+str(hypotheses[i].get_RSS())+" --- rrss: "+str(hypotheses[i].get_rRSS())+" --- re: "+str(hypotheses[i].get_RE()))

            if hypotheses[i].get_SMAPE() < best_hypothesis.get_SMAPE():
                best_hypothesis = copy.deepcopy(hypotheses[i])

        # add the best found hypothesis to the model list
        model = Model(best_hypothesis)

        print("best hypothesis: "+str(best_hypothesis.get_function().to_string())+" --- smape: "+str(best_hypothesis.get_SMAPE())+" --- ar2: "+str(best_hypothesis.get_AR2()) +
              " --- rss: "+str(best_hypothesis.get_RSS())+" --- rrss: "+str(best_hypothesis.get_rRSS())+" --- re: "+str(best_hypothesis.get_RE()))

        return model
