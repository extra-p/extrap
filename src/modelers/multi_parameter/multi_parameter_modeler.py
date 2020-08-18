"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""

import copy
import logging
import warnings
from typing import Sequence

import numpy as np

from entities.coordinate import Coordinate
from entities.functions import ConstantFunction
from entities.functions import MultiParameterFunction
from entities.hypotheses import ConstantHypothesis
from entities.hypotheses import MultiParameterHypothesis
from entities.measurement import Measurement
from entities.model import Model
from entities.terms import MultiParameterTerm
from modelers import single_parameter
from modelers.abstract_modeler import LegacyModeler
from modelers.abstract_modeler import MultiParameterModeler as AbstractMultiParameterModeler
from modelers.modeler_options import modeler_options


@modeler_options
class MultiParameterModeler(AbstractMultiParameterModeler, LegacyModeler):
    """
    This class represents the modeler for multi parameter functions.
    In order to create a model measurements at least 5 points are needed.
    The result is either a constant function or one based on the PMNF.
    """

    NAME = 'Multi-Parameter'

    single_parameter_point_selection = modeler_options.add('auto', str, range=['auto', 'smallest', 'all'],
                                                           description="Sets the point selection method for creating "
                                                                       "the single-parameter models.")
    allow_combinations_of_sums_and_products = modeler_options.add(True, bool,
                                                                  description="Allows models that consist of "
                                                                              "combinations of sums and products.")
    compare_with_RSS = modeler_options.add(False, bool)

    def __init__(self):
        """
        Initialize SingleParameterModeler object.
        """
        super().__init__(use_median=False, single_parameter_modeler=single_parameter.Default())
        # value for the minimum number of measurement points required for modeling
        self.min_measurement_points = 5

    @staticmethod
    def compare_parameter_values(parameter_value_list1, parameter_value_list2):
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

    @staticmethod
    def get_parameter_values(coordinate, parameter_id):
        """
        This method returns the parameter values from the coordinate.
        But only the ones necessary for the compare_parameter_values() method.
        """
        parameter_value_list = []
        _, value = coordinate.get_parameter_value(parameter_id)
        parameter_value_list.append(float(value))
        return parameter_value_list

    def find_all_measurement_points(self, measurements: Sequence[Measurement]):

        if len(measurements) < self.min_measurement_points ** 2:
            return None

        dimensions = measurements[0].coordinate.dimensions
        groups = [
            {} for _ in range(dimensions)
        ]

        for m in measurements:
            for p in range(dimensions):
                coordinate_p_ = m.coordinate[p]
                groups_p_ = groups[p]
                if coordinate_p_ in groups_p_:
                    groups_p_[coordinate_p_].append(m)
                else:
                    groups_p_[coordinate_p_] = [m]

        def coordinate_except(m, pg):
            return tuple(c for pc, c in enumerate(m.coordinate) if pc != pg)

        for pg, g in enumerate(groups):
            cms = iter(g.values())
            first = next(cms)
            len_first = len(first)
            coord_set = set(coordinate_except(m, pg) for m in first)
            for ms in cms:
                if len_first != len(ms):
                    return None
                if any(coordinate_except(m, pg) not in coord_set for m in ms):
                    return None

        def make_measurement(c, ms: Sequence[Measurement]):
            if self.use_median:
                values = [m.median for m in ms]
            else:
                values = [m.mean for m in ms]

            measurement = Measurement(Coordinate(c), ms[0].callpath, ms[0].metric, values)
            measurement.median = measurement.mean
            if measurement.mean == 0:
                measurement.maximum = np.mean([m.maximum for m in ms])
                measurement.minimum = np.mean([m.minimum for m in ms])
                measurement.std = np.mean([m.std for m in ms])
            else:
                measurement.maximum = np.mean([m.maximum / m.mean for m in ms]) * measurement.mean
                measurement.minimum = np.mean([m.minimum / m.mean for m in ms]) * measurement.mean
                measurement.std = np.mean([m.std / m.mean for m in ms]) * measurement.mean

            return measurement

        combined_measurements = [[make_measurement(c, ms) for c, ms in grp.items()]
                                 for p, grp in enumerate(groups)]

        return combined_measurements

    @staticmethod
    def find_first_measurement_points(measurements: Sequence[Measurement]):
        """
        This method returns the measurements that should be used for creating
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

    def create_model(self, measurements: Sequence[Measurement]):
        """
        Create a model for the given callpath and metric using the given data.
        """
        if self.single_parameter_point_selection == 'auto' or self.single_parameter_point_selection == 'all':
            measurements_sp = self.find_all_measurement_points(measurements)
            if not measurements_sp:
                if self.single_parameter_point_selection == 'all':
                    warnings.warn(
                        "Could not use all measurement points. At least 25 measurements are needed; one for each "
                        "combination of parameters.")
                    # use the first base points found for each parameter for modeling for the single parameter functions
                measurements_sp = self.find_first_measurement_points(measurements)
        else:
            # use the first base points found for each parameter for modeling for the single parameter functions
            measurements_sp = self.find_first_measurement_points(measurements)
        # print(coordinates_list)

        # model all single parameter experiments using only the selected points from the step before
        # parameters = list(range(measurements[0].coordinate.dimensions))

        models = self.single_parameter_modeler.model(measurements_sp)
        functions = [m.hypothesis.function for m in models]

        # check if the number of measurements satisfies the reuqirements of the modeler (>=5)
        if len(measurements) < self.min_measurement_points:
            warnings.warn("Number of measurements for each parameter needs to be at least 5"
                          " in order to create a performance model.")
            # return None

        # get the coordinates for modeling
        # coordinates = list(dict.fromkeys(m.coordinate for m in measurements).keys())

        # use all available additional points for modeling the multi parameter models
        constantCost = 0
        meanModel = 0

        for m in measurements:
            meanModel += m.value(self.use_median) / float(len(measurements))
        for m in measurements:
            constantCost += (m.value(self.use_median) - meanModel) * (m.value(self.use_median) - meanModel)

        # find out which parameters should be kept
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
            # constant_coefficient = functions[param].get_constant_coefficient()
            # multi_parameter_function.set_constant_coefficient(constant_coefficient)
            multi_parameter_hypothesis = MultiParameterHypothesis(multi_parameter_function, self.use_median)
            multi_parameter_hypothesis.compute_coefficients(measurements)
            multi_parameter_hypothesis.compute_cost(measurements)
            return Model(multi_parameter_hypothesis)

        hypotheses = []

        # create multiplicative multi parameter term
        mult = MultiParameterTerm(*compound_term_pairs)

        # create additive multi parameter terms
        add = [MultiParameterTerm(ctp) for ctp in compound_term_pairs]

        # add Hypotheses for 2 parameter models
        if len(compound_term_pairs) == 2:
            # create multi parameter functions
            mp_functions = [
                # create f1 function a*b
                MultiParameterFunction(mult),
                # create f4 function a+b
                MultiParameterFunction(*add)
            ]
            if self.allow_combinations_of_sums_and_products:
                mp_functions += [
                    # create f2 function a*b+a
                    MultiParameterFunction(add[0], mult),
                    # create f3 function a*b+b
                    MultiParameterFunction(add[1], mult)
                ]

            # create the hypotheses from the functions
            mph = [MultiParameterHypothesis(f, self.use_median)
                   for f in mp_functions]

            # add the hypothesis to the list
            hypotheses.extend(mph)

        # add Hypotheses for 3 parameter models
        if len(compound_term_pairs) == 3:
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
                MultiParameterFunction(*add)
            ]
            if self.allow_combinations_of_sums_and_products:
                # create multi parameter functions
                mp_functions += [
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
        best_hypothesis.compute_coefficients(measurements)
        best_hypothesis.compute_cost(measurements)
        best_hypothesis.compute_adjusted_rsquared(constantCost, measurements)

        logging.info(f"hypothesis 0: {best_hypothesis.function} --- smape: {best_hypothesis.SMAPE} "
                     f"--- ar2: {best_hypothesis.AR2} --- rss: {best_hypothesis.RSS} "
                     f"--- rrss: {best_hypothesis.rRSS} --- re: {best_hypothesis.RE}")

        # find the best hypothesis
        for i in range(1, len(hypotheses)):
            hypotheses[i].compute_coefficients(measurements)
            hypotheses[i].compute_cost(measurements)
            hypotheses[i].compute_adjusted_rsquared(constantCost, measurements)

            logging.info(f"hypothesis {i}: {hypotheses[i].function} --- smape: {hypotheses[i].SMAPE} "
                         f"--- ar2: {hypotheses[i].AR2} --- rss: {hypotheses[i].RSS} "
                         f"--- rrss: {hypotheses[i].rRSS} --- re: {hypotheses[i].RE}")
            if self.compare_with_RSS:
                if hypotheses[i].RSS < best_hypothesis.RSS:
                    best_hypothesis = copy.deepcopy(hypotheses[i])
            elif hypotheses[i].get_SMAPE() < best_hypothesis.get_SMAPE():
                best_hypothesis = copy.deepcopy(hypotheses[i])

        # add the best found hypothesis to the model list
        model = Model(best_hypothesis)

        logging.info(f"best hypothesis: {best_hypothesis.function} --- smape: {best_hypothesis.SMAPE} "
                     f"--- ar2: {best_hypothesis.AR2} --- rss: {best_hypothesis.RSS} "
                     f"--- rrss: {best_hypothesis.rRSS} --- re: {best_hypothesis.RE}")

        return model
