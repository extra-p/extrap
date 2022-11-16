# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
import logging
import warnings
from itertools import accumulate
from typing import Sequence

from extrap.entities.functions import ConstantFunction
from extrap.entities.functions import MultiParameterFunction
from extrap.entities.hypotheses import ConstantHypothesis
from extrap.entities.hypotheses import MultiParameterHypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.model import Model
from extrap.entities.terms import MultiParameterTerm
from extrap.modelers.modeler_options import modeler_options
from extrap.modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler
from extrap.util.itertools import pairwise
from extrap.util.progress_bar import DUMMY_PROGRESS


@modeler_options
class MultiParameterHwcModeler(MultiParameterModeler):
    """
    This class represents the modeler for multi parameter functions.
    In order to create a model measurements at least 5 points are needed.
    The result is either a constant function or one based on the PMNF.
    """

    NAME = 'Multi-Parameter HWC'
    DESCRIPTION = "Modeler for multi-parameter models; supports full and sparse modeling."

    no_negative_coefficients = modeler_options.add(True, bool)

    def __init__(self):
        """
        Initialize SingleParameterModeler object.
        """
        super().__init__()

    def create_model(self, measurement):
        raise RuntimeError()

    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:

        progress_bar.total += len(measurements) * 1.2
        callpath_data = {}
        for measurement_sequence in measurements:
            progress_bar.update(0.1)
            callpath = measurement_sequence[0].callpath
            coordinates = tuple(m.coordinate.as_tuple() for m in measurement_sequence)
            key = callpath, coordinates
            if key not in callpath_data:
                callpath_data[key] = [measurement_sequence]
            else:
                callpath_data[key].append(measurement_sequence)
        callpath_models = {}
        for key, measurement_set in callpath_data.items():
            model_set = self._generate_single_parameter_models(measurement_set)
            for measurement_sequence, models in zip(measurement_set, model_set):
                progress_bar.update()
                callpath_models[key[0], measurement_sequence[0].metric, key[1]] = self.combine_models(
                    measurement_sequence, models)
        models = []
        for measurement_sequence in measurements:
            progress_bar.update(0.1)
            callpath = measurement_sequence[0].callpath
            metric = measurement_sequence[0].metric
            coordinates = tuple(m.coordinate.as_tuple() for m in measurement_sequence)
            key = callpath, metric, coordinates
            models.append(callpath_models[key])
        return models

    def combine_models(self, measurements: Sequence[Measurement], models_per_parameter: Sequence[Model]) -> Model:
        """
        Create a multi-parameter model using the given measurements.
        """

        functions = [m.hypothesis.function for m in models_per_parameter]

        # check if the number of measurements satisfies the reuqirements of the modeler (>=5)
        if len(measurements) < self.min_measurement_points:
            warnings.warn("Number of measurements for each parameter needs to be at least 5"
                          " in order to create a performance model.")
            # return None

        # get the coordinates for modeling
        # coordinates = list(dict.fromkeys(m.coordinate for m in measurements).keys())

        # use all available additional points for modeling the multi-parameter models
        constantCost = 0
        meanModel = 0

        for m in measurements:
            meanModel += m.value(self.use_median) / float(len(measurements))
        for m in measurements:
            constantCost += (m.value(self.use_median) - meanModel) * (m.value(self.use_median) - meanModel)

        # find out which parameters should be kept
        compound_term_pairs = []

        for i, function in enumerate(functions):
            terms = function.compound_terms
            if len(terms) > 0:
                compound_term = terms[0]
                compound_term_pairs.append((i, compound_term))

        # see if the function is constant
        if len(compound_term_pairs) == 0:
            constant_function = ConstantFunction()
            constant_function.constant_coefficient = meanModel
            constant_hypothesis = ConstantHypothesis(constant_function, self.use_median)
            constant_hypothesis.compute_cost(measurements)
            return Model(constant_hypothesis)

        # in case of only one parameter, make a multi parameter function with only one parameter
        elif len(compound_term_pairs) == 1:
            param, compound_term = compound_term_pairs[0]
            # reset coefficient of compound term
            coefficient = compound_term.coefficient
            compound_term.reset_coefficients()

            # multi parameter function with reused coefficients
            multi_parameter_term = MultiParameterTerm(compound_term_pairs[0])
            multi_parameter_term.coefficient = coefficient
            compound_term.reset_coefficients()
            multi_parameter_function = MultiParameterFunction(multi_parameter_term)
            multi_parameter_function.constant_coefficient = functions[param].constant_coefficient
            multi_parameter_hypothesis = MultiParameterHypothesis(multi_parameter_function, self.use_median)
            multi_parameter_hypothesis.compute_cost(measurements)

            # multi parameter function with newly calculated coefficients using all values
            recomputed_coeffficients_hypothesis = MultiParameterHypothesis(
                MultiParameterFunction(MultiParameterTerm(compound_term_pairs[0])), self.use_median)
            recomputed_coeffficients_hypothesis.compute_coefficients(measurements,
                                                                     no_negative_coefficients= \
                                                                         self.no_negative_coefficients)
            recomputed_coeffficients_hypothesis.compute_cost(measurements)

            # select best
            if self.compare_with_RSS and recomputed_coeffficients_hypothesis.RSS < multi_parameter_hypothesis.RSS:
                multi_parameter_hypothesis = recomputed_coeffficients_hypothesis
            elif recomputed_coeffficients_hypothesis.SMAPE < multi_parameter_hypothesis.SMAPE:
                multi_parameter_hypothesis = recomputed_coeffficients_hypothesis

            multi_parameter_hypothesis.compute_adjusted_rsquared(constantCost, measurements)
            return Model(multi_parameter_hypothesis)

        # reset coefficients of compound terms
        for p, compound_term in compound_term_pairs:
            compound_term.reset_coefficients()

        # create multiplicative multi parameter term
        mult = MultiParameterTerm(*compound_term_pairs)

        # create additive multi parameter terms
        add = [MultiParameterTerm(ctp) for ctp in compound_term_pairs]

        # create multi parameter functions
        mp_functions = [
            # create f1 function a*b
            MultiParameterFunction(mult),
            # create f4 function a+b
            MultiParameterFunction(*add)
        ]

        if not self.allow_combinations_of_sums_and_products:
            pass
        # add Hypotheses for 2 parameter models
        elif len(compound_term_pairs) == 2:
            mp_functions += [
                # create f2 function a*b+a
                MultiParameterFunction(add[0], mult),
                # create f3 function a*b+b
                MultiParameterFunction(add[1], mult)
            ]
        # add Hypotheses for 3 parameter models
        elif len(compound_term_pairs) == 3:
            # create multiplicative multi parameter terms
            # x*y
            mult_x_y = MultiParameterTerm(compound_term_pairs[0], compound_term_pairs[1])
            # y*z
            mult_y_z = MultiParameterTerm(compound_term_pairs[1], compound_term_pairs[2])
            # x*z
            mult_x_z = MultiParameterTerm(compound_term_pairs[0], compound_term_pairs[2])

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
        hypotheses = [MultiParameterHypothesis(f, self.use_median)
                      for f in mp_functions]

        # select one function as the bestHypothesis for the start
        best_hypothesis = copy.deepcopy(hypotheses[0])
        best_hypothesis.compute_coefficients(measurements, no_negative_coefficients=self.no_negative_coefficients)
        best_hypothesis.compute_cost(measurements)
        best_hypothesis.compute_adjusted_rsquared(constantCost, measurements)

        logging.info(f"hypothesis 0: {best_hypothesis.function} --- smape: {best_hypothesis.SMAPE} "
                     f"--- ar2: {best_hypothesis.AR2} --- rss: {best_hypothesis.RSS} "
                     f"--- rrss: {best_hypothesis.rRSS} --- re: {best_hypothesis.RE}")

        # find the best hypothesis
        for i, hypothesis in enumerate(hypotheses):
            hypothesis.compute_coefficients(measurements)
            hypothesis.compute_cost(measurements)
            hypothesis.compute_adjusted_rsquared(constantCost, measurements)

            logging.info(f"hypothesis {i}: {hypothesis.function} --- smape: {hypothesis.SMAPE} "
                         f"--- ar2: {hypothesis.AR2} --- rss: {hypothesis.RSS} "
                         f"--- rrss: {hypothesis.rRSS} --- re: {hypothesis.RE}")

            term_contribution_big_enough = True
            # for all compound terms check if they are smaller than minimum allowed contribution
            for term in hypothesis.function.compound_terms:
                # ignore this hypothesis, since one of the terms contributes less than epsilon to the function
                if term.coefficient == 0 or hypothesis.calc_term_contribution(term, measurements) < self.epsilon:
                    term_contribution_big_enough = False
                    break

            if not term_contribution_big_enough:
                continue
            elif self.compare_with_RSS:
                if hypotheses[i].RSS < best_hypothesis.RSS:
                    best_hypothesis = copy.deepcopy(hypotheses[i])
            elif hypotheses[i].SMAPE < best_hypothesis.SMAPE:
                best_hypothesis = copy.deepcopy(hypotheses[i])

        # add the best found hypothesis to the model list
        model = Model(best_hypothesis)

        logging.info(f"best hypothesis: {best_hypothesis.function} --- smape: {best_hypothesis.SMAPE} "
                     f"--- ar2: {best_hypothesis.AR2} --- rss: {best_hypothesis.RSS} "
                     f"--- rrss: {best_hypothesis.rRSS} --- re: {best_hypothesis.RE}")

        return model

    def _generate_single_parameter_models(self, measurement_set):
        measurements_sp_set = []
        measurements_index = [0]
        for measurements in measurement_set:
            if self.single_parameter_point_selection == 'auto' \
                    or self.single_parameter_point_selection == 'all':
                measurements_sp = self.find_best_measurement_points(measurements)
                measurements_sp_set.extend(measurements_sp)
                measurements_index.append(len(measurements_sp))
            else:
                # use the first base points found for each parameter for modeling of the single parameter functions
                measurements_sp = self.find_first_measurement_points(measurements)
                measurements_sp_set.extend(measurements_sp)
                measurements_index.append(len(measurements_sp))
        # model all single parameter experiments using only the selected points from the step before
        models_sp_set = self.single_parameter_modeler.model(measurements_sp_set)
        return [models_sp_set[s:e] for s, e in pairwise(accumulate(measurements_index))]
