# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
import math
import warnings
from typing import Sequence

import numpy as np

from extrap.entities.functions import SegmentedFunction
from extrap.entities.hypotheses import SingleParameterHypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.model import Model, SegmentedModel
from extrap.modelers.modeler_options import modeler_options
from extrap.modelers.single_parameter.basic import SingleParameterModeler


@modeler_options
class SegmentedModeler(SingleParameterModeler):
    """
    This class represents the modeler for segmented single parameter functions.
    In order to create a model measurements at least 5 points are needed.
    The result is either a constant function or one based on the PMNF.
    """

    NAME = 'Segmented'
    DESCRIPTION = "Modeler for single-parameter models; traverses the search-space of all defined hypotheses. Able to detect segmented behavior in measurements. When segmented data is found the modeler will return two models."

    def __init__(self):
        """
        Initialize SegmentedParameterModeler object.
        """
        super().__init__()

    def create_model(self, measurements: Sequence[Measurement]):
        """
        Create a model for the given callpath and metric using the given data.
        """

        # check if the number of measurements satisfies the requirements of the modeler (>=5)
        if len(measurements) < self.min_measurement_points:
            warnings.warn(
                "Number of measurements for a parameter needs to be at least 5 in order to create a performance model.")

        # identify subsets
        subsets = []
        nr_subsets = len(measurements) - (self.min_measurement_points - 1)
        # print("DEBUG nr_subsets:",nr_subsets)
        for i in range(nr_subsets):
            subset = []
            for j in range(self.min_measurement_points):
                subset.append(measurements[i + j])
            # print("subset:",subset)
            subsets.append(subset)
        # print("DEBUG subsets:",subsets)

        # create a model for each subset
        models = []
        for subset in subsets:

            # create a constant model
            constant_hypothesis, constant_cost = self.create_constant_model(subset)
            logging.debug("Constant model: " + constant_hypothesis.function.to_string())
            logging.debug("Constant model cost: " + str(constant_cost))

            # use constant model when cost is 0
            if constant_cost == 0:
                logging.debug("Using constant model.")
                models.append(Model(constant_hypothesis))

            # otherwise start searching for the best hypothesis based on the PMNF
            else:
                logging.debug("Searching for a single-parameter model.")
                # search for the best single parameter hypothesis
                hypotheses_generator = self.build_hypotheses(subset)
                best_hypothesis = self.find_best_hypothesis(hypotheses_generator, constant_cost, subset,
                                                            constant_hypothesis)
                models.append(best_hypothesis)

        # print("DEBUG models:",models)
        nRSS_values = []
        for m in models:
            # print("nRSS:",m._nRSS)
            nRSS_values.append(abs(m._nRSS))
            # print(str(m.function))

        # print(nRSS_values)
        theta = max(nRSS_values)

        epsilon_values = []
        for i in range(len(subsets)):
            if i == 0:
                epsilon_values.append(-math.inf)
            else:
                epsilon_values.append(nRSS_values[i] / (nRSS_values[i - 1] + 0.000000001))

        dataset_segmented = False
        if theta > 0.5:
            dataset_segmented = True
        if len(epsilon_values)==1 and math.isnan(epsilon_values[0]):
            dataset_segmented = False
        else:
            if np.nanmax(epsilon_values) > 4:
                dataset_segmented = True

        # print("DEBUG dataset_segmented:",dataset_segmented)
        # print("DEBUG epsilon_values:",epsilon_values)

        if dataset_segmented:

            pattern = ""
            for i in range(len(nRSS_values)):
                nRSS = nRSS_values[i]
                epsilon = epsilon_values[i]
                if nRSS >= 0.1:
                    pattern += "1"
                elif epsilon > 4:
                    pattern += "1"
                else:
                    pattern += "0"

            # print("DEBUG pattern:",pattern)

            import re
            index = [m.start() for m in re.finditer(r"1", pattern)][2]
            # print("DEBUG index:",index)
            ones = 0
            for c in pattern:
                if c == "1":
                    ones += 1
            # print("DEBUG ones:",ones)
            change_point = None
            if ones == 3:
                subset = subsets[index - 1]
                change_point = subset[2]
            else:
                subset = subsets[index - 1]
                change_point = [subset[2], subset[3]]

            # print("DEBUG change_point:",change_point)
            models = []
            # if the change point is a common point in both sets
            if isinstance(change_point, Measurement):
                index = measurements.index(change_point)
                # print("DEBUG index:",index)
                subsets = []
                subsets.append(measurements[:index])
                subsets.append(measurements[index:])

            # if the change point is a point between to point of the measurements
            else:
                index_1 = measurements.index(change_point[0])
                index_2 = measurements.index(change_point[1])
                # print("DEBUG index:",index_1, index_2)
                subsets = []
                subsets.append(measurements[:index_1])
                subsets.append(measurements[index_2:])
                # print("DEBUG subsets:",subsets)

            for subset in subsets:
                # create a constant model
                constant_hypothesis, constant_cost = self.create_constant_model(subset)
                logging.debug("Constant model: " + constant_hypothesis.function.to_string())
                logging.debug("Constant model cost: " + str(constant_cost))

                # use constant model when cost is 0
                if constant_cost == 0:
                    logging.debug("Using constant model.")
                    models.append(Model(constant_hypothesis))

                # otherwise start searching for the best hypothesis based on the PMNF
                else:
                    logging.debug("Searching for a single-parameter model.")
                    # search for the best single parameter hypothesis
                    hypotheses_generator = self.build_hypotheses(subset)
                    best_hypothesis = self.find_best_hypothesis(hypotheses_generator, constant_cost, subset,
                                                                constant_hypothesis)
                    models.append(Model(best_hypothesis))

            if isinstance(change_point, Measurement):
                intervals = [(-math.inf, change_point.coordinate[0]), (change_point.coordinate[0], math.inf)]
                change_point = [change_point]
            else:
                intervals = [(-math.inf, change_point[0].coordinate[0]), (change_point[1].coordinate[0], math.inf)]

            function = SegmentedFunction([m.hypothesis.function for m in models], intervals)
            hypothesis = SingleParameterHypothesis(function, self.use_median)
            hypothesis.compute_cost(measurements)
            hypothesis.compute_adjusted_rsquared(self.create_constant_model(measurements)[1], measurements)
            return SegmentedModel(hypothesis, models, change_point)

        else:
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
                logging.debug("Searching for a single-parameter model.")
                # search for the best single parameter hypothesis
                hypotheses_generator = self.build_hypotheses(measurements)
                best_hypothesis = self.find_best_hypothesis(hypotheses_generator, constant_cost, measurements,
                                                            constant_hypothesis)
                return Model(best_hypothesis)
