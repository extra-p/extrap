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
from extrap.entities.model import SegmentedModel
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

    theta_threshold = 0.5
    n_rss_threshold = 0.1
    epsilon_threshold = 4
    eta = 10 ** -16  # small value

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
        if len(measurements) < self.min_measurement_points * 2 - 1:
            warnings.warn("Number of measurements for a parameter needs to be at least "
                          f"{self.min_measurement_points * 2 - 1} in order to create a segmented performance model.")

        measurements = sorted(measurements, key=lambda m: m.coordinate)

        # identify subsets
        nr_subsets = len(measurements) - (self.min_measurement_points - 1)
        logging.debug("Nr. of subsets: %i", nr_subsets)
        subsets = [measurements[i:i + self.min_measurement_points] for i in range(nr_subsets)]
        logging.debug("Subsets: %s", subsets)

        # create a model for each subset
        subset_hypotheses = []
        for subset in subsets:
            # create a constant model
            constant_hypothesis, constant_cost = self.create_constant_model(subset)

            # use constant model when cost is 0
            if constant_cost == 0:
                subset_hypotheses.append(constant_hypothesis)

            # otherwise start searching for the best hypothesis based on the PMNF
            else:
                # search for the best single parameter hypothesis
                hypotheses_generator = self.build_hypotheses(subset)
                best_hypothesis = self.find_best_hypothesis(hypotheses_generator, constant_cost, subset,
                                                            constant_hypothesis)
                subset_hypotheses.append(best_hypothesis)

        logging.debug("Subset hypotheses: %s", subset_hypotheses)
        n_rss_values = np.array([abs(m.nRSS) for m in subset_hypotheses])
        theta = np.max(n_rss_values)
        logging.debug("nRSS values: %s", n_rss_values)

        epsilon_values = np.ndarray(len(subsets))
        epsilon_values[0] = -math.inf
        epsilon_values[1:] = n_rss_values[1:] / (n_rss_values[:-1] + self.eta)
        logging.debug("Epsilon values: %s", epsilon_values)

        dataset_segmented = theta > self.theta_threshold or np.nanmax(epsilon_values) > self.epsilon_threshold
        if len(epsilon_values) == 1 and math.isnan(epsilon_values[0]):
            dataset_segmented = False

        if not dataset_segmented:
            return super().create_model(measurements)

        logging.debug("Detected segmentation")

        pattern = np.zeros_like(n_rss_values, dtype=bool)
        pattern[n_rss_values >= self.n_rss_threshold] = 1
        pattern[epsilon_values > self.epsilon_threshold] = 1
        logging.debug("Segmentation pattern: %s", pattern)

        num_ones = np.sum(pattern)

        indices = [i for i, m in enumerate(pattern) if m == 1]
        index = indices[num_ones // 2]

        if num_ones == self.min_measurement_points - 2:
            subset = subsets[index]
            change_point = [subset[self.min_measurement_points // 2]]
        else:
            subset = subsets[index - 1]
            change_point = [subset[self.min_measurement_points // 2], subset[self.min_measurement_points // 2 + 1]]

        logging.debug("Change point: %s", change_point)

        # if the change point is a common point in both sets
        if len(change_point) == 1:
            index = measurements.index(change_point[0])
            final_subsets = [measurements[:index + 1], measurements[index:]]

        # if the change point is a point between two points of the measurements
        else:
            index_1 = measurements.index(change_point[0])
            index_2 = measurements.index(change_point[1])
            final_subsets = [measurements[:index_1 + 1], measurements[index_2:]]

        logging.debug("Final subsets: %s", final_subsets)

        models = [super(SegmentedModeler, self).create_model(subset) for subset in final_subsets]

        if len(change_point) == 1:
            intervals = [(-math.inf, change_point[0].coordinate[0]), (change_point[0].coordinate[0], math.inf)]
        else:
            intervals = [(-math.inf, change_point[0].coordinate[0]), (change_point[1].coordinate[0], math.inf)]

        function = SegmentedFunction([m.hypothesis.function for m in models], intervals)
        hypothesis = SingleParameterHypothesis(function, self.use_measure)
        hypothesis.compute_cost(measurements)
        hypothesis.compute_adjusted_rsquared(self.create_constant_model(measurements)[1], measurements)
        return SegmentedModel(hypothesis, models, change_point)
