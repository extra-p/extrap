"""
This file is part of the Extra-P software (https://github.com/MeaParvitas/Extra-P)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""


from entities.compound_term import CompoundTerm
from entities.single_parameter_hypothesis import SingleParameterHypothesis
from entities.single_parameter_function import SingleParameterFunction
from entities.constant_function import ConstantFunction
from entities.constant_hypothesis import ConstantHypothesis
from entities.model import Model
import logging
import copy


class SingleParameterModeler:
    """
    This class represents the modeler for single parameter functions.
    In order to create a model measurements at least 5 points are needed.
    The result is either a constant function or one based on the PMNF.
    """

    def __init__(self, experiment, modeler_id, name, coordinates=None):
        """
        Initialize SingleParameterModeler object.
        """
        self.experiment = experiment

        self.modeler_id = modeler_id

        self.name = name

        self.models = []

        # value for the minimum term contribution
        self.epsilon = 0.0005

        # minimum allowed value for a constant coefficient befor it is set to 0
        self.phi = 1e-3

        # value for the minimum number of measurement points required for modeling
        self.min_measurement_points = 5

        # use mean or median measuremnt values to calculate models
        self.median = None

        # check if logarithmic terms should be allowed
        self.allow_log_terms = self.check_parameter_values()

        # create the building blocks for the hypothesis
        self.hypotheses_building_blocks = []
        self.create_default_building_blocks()

        if coordinates != None:
            self.coordinate_ids = coordinates
            self.use_special_coordinates = True
        else:
            self.coordinate_ids = None
            self.use_special_coordinates = False

    def check_parameter_values(self):
        """
        Checkes if the parameter values are smaller than 1.
        In this case log terms are not allowed. 
        """
        coordinates = self.experiment.get_coordinates()
        for coordinate_id in range(len(coordinates)):
            for dimension in range(coordinates[coordinate_id].get_dimensions()):
                _, value = coordinates[coordinate_id].get_parameter_value(dimension)
                if value < 1.0:
                    return False
        return True

    def create_default_building_blocks(self):
        """
        Creates the default building blocks for the single parameter hypothesis
        that will be used during the search for the best hypothesis.
        """
        compound_term = CompoundTerm()
        if self.allow_log_terms == True:
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(0, 1, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(0, 1, 2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 4, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 3, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 4, 2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 3, 2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 2, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 2, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 2, 2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(2, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(2, 3, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 4, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(4, 5, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(2, 3, 2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 4, 2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 1, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 1, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 1, 2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(5, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(5, 4, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(4, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(4, 3, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 2, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 2, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 2, 2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(5, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(7, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(2, 1, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(2, 1, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(2, 1, 2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(9, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(7, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(5, 2, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(5, 2, 1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(5, 2, 2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(8, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(11, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 1, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 1, 1))
            # These were used for relearn
            """
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-0, 1, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-0, 1, -2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 4, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 3, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 4, -2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 3, -2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 2, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 2, -2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-2, 3, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-3, 4, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-2, 3, -2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-3, 4, -2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 1, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 1, -2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-5, 4, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-4, 3, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-3, 2, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-3, 2, -2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-2, 1, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-2, 1, -2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-5, 2, -1))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-5, 2, -2))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-3, 1, -1))
            """
        else:
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 2, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(2, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(4, 5, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(1, 1, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(5, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(4, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 2, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(5, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(7, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(2, 1, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(9, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(7, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(5, 2, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(8, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(11, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(3, 1, 0))
            # These were used for relearn
            """
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 2, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-2, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-3, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-4, 5, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-1, 1, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-5, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-4, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-3, 2, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-5, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-7, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-2, 1, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-9, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-7, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-5, 2, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-8, 3, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-11, 4, 0))
            self.hypotheses_building_blocks.append(compound_term.create_compound_term(-3, 1, 0))
            """

        # print the hypothesis building blocks, compound terms in debug mode
        if logging.DEBUG:
            for hypotheses_building_blocks_id in range(len(self.hypotheses_building_blocks)):
                compound_term = self.hypotheses_building_blocks[hypotheses_building_blocks_id]
                parameter = self.experiment.get_parameter(0)
                logging.debug("Compound term "+str(hypotheses_building_blocks_id)+": "+compound_term.to_string(parameter))

    def create_constant_model(self, measurements):
        """
        Creates a constant model that fits the data using a ConstantFunction.
        """

        # create a constant function
        constant_function = ConstantFunction()

        # compute the constant coefficient
        mean_model = 0
        for measurement_id in range(len(measurements)):
            if self.median == True:
                mean_model += measurements[measurement_id].get_value_median() / len(measurements)
            else:
                mean_model += measurements[measurement_id].get_value_mean() / len(measurements)

        # set the constant coefficient
        constant_function.set_constant_coefficient(mean_model)

        return constant_function

    def build_hypothesis(self, compound_term):
        """
        Builds the next hypothesis that should be analysed based on the given compound term.
        """

        # create single parameter function
        function = SingleParameterFunction()

        # add compound term
        function.add_compound_term(copy.copy(compound_term))

        return function

    def compare_hypotheses(self, old, new, measurements, coordinates):
        """
        Compares the best with the new hypothesis and decides which one is a better fit for the data.
        If the new hypothesis is better than the best one it becomes the best hypothesis.
        The choice is made based on the RSS, since this is the metric optimised by the Regression.
        """

        # get the compound terms of the new hypothesis
        compound_terms = new.get_function().get_compound_terms()

        # for all compound terms check if they are smaller than minimum allowed contribution
        for compound_terms_id in range(len(compound_terms)):

            # ignore this hypothesis, since one of the terms contributes less than epsilon to the function
            if compound_terms[compound_terms_id].get_coefficient() == 0 or new.calc_term_contribution(compound_terms_id, measurements, coordinates) < self.epsilon:
                return False

        # print smapes in debug mode
        logging.debug("next hypothesis SMAPE: "+str(new.get_SMAPE()))
        logging.debug("best hypothesis SMAPE: "+str(old.get_SMAPE()))

        return new.get_SMAPE() < old.get_SMAPE()

    def find_best_hypothesis(self, constant_hypothesis, constant_cost, measurements, coordinates):
        """
        Searches for the best single parameter hypothesis and returns it.
        """

        # create a copy of the constant hypothesis, currently it is the best hypothesis
        best_hypothesis = copy.deepcopy(constant_hypothesis)

        # search for the best hypothesis over all functions that can be build with the basic building blocks using leave one out crossvalidation
        for hypotheses_building_blocks_id in range(len(self.hypotheses_building_blocks)):

            # create next function that will be analyzed
            compound_term = self.hypotheses_building_blocks[hypotheses_building_blocks_id]
            next_function = self.build_hypothesis(compound_term)

            # create single parameter hypothesis from function
            next_hypothesis = SingleParameterHypothesis(next_function, self.median)

            # cycle through points and leave one out per iteration
            for element_id in range(len(measurements)):

                # copy measurements and coordinates to create the training sets
                training_measurements = copy.deepcopy(measurements)
                training_coordinates = copy.deepcopy(coordinates)

                # remove one element from both sets
                training_measurements.pop(element_id)
                training_coordinates.pop(element_id)

                # validation sets
                validation_measurement = copy.deepcopy(measurements[element_id])
                validation_coordinate = copy.deepcopy(coordinates[element_id])

                # compute the model coefficients based on the training data
                next_hypothesis.compute_coefficients(training_measurements, training_coordinates)

                # check if the constant coefficient should actually be 0
                next_hypothesis.clean_constant_coefficient(self.phi, training_measurements)

                # compute the cost of the single parameter model for the validation data
                next_hypothesis.compute_cost(training_measurements, validation_measurement, validation_coordinate)

            # compute the model coefficients using all data
            next_hypothesis.compute_coefficients(measurements, coordinates)
            logging.debug("Single parameter model "+str(hypotheses_building_blocks_id)+": "+next_hypothesis.function.to_string(self.experiment.get_parameter(0)))

            # compute the AR2 for the hypothesis
            next_hypothesis.compute_adjusted_rsquared(constant_cost, measurements)

            # check if hypothesis is valid
            if next_hypothesis.is_valid() == False:
                logging.debug("Numeric imprecision found. Model is invalid and will be ignored.")

            # compare the new hypothesis with the best hypothesis
            elif self.compare_hypotheses(best_hypothesis, next_hypothesis, measurements, coordinates):
                best_hypothesis = copy.deepcopy(next_hypothesis)

        return best_hypothesis

    def create_model(self, callpath_id, metric_id, median):
        """
        Create a model for the given callpath and metric using the given data.
        """
        # set to use mean or median measurement values
        self.median = median

        # select the measurements by callpath_id and metric_id
        all_measurements = self.experiment.get_measurements()
        measurements = []
        for measurement_id in range(len(all_measurements)):
            if all_measurements[measurement_id].get_callpath_id() == callpath_id and all_measurements[measurement_id].get_metric_id() == metric_id:
                measurements.append(all_measurements[measurement_id])

        # check if the number of measurements satisfies the requirements of the modeler (>=5)
        if len(measurements) < self.min_measurement_points:
            logging.error("Number of measurements for a parameter needs to be at least 5 in order to create a performance model.")
            raise RuntimeError("Number of measurements for a parameter needs to be at least 5 in order to create a performance model.")
            return None

        # get the coordinates for modeling
        coordinates = self.experiment.get_coordinates()

        # create a constant function
        constant_function = self.create_constant_model(measurements)

        # create a constant hypothesis from the constant function
        constant_hypothesis = ConstantHypothesis(constant_function, self.median)
        logging.debug("Constant model: "+constant_hypothesis.function.to_string())

        # compute cost of the constant model
        constant_hypothesis.compute_cost(measurements)
        constant_cost = constant_hypothesis.get_RSS()
        logging.debug("Constant model cost: "+str(constant_cost))

        # use constat model when cost is 0
        if constant_cost == 0:
            logging.debug("Using constant model.")
            model = Model(constant_hypothesis, callpath_id, metric_id)
            self.models.append(model)

        # otherwise start searching for the best hypothesis based on the pmnf
        else:
            logging.debug("Searching for a single parameter model.")

            # search for the best single parmater hypothesis
            best_hypothesis = self.find_best_hypothesis(constant_hypothesis, constant_cost, measurements, coordinates)
            model = Model(best_hypothesis, callpath_id, metric_id)
            self.models.append(model)

    def get_models(self):
        return self.models

    def get_model(self, callpath_id, metric_id):
        for model_id in range(len(self.models)):
            model = self.models[model_id]
            if model.get_callpath_id() == callpath_id and model.get_metric_id() == metric_id:
                return model
        return None
