import copy
import logging

from entities.terms import CompoundTerm
from entities.fraction import Fraction
from entities.functions import SingleParameterFunction
from entities.hypotheses import SingleParameterHypothesis

from modelers.abstract_modeler import LegacyModeler


class SearchState:

    def __init__(self, left, center, right, hypothesis):
        self.left = left
        self.center = center
        self.right = right
        self.hypothesis = hypothesis


class RefiningModeler():
    """
    Sample implementation of the refinement modeler, not finished yet!
    """

    def __init__(self, experiment, options):
        self.experiment = experiment
        self.epsilon = 0.0005
        self.options = options

        # check if logarithmic terms should be allowed
        self.allow_log_terms = self.check_parameter_values()

        # init variables for the hypothesis creation
        self.max_log_expo = 2
        self.max_poly_expo = 5
        self.acceptance_threshold = 1.5
        self.termination_threshold = 2.0
        self.nonconstancy_threshold = 1.3

        # create the building blocks for the hypothesis
        self.hypotheses_building_blocks = []
        self.generate_default_building_blocks()

        self.current_hypothesis = 0
        self.current_hypothesis_building_block_vector = []
        self.current_term_count = 0

    def check_parameter_values(self):
        # analyze the parameter values to see if log terms should be allowed or not
        coordinates = self.experiment.get_coordinates()
        for coordinate_id in range(len(coordinates)):
            for dimension in range(coordinates[coordinate_id].get_dimensions()):
                parameter, value = coordinates[coordinate_id].get_parameter_value(
                    dimension)
                if value < 1:
                    return False
        return True

    def create_constant_model(self, measurements):
        constant_function = SingleParameterFunction()
        mean_model = 0
        for measurements_id in range(len(measurements)):
            mean_model += measurements[measurements_id].get_value() / \
                len(measurements)
        constant_function.set_constant_coefficient(mean_model)
        return constant_function

    def init_search_space(self):
        self.current_hypothesis = 0
        # resize this vector and add empty 0 elements
        for _ in range(self.current_term_count):
            self.current_hypothesis_building_block_vector.append(0)
        counter = len(self.current_hypothesis_building_block_vector) - 1
        while counter >= 0:
            self.current_hypothesis_building_block_vector[len(
                self.current_hypothesis_building_block_vector) - 1 - counter] = counter
            self.current_hypothesis += counter * pow(float(len(self.hypotheses_building_blocks)), int(
                len(self.current_hypothesis_building_block_vector) - 1 - counter))
            counter -= 1
        self.current_hypothesis = int(self.current_hypothesis)

    def build_current_hypothesis(self):
        simple_function = SingleParameterFunction()
        for i in range(len(self.current_hypothesis_building_block_vector)):
            print("current_hypothesis_building_block_vector: ",
                  self.current_hypothesis_building_block_vector[i])
            simple_function.add_compound_term(
                self.hypotheses_building_blocks[self.current_hypothesis_building_block_vector[i]])
        return simple_function

    def compare_hypotheses(self, old, new, measurements, coordinates):
        compound_terms = new.get_function().get_compound_terms()
        for i in range(len(compound_terms)):
            if compound_terms[i].get_coefficient() == 0 or new.calc_term_contribution(i, measurements, coordinates) < self.epsilon:
                # This hypothesis is not worth considering, because one of the terms does not actually contribute to the
                # function value in a sufficient way. We have already seen another hypothesis which contains the remaining
                # terms, so we can ignore this one.
                return False
        logging.debug("next hypothesis rss: "+str(new.get_RSS()))
        logging.debug("best hypothesis rss: "+str(old.get_RSS()))
        return new.get_SMAPE() < old.get_SMAPE()

    def next_hypothesis(self):
        not_found = True
        while not_found:
            not_found = False
            self.current_hypothesis += 1
            index = self.current_hypothesis
            if index >= pow(float(len(self.hypotheses_building_blocks)), int(len(self.current_hypothesis_building_block_vector))):
                return False
            for i in range(len(self.current_hypothesis_building_block_vector)):
                self.current_hypothesis_building_block_vector[i] = index % len(
                    self.hypotheses_building_blocks)
                index = index / len(self.hypotheses_building_blocks)
                for j in range(i):
                    if self.current_hypothesis_building_block_vector[i] == self.current_hypothesis_building_block_vector[j]:
                        not_found = True
            if not not_found:
                return True

    def find_best_hypothesis(self, constant_hypothesis, measurements, coordinates):
        best_hypothesis = copy.deepcopy(constant_hypothesis)
        self.init_search_space()

        while True:

            next_function = self.build_current_hypothesis()

            next_hypothesis = SingleParameterHypothesis(next_function)

            logging.debug(
                "Next Hypothesis:"+next_hypothesis.function.to_string(self.experiment.get_parameter(0)))

            next_hypothesis.compute_coefficients(measurements, coordinates)

            logging.debug(
                "Next Hypothesis:"+next_hypothesis.function.to_string(self.experiment.get_parameter(0)))

            next_hypothesis.compute_cost(measurements, coordinates)

            next_hypothesis.compute_cross_validation(measurements, coordinates)

            logging.debug("Best Hypothesis:", best_hypothesis.function.to_string(
                self.experiment.get_parameter(0)))

            if next_hypothesis.is_valid() == False:
                logging.debug(
                    "Numeric imprecision found. Model is invalid and will be ignored.")
            elif self.compare_hypotheses(best_hypothesis, next_hypothesis, measurements, coordinates):
                best_hypothesis = copy.deepcopy(next_hypothesis)

            if self.next_hypothesis() == False:
                break

        return best_hypothesis

    def create_model(self, callpath_id, metric_id):
        # select the measurements by callpath_id and metric_id
        all_measurements = self.experiment.get_measurements()
        measurements = []
        for i in range(len(all_measurements)):
            if all_measurements[i].get_callpath_id() == callpath_id and all_measurements[i].get_metric_id() == metric_id:
                measurements.append(all_measurements[i])

        # TODO: it would be best to have this check when doing the fileio... and remove it here...
        # check if the number of measurements satisfies the reuqirements of the modeler (>=5)
        if len(measurements) < 5:
            logging.warning(
                "Number of measurements needs to be at least 5 in order to create a performance model.")
            return None

        # get the coordinates for modeling
        coordinates = self.experiment.get_coordinates()

        # initialize current term count
        self.current_term_count = 0

        # compute a constant model
        constant_function = self.create_constant_model(measurements)

        # create a hypothesis from the model
        constant_hypothesis = SingleParameterHypothesis(constant_function)

        # compute constant hypothesis cost
        constant_hypothesis.compute_cost(measurements, coordinates)

        # get constant model cost
        constant_cost = constant_hypothesis.get_RSS()

        # use constat model when cost is 0
        if constant_cost == 0:
            return constant_hypothesis

        # otherwise start searching for the best hypothesis based on the pmnf
        states = []
        global_best_hypothesis_index = -1

        for slice_idx in range(self.max_log_expo+1):
            # init for constant model
            current_best_exponent = Fraction(0, 1)

            self.current_term_count += 1
            best_hypothesis = self.find_best_hypothesis(
                constant_hypothesis, measurements, coordinates)
            best_hypothesis.compute_adjusted_rsquared(
                constant_cost, measurements)

            # debug
            # print(self.current_term_count)

            # while self.current_term_count < self.max_term_count:

            #self.current_term_count += 1
            #new_best_hypothesis = self.find_best_hypothesis(best_hypothesis, measurements, coordinates)
            #new_best_hypothesis.compute_adjusted_rsquared(constant_cost, measurements)

            # if new_best_hypothesis.get_AR2() > best_hypothesis.get_AR2():
            #    best_hypothesis = copy.deepcopy(new_best_hypothesis)

        return best_hypothesis
