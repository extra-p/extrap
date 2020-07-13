import copy
import logging
import warnings
from collections import namedtuple
from itertools import product
from typing import List, Tuple, Sequence

from entities.model import Model
from entities.terms import CompoundTerm
from entities.fraction import Fraction
from entities.functions import SingleParameterFunction
from entities.hypotheses import SingleParameterHypothesis

from modelers.abstract_modeler import LegacyModeler

# class SearchState:
#
#     def __init__(self, left, center, right, hypothesis):
#         self.left = left
#         self.center = center
#         self.right = right
#         self.hypothesis = hypothesis

SearchState = namedtuple('SearchState', ['left', 'center', 'right'])


class RefiningModeler(LegacyModeler):
    """
    Sample implementation of the refinement modeler, not finished yet!
    """
    NAME = 'Refining'

    def __init__(self):
        super().__init__(use_median=False)
        self.epsilon = 0.0005

        # check if logarithmic terms should be allowed
        self.allow_log_terms = True

        # init variables for the hypothesis creation
        self.max_log_expo = 2
        self.max_poly_expo = 5
        self.acceptance_threshold = 1.5
        self.termination_threshold = 2.0
        self.nonconstancy_threshold = 1.3

    @staticmethod
    def check_parameter_values(coordinates):
        # analyze the parameter values to see if log terms should be allowed or not
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
            mean_model += measurements[measurements_id].value(self.use_median) / \
                          len(measurements)
        constant_function.set_constant_coefficient(mean_model)
        return constant_function

    def compare_hypotheses(self, old, new, measurements):
        if old is None:
            return True
        compound_terms = new.get_function().get_compound_terms()
        for term in compound_terms:
            if term.coefficient == 0 or new.calc_term_contribution(term, measurements) < self.epsilon:
                # This hypothesis is not worth considering, because one of the terms does not actually contribute to the
                # function value in a sufficient way. We have already seen another hypothesis which contains the
                # remaining terms, so we can ignore this one.
                return False
        logging.debug("next hypothesis rss: " + str(new.get_RSS()))
        logging.debug("best hypothesis rss: " + str(old.get_RSS()))
        return new.get_RSS() < old.get_RSS()

    def create_model(self, measurements):

        # TODO: it would be best to have this check when doing the fileio... and remove it here...
        # check if the number of measurements satisfies the reuqirements of the modeler (>=5)
        if len(measurements) < 5:
            warnings.warn(
                "Number of measurements needs to be at least 5 in order to create a performance model.")
            # return None

        # get the coordinates for modeling
        coordinates = [m.coordinate for m in measurements]

        # initialize current term count
        self.current_term_count = 0

        # compute a constant model
        constant_function = self.create_constant_model(measurements)

        # create a hypothesis from the model
        constant_hypothesis = SingleParameterHypothesis(constant_function, self.use_median)

        # compute constant hypothesis cost
        constant_hypothesis.compute_cost_all_points(measurements)

        # get constant model cost
        constant_cost = constant_hypothesis.get_RSS()

        # use constant model when cost is 0
        if constant_cost == 0:
            return Model(constant_hypothesis)

        # otherwise start searching for the best hypothesis based on the pmnf
        best_hypotheses = [constant_hypothesis]

        # determine all exponents
        # TODO could be set via options
        allow_log = self.check_parameter_values(coordinates)
        poly_expos = range(self.max_poly_expo + 1)
        max_log_expo = self.max_log_expo if allow_log else 0
        log_expos = range(max_log_expo + 1)

        slices: List[Tuple[Sequence, Sequence]] = [(poly_expos, [l]) for l in log_expos]  # beta slices
        if allow_log:
            slices.append(([0], log_expos))  # alpha slice

        # create coarse hypotheses
        hypotheses, expo_idx = tuple(
            zip(*[self.find_best_hypotheses(slice, measurements, ignore_constant=True) for slice in slices]))

        state_per_slice = []
        for i, slice in zip(expo_idx, slices):
            expos = slice[0 if len(slice[0]) > 1 else 1]
            expo_l = expos[0 if i - 1 < 0 else i - 1]
            expo_r = expos[len(expos) - 1 if i + 1 >= len(expos) else i + 1]
            if expo_l == expo_r:
                expo_r += 1
            state_per_slice.append(SearchState(Fraction(expo_l), Fraction(expos[i]), Fraction(expo_r)))

        best_hypothesis = self.iterative_refinement(hypotheses, state_per_slice, slices, measurements)

        term_contribution = best_hypothesis.calc_term_contribution(best_hypothesis.function.compound_terms[0],
                                                                   measurements)
        improvement = constant_hypothesis.SMAPE / best_hypothesis.SMAPE
        if improvement < self.nonconstancy_threshold or term_contribution < self.epsilon:
            best_hypothesis = constant_hypothesis

        return Model(best_hypothesis)

    def iterative_refinement(self, hypotheses, state_per_slice, slices, measurements):
        hypotheses: List[SingleParameterHypothesis] = list(hypotheses)
        best_hypotheses = hypotheses
        best_hypotheses_step = hypotheses
        best_hypotheses_previous = hypotheses

        for i in range(10):
            for s, slice in enumerate(slices):
                expos_old = state_per_slice[s]

                expos = SearchState(
                    expos_old.left.compute_mediant(expos_old.center),
                    expos_old.center,
                    expos_old.center.compute_mediant(expos_old.right)
                )

                if len(slice[0]) > 1:
                    expo_slice = (expos, slice[1])
                else:
                    expo_slice = (slice[0], expos)

                best_hypotheses_step[s], idx = self.find_best_hypotheses(expo_slice, measurements)

                if idx < 1:
                    expos = SearchState(expos_old.left,
                                        expos.left,
                                        expos.center)
                elif idx > 1:
                    expos = SearchState(expos.center,
                                        expos.right,
                                        expos_old.right)

                state_per_slice[s] = expos
            best_hypothesis_step: SingleParameterHypothesis = min(best_hypotheses_step, key=lambda x: x.SMAPE)
            if best_hypotheses[-1].SMAPE / best_hypothesis_step.SMAPE >= self.acceptance_threshold:
                best_hypotheses.append(best_hypothesis_step)

            minimal_slice_improvement = min(best_hypotheses_step[s].SMAPE / best_hypotheses_previous[s].SMAPE
                                            for s in range(len(slices)))
            if i == 0 or minimal_slice_improvement >= self.termination_threshold:
                best_hypotheses_previous, best_hypotheses_step = best_hypotheses_step, best_hypotheses_previous
            else:
                break

        return min(best_hypotheses, key=lambda x: x.SMAPE)

    def build_hypothesis(self, p, l, measurements):
        simple_function = SingleParameterFunction(CompoundTerm.create(p, l))
        hypothesis = SingleParameterHypothesis(simple_function, self.use_median)
        hypothesis.compute_coefficients(measurements)
        hypothesis.compute_cost_all_points(measurements)
        return hypothesis

    def find_best_hypotheses(self, slice, measurements, ignore_constant=False):
        p_slice, l_slice = slice
        best_hypothesis, expo_idx = None, 0

        def check_hypothesis(p, l, i):
            nonlocal best_hypothesis, expo_idx
            if ignore_constant and p == 0 and l == 0:
                return
            hypothesis = self.build_hypothesis(p, l, measurements)
            logging.debug(f"Hypothesis: {hypothesis.function}")

            if not hypothesis.is_valid():
                logging.debug("Numeric imprecision found. Model is invalid and will be ignored.")
            elif self.compare_hypotheses(best_hypothesis, hypothesis, measurements):
                best_hypothesis, expo_idx = hypothesis, i

        if len(p_slice) > 1:
            l = l_slice[0]
            for i, p in enumerate(p_slice):
                check_hypothesis(p, l, i)
        else:
            p = p_slice[0]
            for i, l in enumerate(l_slice):
                check_hypothesis(p, l, i)
        return best_hypothesis, expo_idx
