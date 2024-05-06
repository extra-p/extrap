# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
import warnings
from collections import namedtuple
from operator import attrgetter
from typing import List, Tuple, Sequence

from extrap.entities.fraction import Fraction
from extrap.entities.functions import SingleParameterFunction
from extrap.entities.hypotheses import SingleParameterHypothesis
from extrap.entities.measurement import Measure
from extrap.entities.model import Model
from extrap.entities.terms import CompoundTerm
from extrap.modelers.abstract_modeler import SingularModeler
from extrap.modelers.single_parameter.abstract_base import AbstractSingleParameterModeler

SearchState = namedtuple('SearchState', ['left', 'center', 'right'])


class SingleParameterRefiningHypothesis(SingleParameterHypothesis):

    def __init__(self, function, use_measure, partition_index):
        super().__init__(function, use_measure)
        self.partition_index = partition_index


class RefiningModeler(SingularModeler, AbstractSingleParameterModeler):
    """
    Implementation of the refinement modeler
    """
    NAME = 'Refining'
    DESCRIPTION = "Modeler for single-parameter models; refines the search-space iteratively."

    def __init__(self):
        super().__init__(use_measure=Measure.MEAN)
        self.epsilon = 0.0005

        # init variables for the hypothesis creation
        self.max_log_expo = 2
        self.max_poly_expo = 5
        self.acceptance_threshold = 1.5
        self.termination_threshold = 2.0
        self.nonconstancy_threshold = 1.3

    def create_model(self, measurements):

        # check if the number of measurements satisfies the requirements of the modeler (>=5)
        if len(measurements) < 5:
            warnings.warn(
                "Number of measurements needs to be at least 5 in order to create a performance model.")
            # return None

        # compute a constant model
        constant_hypothesis, constant_cost = self.create_constant_model(measurements)

        # use constant model when cost is 0
        if constant_cost == 0:
            return Model(constant_hypothesis)

        # otherwise start searching for the best hypothesis based on the pmnf

        # determine all exponents
        # TODO could be set via options
        allow_log = self.allow_log_terms and self.are_measurements_log_capable(measurements)
        poly_expos = range(self.max_poly_expo + 1)
        max_log_expo = self.max_log_expo if allow_log else 0
        log_expos = range(max_log_expo + 1)

        slices: List[Tuple[Sequence, Sequence]] = [(poly_expos, [l]) for l in log_expos]  # beta slices
        if allow_log:
            slices.append(([0], log_expos))  # alpha slice

        # create coarse hypotheses
        hypotheses = [self.find_best_hypothesis(self._build_hypotheses_generator(slice, ignore_constant=True),
                                                constant_cost, measurements) for slice in slices]

        # determine exponents for initial state
        state_per_slice = self._determine_initial_state(hypotheses, slices)

        # execute iterative refinement
        best_hypothesis = self.iterative_refinement(hypotheses, state_per_slice, slices, constant_cost, measurements)

        # determine if improvement over constant model is enough
        term_contribution = best_hypothesis.calc_term_contribution(best_hypothesis.function.compound_terms[0],
                                                                   measurements)
        improvement = constant_hypothesis.SMAPE / best_hypothesis.SMAPE
        if improvement < self.nonconstancy_threshold or term_contribution < self.epsilon:
            best_hypothesis = constant_hypothesis

        return Model(best_hypothesis)

    @staticmethod
    def _determine_initial_state(hypotheses, slices):
        state_per_slice = []
        for hypothesis, slice in zip(hypotheses, slices):
            i = hypothesis.partition_index
            exponents = slice[0 if len(slice[0]) > 1 else 1]  # switch between log and polynomial
            # determine exponents for initial state
            expo_l = exponents[i - 1 if i - 1 >= 0 else 0]
            expo_c = exponents[i]
            expo_r = exponents[i + 1 if i + 1 < len(exponents) else -1]
            if expo_l == expo_r:
                # prevent empty search space
                expo_r += 1
            state_per_slice.append(SearchState(Fraction(expo_l), Fraction(expo_c), Fraction(expo_r)))
        return state_per_slice

    def iterative_refinement(self, hypotheses: List[SingleParameterRefiningHypothesis],
                             state_per_slice: List[SearchState], slices,
                             constant_cost, measurements):
        if self.compare_with_RSS:
            selector = attrgetter('RSS')
        else:
            selector = attrgetter('SMAPE')
        best_hypotheses = hypotheses
        best_hypotheses_step = copy.copy(hypotheses)
        best_hypotheses_previous = copy.copy(hypotheses)
        current_acceptance_threshold = self.acceptance_threshold
        for i in range(10):
            for s, slice in enumerate(slices):
                old_state = state_per_slice[s]  # contains old exponents

                # calculates new exponents
                state = SearchState(
                    old_state.left.compute_mediant(old_state.center),
                    old_state.center,
                    old_state.center.compute_mediant(old_state.right)
                )

                # create new partition
                if len(slice[0]) > 1:
                    partition = (state, slice[1])
                else:
                    partition = (slice[0], state)

                # determine best partition
                hypotheses_generator = self._build_hypotheses_generator(partition, ignore_constant=True)
                best_hypotheses_step[s] = self.find_best_hypothesis(hypotheses_generator, constant_cost, measurements)
                partition_index = best_hypotheses_step[s].partition_index

                # clips search space
                if partition_index < 1:
                    state = SearchState(old_state.left,
                                        state.left,
                                        state.center)
                elif partition_index > 1:
                    state = SearchState(state.center,
                                        state.right,
                                        old_state.right)
                state_per_slice[s] = state

            # determine best hypothesis of step
            best_hypothesis_step: SingleParameterHypothesis = min(best_hypotheses_step, key=selector)
            global_best_hypothesis = min(best_hypotheses, key=selector)
            if global_best_hypothesis.SMAPE / best_hypothesis_step.SMAPE >= current_acceptance_threshold:
                best_hypotheses.append(best_hypothesis_step)
                current_acceptance_threshold = self.acceptance_threshold
            else:
                current_acceptance_threshold *= self.acceptance_threshold

            minimal_slice_improvement = max(best_hypotheses_previous[s].SMAPE / best_hypotheses_step[s].SMAPE
                                            for s in range(len(slices)))
            if minimal_slice_improvement >= self.termination_threshold:
                best_hypotheses_previous, best_hypotheses_step = best_hypotheses_step, best_hypotheses_previous
            else:
                break

        return min(best_hypotheses, key=selector)

    def _build_hypotheses_generator(self, partition, ignore_constant=False):
        p_partition, l_partition = partition

        def build_hypothesis(p, l, i):
            simple_function = SingleParameterFunction(CompoundTerm.create(p, l))
            return SingleParameterRefiningHypothesis(simple_function, self.use_measure, i)

        if len(p_partition) > 1:
            l = l_partition[0]
            return (build_hypothesis(p, l, i) for i, p in enumerate(p_partition)
                    if not (ignore_constant and p == 0 and l == 0))
        else:
            p = p_partition[0]
            return (build_hypothesis(p, l, i) for i, l in enumerate(l_partition)
                    if not (ignore_constant and p == 0 and l == 0))
