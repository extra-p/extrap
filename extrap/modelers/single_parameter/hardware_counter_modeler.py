# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import copy
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import numpy

from extrap.entities.annotations.counter_selection_annotation import CounterSelectionAnnotation
from extrap.entities.functions import SingleParameterFunction, Function
from extrap.entities.hypotheses import SingleParameterHypothesis, ConstantHypothesis, Hypothesis
from extrap.entities.measurement import Measurement, Measure
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.entities.terms import Term, MultiParameterTerm, SegmentedTerm
from extrap.modelers.abstract_modeler import AbstractModeler
from extrap.modelers.modeler_options import modeler_options
from extrap.modelers.single_parameter.basic import SingleParameterModeler
from extrap.util.exceptions import RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS


@modeler_options
class AbstractHardwareCounterModeler(AbstractModeler, ABC):
    number_used_counters = modeler_options.add(3, int, range=range(1, 5))
    weighted_terms = modeler_options.add(False, bool, 'Uses all counters.')
    _instruction_counters = modeler_options.add("", str, "List of counters that are used in the modeling process, "
                                                         "separated by comma.",
                                                on_change=lambda self, v: self._update_counters(v))
    negative_coefficients = modeler_options.add(False, bool)
    counter_selection_strategy = modeler_options.add(
        'top_exponent', str,
        "How counters are chosen and combined into the runtime model: 'top_value' (default, counters with the "
        "largest measured values), 'top_exponent' (counters whose fitted model has the steepest growth term), "
        "'sum_counters' (sum all selected counters' measurements and fit one model to that sum), "
        "'greedy_exponent' (add counters steepest-growth-first, keeping only the ones that actually improve the "
        "time model's fit).",
        range=['top_value', 'top_exponent', 'sum_counters', 'greedy_exponent', 'weighted_instructions'])
    track_coordinates = modeler_options.add(True, bool)
    auto_filter_counters = modeler_options.add(False, bool,
                                               "Tries a heuristic filter duplicated counters. "
                                               "It is always better to provide a smaller set of meaningful counters instead.")

    source_modeler: AbstractModeler

    def __init__(self, use_measure, source_modeler, **kwargs):
        """
        Initialize HardwareCounterModeler object.
        """

        self.source_modeler = source_modeler
        super().__init__(use_measure=use_measure, **kwargs)

        self.min_measurement_points = 5
        self.counter_list = [
            "PAPI_LST_INS",
            "PAPI_LD_INS",
            "PAPI_SR_INS",
            "PAPI_BR_INS",
            "PAPI_BR_UCN",
            "PAPI_BR_CN",
            "PAPI_BR_TKN",
            "PAPI_BR_NTK",
            "PAPI_BR_PRC",
            "PAPI_BR_MSP",
            "PAPI_DP_OPS",
            "PAPI_SP_OPS",
            "PAPI_FP_OPS",
            "PAPI_FP_INS",
            "PAPI_VEC_DP",
            "PAPI_VEC_SP",
            "PAPI_VEC_INS",
            "PAPI_L1_TCM",
            "PAPI_L1_ICM",
            "PAPI_L1_DCM",
            "PAPI_L1_TCA",
            "PAPI_L1_ICA",
            "PAPI_L1_DCA",
            "PAPI_L1_DCR",
            "PAPI_L1_DCW",
            "PAPI_L2_TCM",
            "PAPI_L2_ICM",
            "PAPI_L2_DCM",
            "PAPI_L2_TCA",
            "PAPI_L2_ICA",
            "PAPI_L2_DCA",
            "PAPI_L2_DCR",
            "PAPI_L2_DCW",
            "PAPI_TLB_TL",
            "PAPI_TLB_IM",
            "PAPI_TLB_DM",
            "PAPI_TLB_SD",
            # L3 cache: same hierarchy pattern as L1/L2 above.
            "PAPI_L3_TCM",
            "PAPI_L3_ICM",
            "PAPI_L3_DCM",
            "PAPI_L3_TCA",
            "PAPI_L3_ICA",
            "PAPI_L3_DCA",
            "PAPI_L3_DCR",
            "PAPI_L3_DCW",
            # Memory stall cycles: same read/write split pattern as *_DCA.
            "PAPI_MEM_SCY",
            "PAPI_MEM_RCY",
            "PAPI_MEM_WCY",
            # Cache coherence (MESI-style) events. No hardcoded redundancy
            # rule below -- unlike the groups above, these do not have a
            # verified, exact additive relationship to each other or to the
            # *_TCA/*_TCM counters, so they are left for
            # `use_empirical_redundancy_detection` to evaluate case by case.
            "PAPI_CA_SNP",
            "PAPI_CA_SHR",
            "PAPI_CA_CLN",
            "PAPI_CA_INV",
            "PAPI_CA_ITV",
            # General instruction/cycle/stall counters. Several of these are
            # plausibly related to counters above (e.g. PAPI_TOT_INS to the
            # instruction-type counters, PAPI_FMA_INS to the FP counters),
            # but not via a *verified* exact identity (categories can
            # overlap or be architecture-specific, e.g. an FMA may already
            # be counted within PAPI_DP_OPS on some CPUs), so no hardcoded
            # rule is added for them; `use_empirical_redundancy_detection`
            # can still catch an exact relationship where one actually
            # holds on a given system.
            "PAPI_INT_INS",
            "PAPI_TOT_CYC",
            "PAPI_TOT_IIS",
            "PAPI_TOT_INS",
            "PAPI_RES_STL",
            "PAPI_FP_STAL",
            "PAPI_STL_ICY",
            "PAPI_FUL_ICY",
            "PAPI_STL_CCY",
            "PAPI_FUL_CCY",
            "PAPI_HW_INT",
            "PAPI_SYC_INS",
            "PAPI_FMA_INS",
            "PAPI_BTAC_M",
            "PAPI_PRF_DM",
        ]
        self.metrics_to_improve = ['time']
        self.total_count_metric = Metric('PAPI_TOT_INS')
        self.epsilon_minimum_term_contribution = 0.0005  # value for the minimum term contribution

    def _update_counters(self, value: str):
        self.counter_list = [v.strip() for v in value.split(',')]

    def determine_relevant_counters(self, callpath_models):
        actual_counter_list = set(self.counter_list)
        counters_in_experiment = set(str(m.metric) for m in callpath_models)
        actual_counter_list.intersection_update(counters_in_experiment)

        if not self.auto_filter_counters:
            return actual_counter_list

        def check_and_remove(check_for, remove, also_reverse=False):
            if all(counter in actual_counter_list for counter in check_for):
                for counter in remove:
                    if counter not in check_for:
                        try:
                            actual_counter_list.remove(counter)
                        except KeyError:
                            pass
            if also_reverse:
                check_and_remove(remove, check_for, also_reverse=False)

        check_and_remove(['PAPI_BR_CN', 'PAPI_BR_UCN'], ['PAPI_BR_INS'], True)
        check_and_remove(['PAPI_LD_INS', 'PAPI_SR_INS'], ['PAPI_LST_INS'], True)
        all_fp_counters = ["PAPI_DP_OPS", "PAPI_SP_OPS", 'PAPI_FP_OPS', 'PAPI_FP_INS', 'PAPI_VEC_INS', "PAPI_VEC_SP",
                           "PAPI_VEC_DP"]
        check_and_remove(["PAPI_DP_OPS", "PAPI_SP_OPS"], all_fp_counters)
        check_and_remove(["PAPI_VEC_DP", "PAPI_VEC_SP"], all_fp_counters)
        check_and_remove(['PAPI_FP_OPS'], all_fp_counters)
        check_and_remove(['PAPI_FP_INS'], all_fp_counters)
        check_and_remove(['PAPI_VEC_INS'], all_fp_counters)

        # Conditional branches can be decomposed in two independent ways: by
        # outcome (taken vs. not-taken) or by prediction correctness
        # (correctly predicted vs. mispredicted). Both pairs are redundant
        # with PAPI_BR_CN individually, but not with each other (they
        # partition the same total differently), so both may still be kept
        # simultaneously if present.
        check_and_remove(['PAPI_BR_TKN', 'PAPI_BR_NTK'], ['PAPI_BR_CN'], True)
        check_and_remove(['PAPI_BR_PRC', 'PAPI_BR_MSP'], ['PAPI_BR_CN'], True)

        # Cache hierarchy: at a given level, total accesses/misses are the
        # sum of the instruction- and data-side accesses/misses, and data
        # accesses are themselves the sum of data reads and data writes.
        # Resolved top-down (the aggregate-of-aggregates first, its own
        # constituents after), so an aggregate that already absorbed a
        # higher-level check is not redundantly expanded a second time.
        # Note: unlike the branch/load-store/FP rules above, these
        # relationships are conventions used in PAPI's own preset
        # definitions rather than something guaranteed to hold exactly on
        # every architecture/PAPI component (some platforms measure a total
        # like PAPI_L1_TCM directly rather than deriving it from its
        # instruction/data parts). Enabling
        # `use_empirical_redundancy_detection` complements this fixed list
        # by verifying redundancy against actually measured values instead
        # of assuming these definitions hold.
        check_and_remove(['PAPI_L1_ICM', 'PAPI_L1_DCM'], ['PAPI_L1_TCM'], True)
        check_and_remove(['PAPI_L1_ICA', 'PAPI_L1_DCA'], ['PAPI_L1_TCA'], True)
        check_and_remove(['PAPI_L1_DCR', 'PAPI_L1_DCW'], ['PAPI_L1_DCA'], True)
        check_and_remove(['PAPI_L2_ICM', 'PAPI_L2_DCM'], ['PAPI_L2_TCM'], True)
        check_and_remove(['PAPI_L2_ICA', 'PAPI_L2_DCA'], ['PAPI_L2_TCA'], True)
        check_and_remove(['PAPI_L2_DCR', 'PAPI_L2_DCW'], ['PAPI_L2_DCA'], True)
        check_and_remove(['PAPI_L3_ICM', 'PAPI_L3_DCM'], ['PAPI_L3_TCM'], True)
        check_and_remove(['PAPI_L3_ICA', 'PAPI_L3_DCA'], ['PAPI_L3_TCA'], True)
        check_and_remove(['PAPI_L3_DCR', 'PAPI_L3_DCW'], ['PAPI_L3_DCA'], True)

        # Total TLB misses are the sum of instruction- and data-side TLB
        # misses on architectures that expose both separately (same
        # architecture-dependence caveat as the cache counters above).
        check_and_remove(['PAPI_TLB_IM', 'PAPI_TLB_DM'], ['PAPI_TLB_TL'], True)

        # Cycles stalled waiting for memory are split into read- and
        # write-stall cycles, mirroring the *_DCA = *_DCR + *_DCW pattern.
        check_and_remove(['PAPI_MEM_RCY', 'PAPI_MEM_WCY'], ['PAPI_MEM_SCY'], True)

        if not actual_counter_list and str(self.total_count_metric) in counters_in_experiment:
            actual_counter_list.add(str(self.total_count_metric))

        return actual_counter_list

    @staticmethod
    def _term_order(term) -> float:
        """Sum of the exponents making up a (possibly multi-parameter) term -- its growth order."""
        if isinstance(term, MultiParameterTerm):
            return sum(AbstractHardwareCounterModeler._term_order(t) for _, t in term.parameter_term_pairs)
        if isinstance(term, SegmentedTerm):
            return max((AbstractHardwareCounterModeler._term_order(t)
                        for segment in term.segments for t in segment.compound_terms), default=0)
        return sum(st.exponent if st.term_type == "polynomial" else st.exponent * 0.01
                   for st in term.simple_terms)

    @classmethod
    def _max_exponent(cls, model: Model) -> float:
        """The steepest growth order among a model's compound terms (higher = grows faster)."""
        terms = model.hypothesis.function.compound_terms
        if not terms:
            return float('-inf')
        return max(cls._term_order(term) for term in terms)

    def determine_relevant_counter_models(self, callpath_models) -> list[Model]:

        actual_counter_list = self.determine_relevant_counters(callpath_models)

        use_exponent = self.counter_selection_strategy in ('top_exponent', 'greedy_exponent')
        counter_weights = []
        for model in callpath_models:
            metric = model.metric
            if str(metric) in actual_counter_list:
                if use_exponent:
                    weight = self._max_exponent(model)
                else:
                    # TODO maybe add median calculation
                    weight = sum(measurement.mean for measurement in model.measurements)
                counter_weights.append((weight, model, str(metric)))

        counter_weights.sort(
            key=lambda x: x[0],
            reverse=True
        )
        if len(counter_weights) < self.number_used_counters:
            warnings.warn('You have selected more counters than available for modeling. '
                          f'The modeling will use the maximal number of available counters ({len(counter_weights)}).')
        relevant_models = [tpl[1] for tpl in counter_weights[:self.number_used_counters]]

        logging.info(f"Selected counters: " + ", ".join(
            tpl[2] for tpl in counter_weights[:self.number_used_counters]) + " from: " + ", ".join(actual_counter_list))

        return relevant_models

    @abstractmethod
    def make_function(self, *terms: Term) -> Function:
        pass

    @abstractmethod
    def make_hypothesis(self, function: Function, use_measure: Measure) -> Hypothesis:
        pass

    def model(self, measurements: Sequence[Sequence[Measurement]], progress_bar=DUMMY_PROGRESS) -> Sequence[Model]:
        if hasattr(self.source_modeler, 'negative_coefficients'):
            self.source_modeler.negative_coefficients = self.negative_coefficients
        models = self.source_modeler.model(measurements)
        callpath_models = {}
        for model, measurement_sequence in zip(models, measurements):
            callpath = measurement_sequence[0].callpath
            metric = measurement_sequence[0].metric

            if self.track_coordinates:
                coordinates = tuple(m.coordinate.as_tuple() for m in measurement_sequence)
                key = callpath, coordinates
            else:
                key = callpath

            if key not in callpath_models:
                callpath_models[key] = []

            model.callpath = callpath
            model.metric = metric
            model.measurements = measurement_sequence
            callpath_models[key].append(model)

        if self.counter_selection_strategy == "weighted_instructions":
            return self._model_weighted(callpath_models, measurements, progress_bar)

        models = []
        for measurement_sequence in progress_bar(measurements):
            callpath = measurement_sequence[0].callpath
            metric = measurement_sequence[0].metric
            coordinates = tuple(m.coordinate.as_tuple() for m in measurement_sequence)
            if self.track_coordinates:
                key = callpath, coordinates
            else:
                key = callpath
            if str(metric) in self.metrics_to_improve:
                desired_counters = self.determine_relevant_counter_models(callpath_models[key])
                if not desired_counters:
                    for model in callpath_models[key]:
                        if model.metric is metric:
                            models.append(model)
                    warnings.warn(
                        'Missing counter data completely for at least one callpath, falling back to default modeler.')
                    logging.info('Missing counter data completely for:' + str(key))
                    continue
                if len(desired_counters) + 1 > len(coordinates):
                    required_points = len(desired_counters) + 1 - len(coordinates)
                    warnings.warn('The number of measurements is too low for the number of counters you selected. '
                                  f'You should add {required_points} additional measurement '
                                  f'point{"s" if required_points > 1 else ""} or select only {len(coordinates) - 1} '
                                  f'counter{"s" if (len(coordinates) - 1) > 1 else ""} for modeling.')

                original_model = next(m for m in callpath_models[key] if m.metric is metric)

                if self.counter_selection_strategy == 'sum_counters':
                    runtime_model = self._model_sum_counters(desired_counters, callpath, measurement_sequence,
                                                             original_model)
                elif self.counter_selection_strategy == 'greedy_exponent':
                    runtime_model = self._model_greedy_exponent(desired_counters, callpath, measurement_sequence,
                                                                original_model)
                else:
                    runtime_function = self._make_runtime_function(desired_counters)
                    counter_names = [str(model.metric) for model in desired_counters]
                    runtime_model = self._build_model(runtime_function, callpath, measurement_sequence, counter_names,
                                                      original_model)
                models.append(runtime_model)
            else:
                for model in callpath_models[key]:
                    if model.metric is metric:
                        models.append(model)

        return models

    def _make_runtime_function(self, desired_counters):
        runtime_function = self.make_function()
        for model in desired_counters:
            for term in model.hypothesis.function.compound_terms:
                reset_term = copy.copy(term)
                reset_term.reset_coefficients()
                if runtime_function.compound_terms and reset_term in runtime_function.compound_terms:
                    continue
                    # if a term occurs again in another counter, that is less important, then we have probably seen all
                    # important terms
                    # return runtime_function
                runtime_function.add_compound_term(reset_term)
        return runtime_function

    def _exponent_deviation(self, runtime_model: Model, original_model: Model) -> float:
        """Absolute deviation between the runtime model's leading exponent and the original
        runtime-only model's leading exponent."""
        return abs(self._max_exponent(runtime_model) - self._max_exponent(original_model))

    def _build_model(self, runtime_function, callpath, measurement_sequence, counter_names: list[str],
                     original_model: Model = None) -> Model:
        hypothesis = self.make_hypothesis(runtime_function, self.use_measure)
        hypothesis.function.reset_coefficients()
        hypothesis.compute_coefficients(measurement_sequence, negative_coefficients=self.negative_coefficients)
        previous = numpy.seterr(divide='ignore', invalid='ignore')

        # for all compound terms check if they are smaller than minimum allowed contribution
        # ignore the terms that contribute less than epsilon to the function
        small_terms = [i for i, term in enumerate(hypothesis.function.compound_terms)
                       if term.coefficient == 0 or hypothesis.calc_term_contribution(term, measurement_sequence)
                       < self.epsilon_minimum_term_contribution]
        for i in reversed(small_terms):
            del hypothesis.function.compound_terms[i]
        numpy.seterr(**previous)
        hypothesis.compute_cost(measurement_sequence)

        runtime_model = Model(hypothesis, callpath, measurement_sequence[0].metric)
        runtime_model.measurements = measurement_sequence
        exponent_deviation = self._exponent_deviation(runtime_model, original_model) if original_model else None
        runtime_model.annotations.append(
            CounterSelectionAnnotation(self.counter_selection_strategy, counter_names, exponent_deviation))
        return runtime_model

    def _model_sum_counters(self, desired_counters, callpath, measurement_sequence, original_model: Model) -> Model:
        """Sums the measurements of all selected counters and fits a single model to that sum."""
        summed = None
        for model in desired_counters:
            counter_measurements = sorted(model.measurements, key=lambda m: m.coordinate)
            if summed is None:
                summed = [m.copy() for m in counter_measurements]
            else:
                for total, m in zip(summed, counter_measurements):
                    total.merge(m)

        sum_model = self.source_modeler.model([summed])[0]
        runtime_function = self.make_function()
        for term in sum_model.hypothesis.function.compound_terms:
            reset_term = copy.copy(term)
            reset_term.reset_coefficients()
            runtime_function.add_compound_term(reset_term)

        counter_names = [str(model.metric) for model in desired_counters]
        return self._build_model(runtime_function, callpath, measurement_sequence, counter_names, original_model)

    def _model_greedy_exponent(self, desired_counters, callpath, measurement_sequence,
                               original_model: Model) -> Model:
        """Adds counters steepest-growth-first, keeping only the ones that improve the time model's fit."""
        ordered = sorted(desired_counters, key=self._max_exponent, reverse=True)
        accepted = []
        best_model = self._build_model(self.make_function(), callpath, measurement_sequence, [], original_model)
        for model in ordered:
            candidate = accepted + [model]
            candidate_function = self._make_runtime_function(candidate)
            candidate_names = [str(m.metric) for m in candidate]
            candidate_model = self._build_model(candidate_function, callpath, measurement_sequence, candidate_names,
                                                original_model)
            if candidate_model.hypothesis.SMAPE < best_model.hypothesis.SMAPE:
                accepted = candidate
                best_model = candidate_model
        return best_model

    def _model_weighted(self, callpath_models, measurements, progress_bar):
        models = []
        for measurement_sequence in progress_bar(measurements):
            callpath = measurement_sequence[0].callpath
            metric = measurement_sequence[0].metric
            coordinates = tuple(m.coordinate.as_tuple() for m in measurement_sequence)
            key = callpath, coordinates
            if str(metric) in self.metrics_to_improve:
                callpath_data: list = callpath_models[key]
                original_model = next(m for m in callpath_data if m.metric is metric)
                desired_counters = self.determine_relevant_counters(callpath_data)
                counter_weights = {
                    models.metric: numpy.array(
                        [measurement.mean for measurement in sorted(models.measurements, key=lambda m: m.coordinate)])
                    for models in callpath_data
                    if models.metric == self.total_count_metric or str(models.metric) in desired_counters
                }

                if self.total_count_metric not in counter_weights:
                    raise RecoverableError(f"Missing {self.total_count_metric}, which is necessary for weighting.")

                rest_weights = (counter_weights[self.total_count_metric] * 2 -
                                numpy.sum(list(counter_weights.values()), axis=0)) / counter_weights[
                                   self.total_count_metric]
                for metric in counter_weights:
                    counter_weights[metric] /= counter_weights[self.total_count_metric]

                counter_weights[self.total_count_metric] = rest_weights

                runtime_function = self.make_function()
                measurements_time = list(sorted(measurement_sequence, key=lambda m: m.coordinate))

                for model in callpath_models[key]:
                    if model.metric in counter_weights:
                        if isinstance(model.hypothesis, ConstantHypothesis):
                            weighted_measurements = []
                            for m, w in zip(measurements_time, counter_weights[model.metric]):
                                weighted_measurements.append(m.value(self.use_measure) * w)
                            runtime_function.constant_coefficient += numpy.mean(weighted_measurements)
                        else:
                            hypothesis: Hypothesis = copy.deepcopy(model.hypothesis)
                            hypothesis.function.reset_coefficients()
                            weighted_measurements = []
                            for m, w in zip(measurements_time, counter_weights[model.metric]):
                                weighted_measurements.append(m * w)
                            hypothesis.compute_coefficients(weighted_measurements)
                            for compound_term in hypothesis.function.compound_terms:
                                runtime_function += compound_term
                            runtime_function.constant_coefficient += hypothesis.function.constant_coefficient

                hypothesis = self.make_hypothesis(runtime_function, self.use_measure)
                previous = numpy.seterr(divide='ignore', invalid='ignore')

                # for all compound terms check if they are smaller than minimum allowed contribution
                # ignore the terms that contribute less than epsilon to the function
                small_terms = [i for i, term in enumerate(hypothesis.function.compound_terms)
                               if term.coefficient == 0 or hypothesis.calc_term_contribution(term,
                                                                                             measurement_sequence)
                               < self.epsilon_minimum_term_contribution]
                for i in reversed(small_terms):
                    del hypothesis.function.compound_terms[i]
                numpy.seterr(**previous)
                hypothesis.compute_cost(measurement_sequence)

                runtime_model = Model(hypothesis, callpath, measurement_sequence[0].metric)
                runtime_model.measurements = measurement_sequence
                counter_names = [str(m) for m in counter_weights.keys() if m != self.total_count_metric]
                exponent_deviation = self._exponent_deviation(runtime_model, original_model)
                runtime_model.annotations.append(
                    CounterSelectionAnnotation('weighted_instructions', counter_names, exponent_deviation))
                models.append(runtime_model)
            else:
                for model in callpath_models[key]:
                    if model.metric is metric:
                        models.append(model)
        return models


@modeler_options
class HardwareCounterModeler(AbstractHardwareCounterModeler):
    NAME = 'Hardware Counter'
    DESCRIPTION = 'Creates a runtime model based on data of hardware performance counters'

    def __init__(self):
        """
        Initialize HardwareCounterModeler object.
        """
        super().__init__(use_measure=Measure.MEAN, source_modeler=SingleParameterModeler())

    make_function = SingleParameterFunction
    make_hypothesis = SingleParameterHypothesis
