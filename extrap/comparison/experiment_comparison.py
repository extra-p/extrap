# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import copy
from typing import List, Sequence, Mapping

import sympy
from marshmallow import fields

from extrap.comparison.entities.comparison_model import ComparisonModel
from extrap.comparison.entities.comparison_model_generator import ComparisonModelGenerator
from extrap.comparison.entities.projection_info import ProjectionInfo, ProjectionInfoSchema
from extrap.comparison.matchers import AbstractMatcher
from extrap.comparison.matches import IdentityMatches, MutableAbstractMatches
from extrap.comparison.metric_conversion import AbstractMetricConverter
from extrap.entities.callpath import CallpathSchema
from extrap.entities.calltree import Node
from extrap.entities.experiment import Experiment, ExperimentSchema
from extrap.entities.function_computation import ComputationFunction
from extrap.entities.metric import Metric, MetricSchema
from extrap.entities.model import Model
from extrap.entities.parameter import Parameter
from extrap.entities.scaling_type import ScalingType
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.postprocessing.arithmetic_intensity_calculation import ArithmeticIntensityCalculation
from extrap.util.exceptions import RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS
from extrap.util.serialization_schema import ListToMappingField
from extrap.util.unique_list import UniqueList

COMPARISON_NODE_NAME = '[Comparison]'
TAG_COMPARISON_NODE = 'comparison__node'
TAG_COMPARISON_NODE__comparison = 'comparison'


class ComparisonError(RecoverableError):
    NAME = "Experiment Comparison Error"


class ComparisonExperiment(Experiment):
    def __init__(self, exp1: Experiment, exp2: Experiment, matcher: AbstractMatcher):
        super(ComparisonExperiment, self).__init__()
        self.experiment_names: List[str] = ['exp1', 'exp2']
        if exp1 is None and exp2 is None:
            self.compared_experiments: List[Experiment] = []
            return
        self.compared_experiments: List[Experiment] = [exp1, exp2]
        self.matcher = matcher
        self.exp1 = exp1
        self.exp2 = exp2
        self.modelers_match: Mapping[str, Sequence[ModelGenerator]] = {}
        self.parameter_mapping: Mapping[str, Sequence[str]] = {}
        self.projection_info: ProjectionInfo = ProjectionInfo(len(self.compared_experiments))

    def do_comparison(self, progress_bar=DUMMY_PROGRESS):
        progress_bar.total += 3
        num_metrics = max(len(e.metrics) for e in self.compared_experiments)
        num_callpaths = max(len(e.callpaths) for e in self.compared_experiments)
        num_models = max(len(e.modelers) for e in self.compared_experiments)

        progress_metrics_target = progress_bar.create_target(num_metrics)
        progress_call_tree_target = progress_bar.create_target(num_callpaths)
        progress_modelers_target = progress_bar.create_target(num_models * num_metrics * num_callpaths)

        self.do_parameter_mapping()

        self.do_initial_checks()
        progress_bar.update(3)

        if not self.try_metric_merge():
            self.do_metrics_merge(progress_bar)
        progress_bar.reach_target(progress_metrics_target)

        self.do_call_tree_merge(progress_bar)
        progress_bar.reach_target(progress_call_tree_target)

        self.do_model_set_merge(progress_bar)
        progress_bar.reach_target(progress_modelers_target)

    def do_initial_checks(self):
        if self.exp1.parameters != self.exp2.parameters:
            raise ComparisonError("Parameters do not match.")
        # if self.exp1.coordinates != self.exp2.coordinates:
        #     raise ComparisonError("Coordinates do not match.")
        if self.exp1.scaling is not None and self.exp2.scaling is not None and ScalingType(
                self.exp1.scaling) != ScalingType(self.exp2.scaling):
            raise ComparisonError("Scaling does not match.")
        self.parameters = self.exp1.parameters
        self.coordinates = UniqueList(self.exp1.coordinates)
        self.coordinates.extend(self.exp2.coordinates)
        self.scaling = self.exp1.scaling

    def try_metric_merge(self):
        if self.exp1.metrics != self.exp2.metrics:
            return False
        self.metrics = self.exp1.metrics
        self.metrics_match = IdentityMatches(2, self.metrics)
        return True

    def do_metrics_merge(self, progress_bar=DUMMY_PROGRESS):
        self.metrics, self.metrics_match, converters = self.matcher.match_metrics(self.exp1.metrics, self.exp2.metrics,
                                                                                  progress_bar=progress_bar)
        if converters:
            self._apply_metric_converters(converters)

    def do_call_tree_merge(self, progress_bar=DUMMY_PROGRESS):
        self.call_tree, self.call_tree_match = self.matcher.match_call_tree(self.exp1.call_tree, self.exp2.call_tree)
        if hasattr(self.matcher, 'make_measurements_and_update_call_tree'):
            self.measurements = self.matcher.make_measurements_and_update_call_tree(self, self.call_tree_match,
                                                                                    self.exp1.measurements,
                                                                                    self.exp2.measurements)
        else:
            self.measurements = self._make_measurements_and_update_call_tree(self.call_tree_match,
                                                                             self.exp1.measurements,
                                                                             self.exp2.measurements)
        self.callpaths, self.callpaths_match = self._callpaths_from_tree(self.call_tree_match)

    def _callpaths_from_tree(self, match):
        cp_match = {n.path: [n1.path if n1 else None for n1 in n12] for n, n12 in match.items()}
        return list(cp_match.keys()), cp_match

    def _make_measurements_and_update_call_tree(self, call_tree_match: MutableAbstractMatches[Node],
                                                *source_measurements):
        measurements = {}
        new_matches = {}
        comparison_nodes = {}
        for metric, source_metrics in self.metrics_match.items():
            for node, source_nodes in call_tree_match.items():
                comparison_node = comparison_nodes.get(node)
                if not comparison_node:
                    comparison_node = Node(COMPARISON_NODE_NAME, node.path.concat(COMPARISON_NODE_NAME))
                    comparison_node.path.tags[TAG_COMPARISON_NODE] = TAG_COMPARISON_NODE__comparison
                    new_matches[comparison_node] = [None] * len(source_measurements)
                for i, (s_node, s_metric, s_measurement, s_name) in enumerate(
                        zip(source_nodes, source_metrics, source_measurements, self.experiment_names)):
                    source_key = (s_node.path, s_metric)
                    if source_key in s_measurement:
                        name = f"[{s_name}] {node.name}"
                        part_node = comparison_node.find_child(name)
                        if not part_node:
                            cp = comparison_node.path.concat(name)
                            ct_node = Node(name, cp)
                            comparison_node.add_child_node(ct_node)
                            new_match = [None] * len(source_measurements)
                            new_match[i] = s_node
                            new_matches[ct_node] = new_match
                        else:
                            cp = part_node.path
                        measurements[cp, metric] = s_measurement[source_key]
                if comparison_node.childs and node not in comparison_nodes:
                    node.add_child_node(comparison_node)
                    comparison_nodes[node] = comparison_node

        call_tree_match.update(new_matches)
        return measurements

    def do_model_set_merge(self, progress_bar=DUMMY_PROGRESS):
        if not self.modelers_match:
            self.modelers_match = self.matcher.match_modelers(self.exp1.modelers, self.exp2.modelers)
        if hasattr(self.matcher, 'make_model_generator'):
            self.modelers = [self.matcher.make_model_generator(self, k, match, progress_bar) for k, match in
                             self.modelers_match.items()]
        else:
            self.modelers = [self._make_model_generator(k, match) for k, match in self.modelers_match.items()]

    def _make_model_generator(self, name: str, modelers: Sequence[ModelGenerator]) -> ComparisonModelGenerator:
        mg = ComparisonModelGenerator(self, name, modelers[0].modeler.use_median)
        mg.models = {}
        for metric, source_metrics in self.metrics_match.items():
            for node, source_nodes in self.call_tree_match.items():
                models = []
                for i, (s_node, s_metric, s_modeler, s_name) in enumerate(
                        zip(source_nodes, source_metrics, modelers, self.experiment_names)):
                    if s_node is not None:
                        source_key = (s_node.path, s_metric)
                        if source_key in s_modeler.models:
                            models.append(s_modeler.models[source_key].with_callpath(node.path))
                if len(models) == 1:
                    mg.models[node.path, metric] = models[0]
                elif len(models) > 1:
                    mg.models[node.path, metric] = ComparisonModel(node.path, metric, models)
        return mg

    def _apply_metric_converters(self, converters: Sequence[AbstractMetricConverter]):
        for converter in converters:
            for i, exp in enumerate([self.exp1, self.exp2]):
                for callpath in exp.callpaths:
                    try:
                        measurements = converter.convert_measurements(i,
                                                                      [exp.measurements[callpath, metric] for metric in
                                                                       converter.get_required_metrics(i)])
                    except KeyError:
                        continue
                    except ZeroDivisionError:
                        continue
                    exp.measurements[callpath, converter.new_metric] = measurements
                    for model_set in exp.modelers:
                        try:
                            model = converter.convert_models(i, [model_set.models[callpath, metric] for metric in
                                                                 converter.get_required_metrics(i)], measurements)
                            if model:
                                model_set.models[callpath, converter.new_metric] = model
                        except ZeroDivisionError:
                            continue

    def do_parameter_mapping(self):
        for name, old_names in self.parameter_mapping.items():
            for old_name, experiment in zip(old_names, self.compared_experiments):
                if name == old_name:
                    continue
                idx = experiment.parameters.index(Parameter(old_name))
                experiment.parameters[idx] = Parameter(name)

    def project_expected_performance(self, projection_info: ProjectionInfo, pbar):
        ar_int_preprocess = None
        target_id = projection_info.target_experiment_id
        if projection_info.num_mem_transfers_metric:
            ar_int_preprocess = ArithmeticIntensityCalculation(self)
            ar_int_preprocess.bytes_per_mem_transfer = projection_info.bytes_per_mem
            ar_int_preprocess.num_mem_transfers_metric = projection_info.num_mem_transfers_metric
            ar_int_preprocess.flops_dp_metric = projection_info.fp_dp_metric
            for eid, experiment in enumerate(self.compared_experiments):
                if eid == target_id:
                    continue
                if projection_info.fp_dp_metric not in experiment.metrics or \
                        projection_info.num_mem_transfers_metric not in experiment.metrics:
                    continue
                for model_set in experiment.modelers:
                    ar_int_models = ar_int_preprocess.generate_arithmetic_intensity_models(model_set.models, pbar)
                    model_set.models.update(ar_int_models)
        new_metrics = {metric: Metric("Expected " + metric.name, projection__original_metric=metric.name) for metric in
                       projection_info.metrics_to_project}
        self.metrics.extend(new_metrics.values())
        for modeler in self.modelers:
            pbar.total += len(modeler.models)
        for modeler in self.modelers:
            new_models = {}
            for (callpath, metric), c_model in modeler.models.items():
                pbar.update()
                if metric not in projection_info.metrics_to_project:
                    continue
                if not isinstance(c_model, ComparisonModel):
                    continue
                models = []
                for i, model in enumerate(c_model.models):
                    if i == target_id:
                        models.append(model)
                    else:
                        ar_int = sympy.oo
                        if ar_int_preprocess:
                            compared_modeler = self.modelers_match[modeler.name][i].models
                            key = (callpath, ar_int_preprocess.arithmetic_intensity_metric)
                            if key in compared_modeler:
                                ar_int_func: ComputationFunction = compared_modeler[key].hypothesis.function
                                ar_int = ar_int_func.sympy_function

                        roofline_performance = sympy.Min(projection_info.peak_mem_bandwidth_in_gbytes_per_s[i] * ar_int,
                                                         projection_info.peak_performance_in_gflops_per_s[i])
                        target_roofline_performance = sympy.Min(projection_info.peak_mem_bandwidth_in_gbytes_per_s[
                                                                    target_id] * ar_int,
                                                                projection_info.peak_performance_in_gflops_per_s[
                                                                    target_id])
                        scaling_factor = roofline_performance / target_roofline_performance
                        hypothesis = copy.copy(model.hypothesis)
                        hypothesis.function = (ComputationFunction(model.hypothesis.function) * scaling_factor)
                        models.append(Model(hypothesis, model.callpath, new_metrics[model.metric]))

                model = ComparisonModel(callpath, new_metrics[metric], models)
                new_models[callpath, model.metric] = model

            modeler.models.update(new_models)


class ComparisonExperimentSchema(ExperimentSchema):
    callpaths = ListToMappingField(CallpathSchema, 'name', list_type=UniqueList)
    metrics = ListToMappingField(MetricSchema, 'name', list_type=UniqueList)
    experiment_names = fields.List(fields.String())
    projection_info = fields.Nested(ProjectionInfoSchema)

    def create_object(self):
        return ComparisonExperiment(None, None, None)

    def postprocess_object(self, obj: Experiment):
        experiment = super().postprocess_object(obj)
        experiment.call_tree.ensure_callpaths_exist()
        return experiment
