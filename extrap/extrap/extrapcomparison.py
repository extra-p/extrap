# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import argparse
import logging
import warnings
from itertools import groupby, islice
from operator import itemgetter
from pathlib import Path

import extrap
from extrap.comparison.entities.comparison_model import ComparisonModel
from extrap.comparison.entities.projection_info import ProjectionInfo
from extrap.comparison.experiment_comparison import ComparisonExperiment
from extrap.comparison.matchers import all_matchers
from extrap.entities.metric import Metric
from extrap.fileio.experiment_io import read_experiment
from extrap.util.progress_bar import ProgressBar


class dummy_file_stream:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def write(self, value):
        pass


def main(raw_args=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=extrap.__description__, add_help=False)
    positional_arguments = parser.add_argument_group("Positional arguments")
    basic_arguments = parser.add_argument_group("Optional arguments")
    basic_arguments.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                                 help='Show this help message and exit')

    basic_arguments.add_argument("--version", action="version",
                                 version=extrap.__title__ + " Comparison " + extrap.__version__,
                                 help="Show program's version number and exit")
    basic_arguments.add_argument("--log", action="store", dest="log_level", type=str.lower, default='warning',
                                 choices=['debug', 'info', 'warning', 'error', 'critical'],
                                 help="Set program's log level (default: %(default)s)")

    exp1_options = parser.add_argument_group("Experiment 1 Options")

    exp1_options.add_argument("path1", metavar="FILEPATH_EXPERIMENT_1", type=str, action="store",
                              help="Specify a file path for Extra-P to work with")

    exp2_options = parser.add_argument_group("Experiment 2 Options")

    exp2_options.add_argument("path2", metavar="FILEPATH_EXPERIMENT_2", type=str, action="store",
                              help="Specify a file path for Extra-P to work with")

    comparison_options = parser.add_argument_group("Comparison options")
    comparison_options.add_argument("--name", nargs=2, default=[])
    comparison_options.add_argument("--mapping-provider", choices=[m for m in all_matchers], required=True)
    comparison_options.add_argument("--parameter-values", nargs='+', help="Tuples of parameter values to compare.")
    comparison_options.add_argument("--model-set-names", nargs=2, default=[])
    comparison_options.add_argument("--parameter-mapping", nargs='+')
    comparison_options.add_argument('--num-results', type=int, default=5)
    comparison_options.add_argument("--metrics", nargs='+', default=['time'], metavar="METRIC")

    projection_options = parser.add_argument_group("Projection options")
    projection_options.add_argument("--project", action='store_true')
    projection_options.add_argument("--target", choices=[1, 2], type=int, default=2)
    projection_options.add_argument("--mem-bandwidth", nargs=2, type=float)
    projection_options.add_argument("--peak-performance", nargs=2, type=float)
    projection_options.add_argument("--projection-metrics", nargs='+', default=['time'], metavar="METRIC")

    projection_options.add_argument("--use-arithmetic-intensity", nargs=3, metavar="",
                                    help="FP Metric, Mem Transfer Metric, Bytes per Transfer")

    parser.add_argument('--disable-progress', action="store_true")
    parser.add_argument('--out', action="store", metavar="OUTPUT_PATH",
                        help="Specify the output path for comparison results")

    parser.add_argument("--non-interactive", action='store_true')

    arguments = parser.parse_args(raw_args)

    # set log level
    loglevel = logging.getLevelName(arguments.log_level.upper())

    # set log format location etc.
    if loglevel == logging.DEBUG:
        logging.basicConfig(
            format="%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s - %(funcName)10s(): %(message)s",
            level=loglevel, datefmt="%m/%d/%Y %I:%M:%S %p")
    else:
        logging.basicConfig(
            format="%(levelname)s: %(message)s", level=loglevel)

    with ProgressBar(desc='Loading experiment 1', disable=arguments.disable_progress) as pbar:
        exp1 = read_experiment(arguments.path1, pbar)
    with ProgressBar(desc='Loading experiment 2', disable=arguments.disable_progress) as pbar:
        exp2 = read_experiment(arguments.path2, pbar)

    mapping_provider = all_matchers[arguments.mapping_provider]
    comp = ComparisonExperiment(exp1, exp2, mapping_provider())
    if arguments.name:
        comp.experiment_names = arguments.name
    else:
        comp.experiment_names = [Path(arguments.path1).stem, Path(arguments.path2).stem]

    if arguments.parameter_mapping:
        mapping = {}
        for pm in arguments.parameter_mapping:
            try:
                p, old_ps = pm.split('=')
                old_ps = old_ps.split(',')
                if len(old_ps) != 2:
                    raise ValueError("Both parameters must be part of the parameter-mapping expression.")
                mapping[p] = old_ps
            except Exception as e:
                print(f"Could not interpret parameter mapping: {pm}: {e}")
                exit(1)
        comp.parameter_mapping = mapping

    if arguments.model_set_names:
        first_modeler = next(m for m in comp.exp1.modelers if m.name == arguments.model_set_names[0])
        second_modeler = next(m for m in comp.exp2.modelers if m.name == arguments.model_set_names[1])
        if not first_modeler or not second_modeler:
            warnings.warn(
                "Could not find model set with corresponding name. Using the first match by matching provider.")
        comp.modelers_match = {'Comparison': [first_modeler, second_modeler]}
    with ProgressBar(desc='Comparing', disable=arguments.disable_progress) as pbar:
        comp.do_comparison(pbar)

    if arguments.project:
        pinfo = ProjectionInfo(2)
        pinfo.target_experiment_id = arguments.target - 1
        pinfo.metrics_to_project = [Metric(m) for m in arguments.projection_metrics]
        pinfo.peak_performance_in_gflops_per_s = arguments.peak_performance
        pinfo.peak_mem_bandwidth_in_gbytes_per_s = arguments.mem_bandwidth
        if 'use-arithmetic-intensity' in pinfo:
            fp_dp_metric, num_mem_transfers_metric, bytes_per_mem = arguments['use-arithmetic-intensity']
            pinfo.fp_dp_metric = Metric(fp_dp_metric)
            pinfo.num_mem_transfers_metric = Metric(num_mem_transfers_metric)
            pinfo.bytes_per_mem = int(bytes_per_mem)

        with ProgressBar(desc='Projection', disable=arguments.disable_progress) as pbar:
            comp.project_expected_performance(pinfo, pbar)

    metrics = [Metric(m) for m in arguments.metrics]

    if arguments.out:
        file_handle = open(arguments.out, 'w')
    else:
        file_handle = dummy_file_stream()
    with file_handle as file:

        if arguments.parameter_values:
            for p_val_string in arguments.parameter_values:
                p_values = [float(p) for p in p_val_string.strip().split(',')]
                res = format_results(comp, metrics, p_values, arguments.num_results)
                file.write(res)
                print(res)

        if arguments.non_interactive:
            return
        while True:
            message = f"Please enter values for all parameters ({', '.join(p.name for p in comp.parameters)}) at which the comparison should be performed (exit with q): "
            p_values_str = input(message)

            if p_values_str.lower() in ['quit', 'exit', 'q']:
                break

            p_values = [float(p.strip()) for p in p_values_str.split(',')]
            print('-' * (len(message) + len(p_values_str)))
            print()
            res = format_results(comp, metrics, p_values, arguments.num_results)
            file.write(res)
            print(res)


def format_results(comp, metrics, p_values, num_results):
    output = ""
    for comp_metric in metrics:
        output += (f"Metric: {comp_metric.name} for ({', '.join(p.name for p in comp.parameters)})="
                   f"({', '.join(str(p) for p in p_values)})\n")

        result_list = []

        for key, model in comp.modelers[0].models.items():
            if not isinstance(model, ComparisonModel):
                continue
            if not key[1] == comp_metric:
                continue
            result = model.hypothesis.function.evaluate(p_values)
            diff = result[1] - result[0]
            result_list.append((diff, result, key, model))

        result_list.sort(key=itemgetter(0))

        result_groups = groupby(result_list, lambda x: -1 if x[0] < 0 else 1 if x[0] > 0 else 0)

        for key, res_group in result_groups:
            if key == -1:
                output += f"{comp.experiment_names[0]} > {comp.experiment_names[1]}\n"
            elif key == 1:
                output += f"{comp.experiment_names[0]} < {comp.experiment_names[1]}\n"
                res_group = reversed(list(res_group))
            else:
                continue
            for diff, res, (callpath, metric), model in islice(res_group, num_results):
                output += f"{abs(diff)} \t {callpath} \t {model.hypothesis.function.to_string(*comp.parameters)}\n"
    output += "\n"
    return output


if __name__ == "__main__":
    main()
