# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
import multiprocessing
import os.path
import re
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import List

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.fileio.file_reader.abstract_directory_reader import AbstractDirectoryReader
from extrap.fileio.file_reader.nv_reader.agg_ncu_report import AggNcuReport
from extrap.fileio.file_reader.nv_reader.ncu_report import NcuReport
from extrap.fileio.file_reader.nv_reader.nsys_db import NsysReport
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import DUMMY_PROGRESS

ns_per_s = 10 ** 9


class NsightFileReader(AbstractDirectoryReader):
    NAME = "nsight"
    GUI_ACTION = "Open set of &Nsight files"
    DESCRIPTION = "Load a set of Nsight Systems files and generate a new experiment"
    CMD_ARGUMENT = "--nsight"
    LOADS_FROM_DIRECTORY = True

    legacy_format = False
    ignore_device_attributes = True

    def read_experiment(self, dir_name, pbar=DUMMY_PROGRESS, selected_metrics=None, only_time=False):
        # read the paths of the nsight files in the given directory with dir_name
        path = Path(dir_name)
        if not path.is_dir():
            raise FileFormatError(f'NV file path must point to a directory: {dir_name}')
        if self.legacy_format:
            nv_files = list(path.glob('[!.]*.sqlite'))
        else:
            nv_files = list(path.glob('*/[!.]*.sqlite'))
        if not nv_files:
            if self.legacy_format:
                ncu_files = list(path.glob('[!.]*.ncu-rep'))
            else:
                ncu_files = list(path.glob('*/[!.]*.ncu-rep'))
            if ncu_files:
                return self.read_ncu_files(dir_name, ncu_files, pbar=DUMMY_PROGRESS, selected_metrics=None,
                                           only_time=False)
            else:
                raise FileFormatError(f'No sqlite or ncu-rep files were found in: {dir_name}')
        pbar.total += len(nv_files) + 6
        # iterate over all folders and read the nv profiles in them
        experiment = Experiment()

        pbar.step("Reading NV files")
        if self.legacy_format:
            parameter_names, parameter_values = self._determine_parameter_values_legacy(nv_files, pbar)
        else:
            parameter_names, parameter_values = self._determine_parameters_from_paths(nv_files, pbar)
        for p in parameter_names:
            experiment.add_parameter(Parameter(p))

        pbar.step("Reading sqlite files")

        show_warning_skipped_metrics = set()
        aggregated_values = defaultdict(list)

        try:
            pool = None
            num_points = 0
            reordered_files = sorted(zip(nv_files, parameter_values), key=itemgetter(1))
            for parameter_value, point_group in groupby(reordered_files, key=itemgetter(1)):
                num_points += 1
                point_group = list(point_group)
                # create coordinate
                coordinate = Coordinate(parameter_value)
                experiment.add_coordinate(coordinate)

                aggregated_values.clear()
                metric = Metric('time')
                metric_bytes = Metric('bytes_transferred')
                correponding_agg_ncu_path = None
                for path, _ in point_group:
                    pbar.update()
                    with NsysReport(path) as parsed:
                        # iterate over all callpaths and get time
                        for callpath, duration in parsed.get_gpu_idle():
                            pbar.update(0)
                            if duration:
                                cp_obj = Callpath(callpath + '->GPU IDLE', gpu__idle=True, agg__category='GPU IDLE',
                                                  validation__ignore__num_measurements=True)
                                aggregated_values[(cp_obj, metric)].append(duration / ns_per_s)

                        for id, callpath, overlap_name, duration, durationGPU, syncType, other_duration in parsed.get_synchronization():
                            pbar.update(0)
                            if overlap_name:
                                cp_obj = Callpath(callpath + "->" + syncType + "->OVERLAP",
                                                  validation__ignore__num_measurements=True,
                                                  gpu__overlap='agg', agg__usage_disabled=True)
                                aggregated_values[(cp_obj, metric)] = [0]

                                overlap_cp = Callpath(callpath + "->" + syncType + "->OVERLAP->" + overlap_name,
                                                      gpu__overlap=True, gpu__kernel=True,
                                                      validation__ignore__num_measurements=True)
                                aggregated_values[(overlap_cp, metric)].append(durationGPU / ns_per_s)
                            else:
                                if duration:
                                    cp_obj = Callpath(callpath + "->" + syncType,
                                                      agg__category__comparison_cpu_gpu='GPU SYNC')
                                    aggregated_values[(cp_obj, metric)].append(duration / ns_per_s)
                                cp_obj = Callpath(callpath + "->" + syncType + "->WAIT", agg__usage_disabled=True)
                                aggregated_values[(cp_obj, metric)].append(other_duration / ns_per_s)

                        for id, callpath, kernelName, duration, durationGPU, other_duration in parsed.get_kernel_runtimes():
                            pbar.update(0)
                            if kernelName:
                                if duration:
                                    aggregated_values[(Callpath(callpath + "->" + kernelName), metric)].append(
                                        duration / ns_per_s)
                                cp_obj = Callpath(callpath + "->" + kernelName + "->GPU " + kernelName,
                                                  gpu__kernel=True, agg__category='GPU',
                                                  agg__category__comparison_cpu_gpu=None)
                                aggregated_values[(cp_obj, metric)].append(durationGPU / ns_per_s)
                            elif duration:
                                aggregated_values[(Callpath(callpath), metric)].append(duration / ns_per_s)
                        for id, callpath, overlap_name, duration, durationGPU in parsed.get_mem_alloc_overlap():
                            pbar.update(0)
                            # TODO solve parent cudaFree/cudaMalloc consists of overlap and real value
                            #   maybe: cudaFree/Malloc<without overlap> ->OVERLAP<actual overlap> ->[Kernels...<overlap of each kernel>]
                            if overlap_name:
                                cp_obj = Callpath(callpath + "->OVERLAP->" + overlap_name,
                                                  gpu__overlap=True, gpu__kernel=True,
                                                  validation__ignore__num_measurements=True)
                                aggregated_values[(cp_obj, metric)].append(durationGPU / ns_per_s)
                            elif duration:
                                cp_obj = Callpath(callpath + "->GPU MEM", )
                                aggregated_values[(Callpath(callpath), metric)].append(duration / ns_per_s)
                        for id, callpath, overlap_name, duration, bytes, kind, blocking, duration_copy, duration_overlap in parsed.get_mem_copies():
                            pbar.update(0)
                            if duration:
                                if duration_copy and blocking:
                                    duration -= duration_copy
                                aggregated_values[(Callpath(callpath), metric)].append(duration / ns_per_s)
                            if duration_copy:
                                if duration_overlap:
                                    duration_copy -= duration_overlap
                                sep = "->BLOCKING " if blocking else "->"
                                cp_obj = Callpath(callpath + sep + kind, gpu__mem_copy=True)
                                if blocking:
                                    cp_obj.tags['gpu__blocking__mem_copy'] = True
                                else:
                                    cp_obj.tags['agg__category'] = 'GPU MEM',
                                    cp_obj.tags['gpu__blocking__mem_copy'] = False
                                    cp_obj.tags['agg__category__comparison_cpu_gpu'] = None
                                aggregated_values[(cp_obj, metric)].append(duration_copy / ns_per_s)
                                aggregated_values[(cp_obj, metric_bytes)].append(bytes)
                                if duration_overlap:
                                    cp_overlap = cp_obj.concat('OVERLAP', gpu__overlap=True, agg__disabled=True)
                                    aggregated_values[(cp_overlap, metric)].append(duration_overlap / ns_per_s)
                            elif duration_overlap:
                                sep = "->BLOCKING " if blocking else "->"
                                cp_obj = Callpath(callpath + sep + kind + '->OVERLAP->' + overlap_name,
                                                  gpu__overlap=True)
                                aggregated_values[(cp_obj, metric)].append(duration_overlap / ns_per_s)

                        for id, callpath, overlap_name, duration, bytes, blocking, duration_set in parsed.get_mem_sets():
                            pbar.update(0)
                            if duration:
                                aggregated_values[(Callpath(callpath), metric)].append(duration / ns_per_s)
                            if duration_set:
                                sep = "->BLOCKING GPU MEMSET" if blocking else "->GPU MEMSET"
                                cp_obj = Callpath(callpath + sep, gpu__mem_set=True)
                                if blocking:
                                    cp_obj.tags['gpu__blocking__mem_set'] = True
                                    cp_obj.tags['agg__usage_disabled'] = True
                                else:
                                    cp_obj.tags['agg__category'] = 'GPU MEM',
                                    cp_obj.tags['gpu__blocking__mem_set'] = False
                                    cp_obj.tags['agg__category__comparison_cpu_gpu'] = None
                                aggregated_values[(cp_obj, metric)].append(duration_set / ns_per_s)
                                aggregated_values[(cp_obj, metric_bytes)].append(bytes)

                        for id, callpath, name, duration in parsed.get_os_runtimes():
                            pbar.update(0)
                            if duration:
                                aggregated_values[(Callpath(callpath), metric)].append(duration / ns_per_s)

                        temp_correponding_agg_ncu_path = Path(path).with_suffix(".ncu-rep.agg")
                        if temp_correponding_agg_ncu_path.exists():
                            correponding_ncu_path = None
                            correponding_agg_ncu_path = temp_correponding_agg_ncu_path
                        else:
                            correponding_ncu_path = Path(path).with_suffix(".nsight-cuprof-report")
                            if not correponding_ncu_path.exists():
                                correponding_ncu_path = Path(path).with_suffix(".ncu-rep")
                        if not correponding_agg_ncu_path and correponding_ncu_path.exists() and not only_time:
                            if pool is None:
                                pool = multiprocessing.Pool()
                            with NcuReport(correponding_ncu_path) as ncu_report:
                                ignore_metrics = None
                                if self.ignore_device_attributes:
                                    ignore_metrics = ['device__attribute', 'nvlink__']
                                measurements = ncu_report.get_measurements_parallel(parsed.get_kernelid_paths(), pool,
                                                                                    ignore_metrics=ignore_metrics)
                                for (callpath, metric_id), v in measurements.items():
                                    aggregated_values[
                                        (Callpath(callpath), Metric(ncu_report.string_table[metric_id]))].append(v)

                # add measurements to experiment
                for (callpath, metric), values in aggregated_values.items():
                    pbar.update(0)
                    experiment.add_measurement(Measurement(coordinate, callpath, metric, values))

                if correponding_agg_ncu_path:
                    with  AggNcuReport(correponding_agg_ncu_path) as agg_report, NsysReport(path) as parsed:
                        ignore_metrics = None
                        if self.ignore_device_attributes:
                            ignore_metrics = ['device__attribute', 'nvlink__']
                        measurements = agg_report.get_measurements(ignore_metrics=ignore_metrics)
                        for (callpath_enc, metric_id), values in measurements.items():
                            metric = Metric(agg_report.string_table[metric_id])
                            callpath = Callpath(parsed.decode_callpath(callpath_enc))
                            experiment.add_measurement(Measurement(coordinate, callpath, metric, values))
        finally:
            if pool:
                pool.close()

        pbar.step("Unify call-trees")
        to_delete = []
        # determine common callpaths for common calltree
        # add common callpaths and metrics to experiment
        for key, value in pbar.__call__(experiment.measurements.items(), len(experiment.measurements), scale=0.1):
            value: List[Measurement]

            if len(value) < num_points and (
                    not key[0].lookup_tag('validation__ignore__num_measurements', False) or len(value) <= 1):
                to_delete.append(key)
            else:
                (callpath, metric) = key
                # if len(value) < num_points and 'gpu__overlap' in callpath.tags:
                #     # construct empty measurements for overlap
                #     measurements = {c: Measurement(c, callpath, metric, None) for c in experiment.coordinates}
                #     for v in value:
                #         del measurements[v.coordinate]
                #     value.extend(measurements.values())

                experiment.add_callpath(callpath)
                experiment.add_metric(metric)
        for key in to_delete:
            pbar.update(0)
            del experiment.measurements[key]

        # determine calltree
        call_tree = io_helper.create_call_tree(experiment.callpaths, pbar, progress_scale=0.1)
        experiment.call_tree = call_tree

        io_helper.validate_experiment(experiment, pbar)
        pbar.update()
        return experiment

    def _determine_parameter_values_legacy(self, files, pbar):
        pbar.step("Reading NV files")
        parameter_names_initial = []
        parameter_names = []
        parameter_values = []
        parameter_dict = defaultdict(set)
        progress_step_size = 5 / len(files)
        for path_id, path in enumerate(files):
            pbar.update(progress_step_size)
            folder_name = path.name
            logging.debug(f"NV file: {path}")

            # create the parameters
            par_start = folder_name.find(".") + 1
            par_end = folder_name.find(".r")
            par_end = None if par_end == -1 else par_end
            parameters = folder_name[par_start:par_end]
            # parameters = folder_name.split(".")

            param_list = re.split('([0-9.,]+)', parameters)

            parameter_names = [n for i, n in enumerate(param_list) if i % 2 == 0]
            parameter_value = [float(n.replace(',', '.').rstrip('.')) for i, n in enumerate(param_list) if i % 2 == 1]

            # check if parameter already exists
            if path_id == 0:
                parameter_names_initial = parameter_names
            elif parameter_names != parameter_names_initial:
                raise FileFormatError(
                    f"Parameters must be the same and in the same order: {parameter_names} is not {parameter_names_initial}.")

            for n, v in zip(parameter_names, parameter_value):
                parameter_dict[n].add(v)
            parameter_values.append(parameter_value)
        # determine non-constant parameters and add them to experiment
        for i in reversed(range(len(parameter_names))):
            p = parameter_names[i]
            if len(parameter_dict[p]) <= 1:
                for pv in parameter_values:
                    del pv[i]

        parameter_names = [p for p in parameter_names if len(parameter_dict[p]) > 1]

        return parameter_names, parameter_values

    def read_ncu_files(self, dir_name, ncu_files, pbar=DUMMY_PROGRESS, selected_metrics=None, only_time=False):
        pbar.total += len(ncu_files) + 6
        # iterate over all folders and read the cube profiles in them
        experiment = Experiment()

        if self.legacy_format:
            parameter_names, parameter_values = self._determine_parameter_values_legacy(ncu_files, pbar)
        else:
            parameter_names, parameter_values = self._determine_parameters_from_paths(ncu_files, pbar)
        for p in parameter_names:
            experiment.add_parameter(Parameter(p))

        aggregated_values = defaultdict(list)

        pool = None
        num_points = 0
        reordered_files = sorted(zip(ncu_files, parameter_values), key=itemgetter(1))
        for parameter_value, point_group in groupby(reordered_files, key=itemgetter(1)):
            num_points += 1
            point_group = list(point_group)
            # create coordinate
            coordinate = Coordinate(parameter_value)
            experiment.add_coordinate(coordinate)

            aggregated_values.clear()
            agg_path = str(point_group[0][0]) + '.agg'
            if os.path.exists(agg_path):
                with AggNcuReport(agg_path) as agg_report:
                    pbar.update(agg_report.count)
                    if self.ignore_device_attributes:
                        measurements = agg_report.get_measurements_unmapped(
                            ignore_metrics=['device__attribute', 'nvlink__'])
                    else:
                        measurements = agg_report.get_measurements_unmapped()
                    for (kernelId, metricId), values in measurements.items():
                        pbar.update(0)
                        experiment.add_measurement(
                            Measurement(coordinate, Callpath('main->' + agg_report.kernel_names[kernelId]),
                                        Metric(agg_report.string_table[metricId]), values))
            else:
                for path, _ in point_group:
                    pbar.update()
                    with NcuReport(path) as ncu_report:
                        if self.ignore_device_attributes:
                            measurements = ncu_report.get_measurements_unmapped(
                                ignore_metrics=['device__attribute', 'nvlink__'])
                        else:
                            measurements = ncu_report.get_measurements_unmapped()
                        for (callpath, metricId), v in measurements.items():
                            pbar.update(0)
                            aggregated_values[
                                (Callpath(callpath), Metric(ncu_report.string_table[metricId]))].append(v)

                # add measurements to experiment
                for (callpath, metric), values in aggregated_values.items():
                    pbar.update(0)
                    experiment.add_measurement(Measurement(coordinate, callpath, metric, values))

        to_delete = []
        for key, value in pbar.__call__(experiment.measurements.items(), len(experiment.measurements), scale=0.1):
            value: List[Measurement]
            if len(value) < num_points and (
                    not key[0].lookup_tag('validation__ignore__num_measurements', False) or len(value) <= 1):
                to_delete.append(key)
            else:
                (callpath, metric) = key
                # if len(value) < num_points and 'gpu__overlap' in callpath.tags:
                #     # construct empty measurements for overlap
                #     measurements = {c: Measurement(c, callpath, metric, None) for c in experiment.coordinates}
                #     for v in value:
                #         del measurements[v.coordinate]
                #     value.extend(measurements.values())

                experiment.add_callpath(callpath)
                experiment.add_metric(metric)
        for key in to_delete:
            pbar.update(0)
            del experiment.measurements[key]

        # determine calltree
        call_tree = io_helper.create_call_tree(experiment.callpaths, pbar, progress_scale=0.1)
        experiment.call_tree = call_tree

        io_helper.validate_experiment(experiment, pbar)
        pbar.update()
        return experiment
