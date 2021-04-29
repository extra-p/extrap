# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
import multiprocessing
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
from extrap.fileio.file_reader import FileReader
from extrap.fileio.file_reader.nv_reader.ncu_report import NcuReport
from extrap.fileio.file_reader.nv_reader.nsys_db import NsysReport
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import DUMMY_PROGRESS


class NsightFileReader(FileReader):
    NAME = "nsight"
    GUI_ACTION = "Open set of &Nsight files"
    DESCRIPTION = "Load a set of Nsight Systems files and generate a new experiment"
    CMD_ARGUMENT = "--nsight"
    LOADS_FROM_DIRECTORY = True

    def read_experiment(dir_name, pbar=DUMMY_PROGRESS, selected_metrics=None, only_time=True):
        # read the paths of the nsight files in the given directory with dir_name
        path = Path(dir_name)
        if not path.is_dir():
            raise FileFormatError(f'NV file path must point to a directory: {dir_name}')
        nv_files = list(path.glob('[!.]*.sqlite'))
        if not nv_files:
            raise FileFormatError(f'No sqlite files were found in: {dir_name}')
        pbar.total += len(nv_files) + 6
        # iterate over all folders and read the cube profiles in them
        experiment = Experiment()

        pbar.step("Reading NV files")
        parameter_names_initial = []
        parameter_names = []
        parameter_values = []
        parameter_dict = defaultdict(set)
        progress_step_size = 5 / len(nv_files)
        for path_id, path in enumerate(nv_files):
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
        parameter_selection_mask = []
        for i, p in enumerate(parameter_names):
            if len(parameter_dict[p]) > 1:
                experiment.add_parameter(Parameter(p))
                parameter_selection_mask.append(i)

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
                coordinate = Coordinate(parameter_value[i] for i in parameter_selection_mask)
                experiment.add_coordinate(coordinate)

                aggregated_values.clear()
                metric = Metric('time')
                for path, _ in point_group:
                    pbar.update()
                    with NsysReport(path) as parsed:
                        # iterate over all callpaths and get time
                        for id, _, callpath, kernelName, duration, durationGPU, syncType, other_duration in parsed.get_kernel_runtimes():
                            pbar.update(0)

                            if syncType:
                                if kernelName:
                                    aggregated_values[
                                        (Callpath(callpath + "->" + syncType + "->OVERLAP",
                                                  gpu__overlap=True, gpu__overlap__agg=True, agg__usage__disabled=True),
                                         metric)] = [0]

                                    overlap_cp = Callpath(callpath + "->" + syncType + "->OVERLAP->" + kernelName,
                                                          gpu_overlap=True, gpu_kernel=True)
                                    aggregated_values[(overlap_cp, metric)].append(durationGPU / 10 ** 9)
                                else:
                                    if duration:
                                        aggregated_values[(
                                            Callpath(callpath + "->" + syncType), metric)].append(
                                            duration / 10 ** 9)
                                    aggregated_values[(
                                        Callpath(callpath + "->" + syncType + "->WAIT", agg__usage__disabled=True),
                                        metric)].append(
                                        other_duration / 10 ** 9)
                            elif kernelName:
                                if duration:
                                    aggregated_values[(Callpath(callpath + "->" + kernelName), metric)].append(
                                        duration / 10 ** 9)
                                aggregated_values[
                                    (Callpath(callpath + "->" + kernelName + "->GPU", gpu=True), metric)].append(
                                    durationGPU / 10 ** 9)
                            elif duration:
                                aggregated_values[(Callpath(callpath), metric)].append(duration / 10 ** 9)
                        for id, _, callpath, name, duration in parsed.get_os_runtimes():
                            pbar.update(0)
                            if duration:
                                aggregated_values[(Callpath(callpath), metric)].append(duration / 10 ** 9)

                        correponding_ncu_path = Path(path).with_suffix(".nsight-cuprof-report")
                        if correponding_ncu_path.exists() and not only_time:
                            if pool is None:
                                pool = multiprocessing.Pool()
                            with NcuReport(correponding_ncu_path) as ncuReport:
                                measurements = ncuReport.get_measurements_parallel(parsed.get_kernelid_paths(), pool)
                                for (callpath, metricId), v in measurements.items():
                                    aggregated_values[
                                        (Callpath(callpath), Metric(ncuReport.string_table[metricId]))].append(v)

                # add measurements to experiment
                for (callpath, metric), values in aggregated_values.items():
                    pbar.update(0)
                    experiment.add_measurement(Measurement(coordinate, callpath, metric, values))
        finally:
            if pool:
                pool.close()

        pbar.step("Unify calltrees")
        to_delete = []
        # determine common callpaths for common calltree
        # add common callpaths and metrics to experiment
        for key, value in pbar.__call__(experiment.measurements.items(), len(experiment.measurements), scale=0.1):
            value: List[Measurement]
            if len(value) < num_points and 'gpu_overlap' not in key[0].tags:
                to_delete.append(key)
            else:
                (callpath, metric) = key
                # if len(value) < num_points and 'gpu_overlap' in callpath.tags:
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
