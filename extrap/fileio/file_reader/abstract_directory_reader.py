# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import abc
import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path

from extrap.entities.scaling_type import ScalingType
from extrap.fileio.file_reader import FileReader
from extrap.util.dynamic_options import DynamicOptions
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import DUMMY_PROGRESS


class AbstractDirectoryReader(FileReader, abc.ABC):

    def _find_files_in_directory(self, path, glob_pattern, progress_bar=DUMMY_PROGRESS):
        path = Path(path)
        if not path.is_dir():
            raise FileFormatError(f'{self.NAME.capitalize()} file path must point to a directory: {path}')
        files = list(path.glob(glob_pattern))
        if not files:
            raise FileFormatError(f'No {self.NAME} files were found in: {path}')
        progress_bar.total += len(files) + 6
        return files

    @staticmethod
    def _determine_parameters_from_paths(paths, progress_bar=DUMMY_PROGRESS):
        parameter_names_initial = []
        parameter_names = []
        parameter_values = []
        parameter_dict = defaultdict(set)
        progress_step_size = 5 / len(paths)
        for path_id, path in enumerate(paths):
            progress_bar.update(progress_step_size)
            folder_name = path.parent.name
            logging.debug(f"File: {path} Folder: {folder_name}")

            # create the parameters
            par_start = folder_name.find(".") + 1
            par_end = folder_name.find(".r")
            par_end = None if par_end == -1 else par_end
            parameters = folder_name[par_start:par_end]
            # parameters = folder_name.split(".")

            param_list = re.split('([0-9.,]+)', parameters)
            if len(param_list) <= 1:
                raise FileFormatError(f"Could not detect parameter in {folder_name}")
            param_list.remove("")

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

            if len(parameter_names) >= 1 and parameter_names[0] == "scorep-":
                warnings.warn(
                    f"Could not detect any parameter names in the name of folder: {folder_name}. "
                    f"Please follow the usage guide under "
                    f"<a href=https://github.com/extra-p/extrap/blob/master/docs/file-formats.md#cube-file-format>"
                    f"https://github.com/extra-p/extrap/blob/master/docs/file-formats.md#cube-file-format</a>.")

        # determine and remove non-constant parameters

        for i in reversed(range(len(parameter_names))):
            p = parameter_names[i]
            if len(parameter_dict[p]) <= 1:
                for pv in parameter_values:
                    del pv[i]

        parameter_names = [p for p in parameter_names if len(parameter_dict[p]) > 1]

        return parameter_names, parameter_values


class AbstractScalingConversionReader(FileReader, DynamicOptions, abc.ABC):
    scaling_type: ScalingType = DynamicOptions.add(ScalingType.WEAK, ScalingType,
                                                   range={i.name.lower(): i.value for i in ScalingType})
    scaling_type.explanation_below = ("Select the type of scaling analysis.<br>"
                                      "Use <b>strong</b> scaling if the problem size remains unchanged while adding "
                                      "more computational resources (e.g., nodes, processes, cores, threads) are "
                                      "added.<br>"
                                      "If the problem size was scaled alongside the computational resources,"
                                      "choose either <b>weak</b> scaling or <b>weak_threaded</b> scaling when your "
                                      "application uses multithreading.")
