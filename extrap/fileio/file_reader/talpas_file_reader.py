# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import json
import os
from abc import ABC
from json import JSONDecodeError
from pathlib import Path
from typing import Union

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.fileio.file_reader import FileReader
from extrap.fileio.io_helper import create_call_tree
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import DUMMY_PROGRESS, ProgressBar


class TalpasFileReader(FileReader, ABC):
    NAME = "talpas"
    SHORT_NAME = "Open Tal&pas input"
    LONG_NAME = "Open Talpas input file"
    FILTER = "Talpas Files (*.txt);;All Files (*)"
    CMD_ARGUMENT = "--talpas"

    def read_experiment(self, path: Union[Path, str], progress_bar: ProgressBar = DUMMY_PROGRESS) -> Experiment:
        # create an experiment object to save the date loaded from the text file
        experiment = Experiment()

        complete_data = {}
        parameters = None

        progress_bar.total += os.path.getsize(path)
        # read talpas file into complete_data
        with open(path) as file:

            progress_bar.step('Reading file')
            for ln, line in enumerate(file):
                progress_bar.update(len(line))
                if line.isspace():
                    continue
                line = line.replace(';', ',')

                try:
                    data = json.loads(line)
                except JSONDecodeError as error:
                    raise FileFormatError(
                        f'Decoding of line {ln} failed: {str(error).replace(",", ";")}. Line: "{line}"')
                try:
                    key = Callpath(data['callpath']), Metric(data['metric'])
                    if parameters is None:
                        parameters = [Parameter(p) for p in data['parameters'].keys()]
                    coordinate = Coordinate(data['parameters'][p.name] for p in parameters)
                    io_helper.append_to_repetition_dict(complete_data, key, coordinate, data['value'], progress_bar)
                except KeyError as error:
                    raise FileFormatError(f'Missing property in line {ln}: {str(error)}. Line: "{line}"')

        # create experiment
        io_helper.repetition_dict_to_experiment(complete_data, experiment, progress_bar)

        for p in parameters:
            experiment.add_parameter(p)

        call_tree = create_call_tree(experiment.callpaths, progress_bar)
        experiment.call_tree = call_tree

        io_helper.validate_experiment(experiment, progress_bar)

        return experiment
