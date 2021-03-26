# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import json
import os
from json import JSONDecodeError

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.entities.metric import Metric
from extrap.entities.parameter import Parameter
from extrap.fileio import io_helper
from extrap.fileio.io_helper import create_call_tree
from extrap.util.exceptions import FileFormatError
from extrap.util.progress_bar import DUMMY_PROGRESS


def read_jsonlines_file(path, progress_bar=DUMMY_PROGRESS):
    # create an experiment object to save the date loaded from the text file
    experiment = Experiment()

    complete_data = {}
    parameters = None
    default_callpath = Callpath('<root>')
    default_metric = Metric('<default>')

    progress_bar.total += os.path.getsize(path)

    # read jsonlines file into complete_data
    with open(path) as file:
        progress_bar.step('Reading file')
        for ln, line in enumerate(file):
            progress_bar.update(len(line))
            if line.isspace():
                continue

            try:
                data = json.loads(line)
            except JSONDecodeError as error:
                raise FileFormatError(f'Decoding of line {ln} failed: {str(error)}. Line: "{line}"')
            try:
                if 'callpath' in data:
                    callpath = Callpath(data['callpath'])
                else:
                    callpath = default_callpath

                if 'metric' in data:
                    metric = Metric(data['metric'])
                else:
                    metric = default_metric
                key = callpath, metric
                if parameters is None:  # ensures uniform order of paremeters
                    parameters = [Parameter(p) for p in data['params'].keys()]
                coordinate = Coordinate(data['params'][p.name] for p in parameters)
                io_helper.append_to_repetition_dict(complete_data, key, coordinate, data['value'], progress_bar)
            except KeyError as error:
                raise FileFormatError(f'Missing property in line {ln}: {str(error)}. Line: "{line}"')

    # create experiment
    io_helper.repetition_dict_to_experiment(complete_data, experiment, progress_bar)

    for p in parameters:
        experiment.add_parameter(p)

    callpaths = experiment.callpaths
    experiment.call_tree = create_call_tree(callpaths, progress_bar)

    io_helper.validate_experiment(experiment, progress_bar)

    return experiment
