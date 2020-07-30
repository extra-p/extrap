"""
This file is part of the Extra-P software (https://github.com/extra-p/extrap)

Copyright (c) 2020 Technical University of Darmstadt, Darmstadt, Germany

All rights reserved.
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""
import json
import os
from json import JSONDecodeError

from entities.callpath import Callpath
from entities.coordinate import Coordinate
from entities.experiment import Experiment
from entities.measurement import Measurement
from entities.metric import Metric
from entities.parameter import Parameter
from fileio import io_helper
from fileio.io_helper import create_call_tree
from util.exceptions import FileFormatError
from util.progress_bar import DUMMY_PROGRESS


def read_talpas_file(path, progress_bar=DUMMY_PROGRESS):
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
                raise FileFormatError(f'Decoding of line {ln} failed: {str(error).replace(",", ";")}. Line: "{line}"')
            try:
                key = Callpath(data['callpath']), Metric(data['metric'])
                if parameters is None:
                    parameters = [Parameter(p) for p in data['parameters'].keys()]
                coordinate = Coordinate(data['parameters'][p.name] for p in parameters)
                if key in complete_data:
                    if coordinate in complete_data[key]:
                        complete_data[key][coordinate].append(data['value'])
                    else:
                        complete_data[key][coordinate] = [data['value']]
                        progress_bar.total += 1
                else:
                    complete_data[key] = {
                        coordinate: [data['value']]
                    }
                    progress_bar.total += 1
            except KeyError as error:
                raise FileFormatError(f'Missing property in line {ln}: {str(error)}. Line: "{line}"')

    # create experiment
    progress_bar.step('Creating experiment')
    for mi, key in enumerate(complete_data):
        progress_bar.update()
        callpath, metric = key
        measurementset = complete_data[key]
        experiment.add_callpath(callpath)
        experiment.add_metric(metric)
        for coordinate in measurementset:
            values = measurementset[coordinate]
            experiment.add_coordinate(coordinate)
            experiment.add_measurement(
                Measurement(coordinate, callpath, metric, values))

    for p in parameters:
        experiment.add_parameter(p)

    callpaths = experiment.get_callpaths()
    call_tree = create_call_tree(callpaths, progress_bar)
    experiment.add_call_tree(call_tree)

    io_helper.validate_experiment(experiment, progress_bar)

    return experiment
