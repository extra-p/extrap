"""
This file is part of the Extra-P software (https://github.com/extra-p/extrap)

Copyright (c) 2020 Technical University of Darmstadt, Darmstadt, Germany

All rights reserved.
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""


from entities.parameter import Parameter
from entities.measurement import Measurement
from entities.coordinate import Coordinate
from entities.callpath import Callpath
from entities.metric import Metric
from entities.experiment import Experiment
from fileio.io_helper import compute_repetitions
from fileio.io_helper import create_call_tree
import logging
import json


def read_talpas_file(path,progress_event=lambda _:_):
    # create an experiment object to save the date loaded from the text file
    experiment = Experiment()

    complete_data = {}

    # read talpas file into complete_data
    with open(path) as file:
        for line in file:
            if line.isspace():
                continue
            line = line.replace(';', ',')

            data = json.loads(line)
            key = Callpath(data['callpath']), Metric(data['metric'])
            coordinate = Coordinate([(Parameter(p), v)
                                     for p, v in data['parameters'].items()])
            if key in complete_data:
                if coordinate in complete_data[key]:
                    complete_data[key][coordinate].append(data['value'])
                else:
                    complete_data[key][coordinate] = [data['value']]
            else:
                complete_data[key] = {
                    coordinate: [data['value']]
                }

    # create experiment
    for key in complete_data:
        callpath, metric = key
        measurementset = complete_data[key]
        experiment.add_callpath(callpath)
        experiment.add_metric(metric)
        for coordinate in measurementset:
            values = measurementset[coordinate]
            experiment.add_coordinate(coordinate)
            experiment.add_measurement(
                Measurement(coordinate, callpath, metric, values))

    for p in coordinate._parameters:
        experiment.add_parameter(p)

    callpaths = experiment.get_callpaths()
    call_tree = create_call_tree(callpaths)
    experiment.add_call_tree(call_tree)

    return experiment
