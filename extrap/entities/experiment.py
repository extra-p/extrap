# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
from itertools import chain
from typing import List, Dict, Tuple

from marshmallow import fields, validate, pre_load

import extrap
from extrap.entities.callpath import Callpath, CallpathSchema
from extrap.entities.calltree import CallTree
from extrap.entities.coordinate import Coordinate
from extrap.entities.measurement import Measurement, MeasurementSchema
from extrap.entities.metric import Metric, MetricSchema
from extrap.entities.parameter import Parameter, ParameterSchema
from extrap.fileio import io_helper
from extrap.modelers.model_generator import ModelGenerator, ModelGeneratorSchema
from extrap.util.deprecation import deprecated
from extrap.util.progress_bar import DUMMY_PROGRESS
from extrap.util.serialization_schema import Schema, TupleKeyDict
from extrap.util.unique_list import UniqueList


class Experiment:

    def __init__(self):
        self.callpaths: List[Callpath] = UniqueList()
        self.metrics: List[Metric] = UniqueList()
        self.parameters: List[Parameter] = UniqueList()
        self.coordinates: List[Coordinate] = UniqueList()
        self.measurements: Dict[Tuple[Callpath,
                                      Metric], List[Measurement]] = {}
        self.call_tree: CallTree = None
        self.modelers: List[ModelGenerator] = []
        self.scaling = None

    def add_modeler(self, modeler):
        self.modelers.append(modeler)

    def add_metric(self, metric):
        self.metrics.append(metric)

    def add_parameter(self, parameter: Parameter):
        self.parameters.append(parameter)

    def add_coordinate(self, coordinate):
        self.coordinates.append(coordinate)

    def add_callpath(self, callpath: Callpath):
        self.callpaths.append(callpath)

    @deprecated("Use property directly.")
    def get_measurement(self, coordinate_id, callpath_id, metric_id):
        callpath = self.callpaths[callpath_id]
        metric = self.metrics[metric_id]
        coordinate = self.coordinates[coordinate_id]

        try:
            measurements = self.measurements[(callpath, metric)]
        except KeyError as e:
            return None

        for measurement in measurements:
            if measurement.coordinate == coordinate:
                return measurement
        return None

    def add_measurement(self, measurement: Measurement):
        key = (measurement.callpath,
               measurement.metric)
        if key in self.measurements:
            self.measurements[key].append(measurement)
        else:
            self.measurements[key] = [measurement]

    def clear_measurements(self):
        self.measurements = {}

    def debug(self):
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            return
        for i in range(len(self.metrics)):
            logging.debug("Metric " + str(i + 1) + ": " + self.metrics[i].name)
        for i in range(len(self.parameters)):
            logging.debug("Parameter " + str(i + 1) + ": " +
                          self.parameters[i].name)
        for i in range(len(self.callpaths)):
            logging.debug("Callpath " + str(i + 1) + ": " +
                          self.callpaths[i].name)
        for i, coordinate in enumerate(self.coordinates):
            logging.debug(f"Coordinate {i + 1}: {coordinate}")
        for i, measurement in enumerate(chain.from_iterable(self.measurements.values())):
            callpath = measurement.callpath
            metric = measurement.metric
            coordinate = measurement.coordinate
            value_mean = measurement.mean
            value_median = measurement.median
            logging.debug(
                f"Measurement {i}: {metric}, {callpath}, {coordinate}: {value_mean} (mean), {value_median} (median)")


class ExperimentSchema(Schema):
    _version_ = fields.Constant(extrap.__version__, data_key=extrap.__title__)
    scaling = fields.Str(required=False, allow_none=True, validate=validate.OneOf(['strong', 'weak']))
    parameters = fields.List(fields.Nested(ParameterSchema))
    measurements = TupleKeyDict(keys=(fields.Nested(CallpathSchema), fields.Nested(MetricSchema)),
                                values=fields.List(fields.Nested(MeasurementSchema, exclude=('callpath', 'metric'))))

    modelers = fields.List(fields.Nested(ModelGeneratorSchema), missing=[], required=False)

    def set_progress_bar(self, pbar):
        self.context['progress_bar'] = pbar

    @pre_load
    def add_progress(self, data, **kwargs):
        if 'progress_bar' in self.context:
            pbar = self.context['progress_bar']
            models = 0
            ms = data.get('measurements')
            if ms:
                for cp in ms.values():
                    for m in cp.values():
                        models += 1
                        pbar.total += len(m)
            pbar.total += models
            ms = data.get('modelers')
            if ms:
                pbar.total += len(ms)
                pbar.total += len(ms) * models
            pbar.update(0)
        return data

    def create_object(self):
        return Experiment()

    def postprocess_object(self, obj: Experiment):
        if 'progress_bar' in self.context:
            pbar = self.context['progress_bar']
        else:
            pbar = DUMMY_PROGRESS

        for (callpath, metric), measurement in obj.measurements.items():
            obj.add_callpath(callpath)
            obj.add_metric(metric)
            for m in measurement:
                obj.add_coordinate(m.coordinate)
                m.callpath = callpath
                m.metric = metric
            pbar.update()

        obj.call_tree = io_helper.create_call_tree(obj.callpaths)
        for modeler in obj.modelers:
            modeler.experiment = obj
            for key, model in modeler.models.items():
                model.measurements = obj.measurements[key]
            pbar.update()

        return obj
