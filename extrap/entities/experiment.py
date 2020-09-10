import logging
from itertools import chain
from typing import List, Dict, Tuple

from marshmallow import fields, validate

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

    @property
    @deprecated("Use modelers property instead.")
    def modeler(self):
        return self.modelers

    @deprecated("Use property directly.")
    def set_scaling(self, scaling_type):
        self.scaling = scaling_type

    @deprecated("Use property directly.")
    def get_scaling(self):
        return self.scaling

    @deprecated("Use property directly.")
    def get_modeler(self, modeler_id):
        return self.modelers[modeler_id]

    def add_modeler(self, modeler):
        self.modelers.append(modeler)

    @deprecated("Use property directly.")
    def get_new_modeler_id(self):
        return len(self.modeler) + 1

    @deprecated("Use property directly.")
    def get_call_tree(self):
        return self.call_tree

    @deprecated("Use property directly.")
    def add_call_tree(self, call_tree):
        self.call_tree = call_tree

    def add_metric(self, metric):
        self.metrics.append(metric)

    @deprecated("Use property directly.")
    def get_metric(self, metric_id):
        return self.metrics[metric_id]

    @deprecated("Use property directly.")
    def get_metrics(self):
        return self.metrics

    @deprecated("Use property directly.")
    def get_len_metrics(self):
        return len(self.metrics)

    @deprecated("Use property directly.")
    def get_metric_id(self, metric_name):
        for metric_id in range(len(self.metrics)):
            if self.metrics[metric_id].name == metric_name:
                return metric_id
        return -1

    @deprecated("Use property directly.")
    def metric_exists(self, metric_name):
        return Metric(metric_name) in self.metrics

    def add_parameter(self, parameter: Parameter):
        self.parameters.append(parameter)

    @deprecated("Use property directly.")
    def get_parameter_id(self, parameter_name):
        for parameter_id in range(len(self.parameters)):
            if self.parameters[parameter_id].name == parameter_name:
                return parameter_id
        return -1

    @deprecated("Use property directly.")
    def parameter_exists(self, parameter_name):
        for parameter_id in range(len(self.parameters)):
            if self.parameters[parameter_id].name == parameter_name:
                return True
        return False

    @deprecated("Use property directly.")
    def get_parameter(self, parameter_id):
        return self.parameters[parameter_id]

    @deprecated("Use property directly.")
    def get_parameters(self):
        return self.parameters

    def add_coordinate(self, coordinate):
        self.coordinates.append(coordinate)

    @deprecated("Use property directly.")
    def get_coordinate(self, coordinate_id):
        return self.coordinates[coordinate_id]

    @deprecated("Use property directly.")
    def get_coordinates(self):
        return self.coordinates

    @deprecated("Use property directly.")
    def get_coordinate_id(self, coordinate):
        for coordinate_id in range(len(self.coordinates)):
            if self.coordinates[coordinate_id].get_as_string() == coordinate.get_as_string():
                return coordinate_id
        return -1

    @deprecated("Use property directly.")
    def coordinate_exists(self, coordinate):
        for coordinate_id in range(len(self.coordinates)):
            if self.coordinates[coordinate_id].get_as_string() == coordinate.get_as_string():
                return True
        return False

    @deprecated("Use property directly.")
    def get_len_coordinates(self):
        return len(self.coordinates)

    def add_callpath(self, callpath: Callpath):
        self.callpaths.append(callpath)

    @deprecated("Use property directly.")
    def get_len_callpaths(self):
        return len(self.callpaths)

    @deprecated("Use property directly.")
    def get_callpath(self, callpath_id):
        return self.callpaths[callpath_id]

    @deprecated("Use property directly.")
    def get_callpaths(self):
        return self.callpaths

    @deprecated("Use property directly.")
    def get_callpath_id(self, callpath_name):
        for callpath_id in range(len(self.callpaths)):
            if self.callpaths[callpath_id].name == callpath_name:
                return callpath_id
        return -1

    @deprecated("Use property directly.")
    def callpath_exists(self, callpath_name):
        for callpath_id in range(len(self.callpaths)):
            if self.callpaths[callpath_id].name == callpath_name:
                return True
        return False

    @deprecated("Use property directly.")
    def get_measurements(self):
        return list(chain.from_iterable(self.measurements.values()))

    @deprecated("Use property directly.")
    def get_measurement(self, coordinate_id, callpath_id, metric_id):
        callpath = self.callpaths[callpath_id]
        metric = self.metrics[metric_id]
        coordinate = self.coordinates[coordinate_id]

        measurements = self.measurements[(callpath, metric)]

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
        for i in range(len(self.coordinates)):
            dimensions = self.coordinates[i].get_dimensions()
            coordinate_string = "Coordinate " + str(i + 1) + ": ("
            for dimension in range(dimensions):
                parameter, value = self.coordinates[i].get_parameter_value(
                    dimension)
                parameter_name = parameter.name
                coordinate_string += parameter_name + "=" + str(value) + ","
            coordinate_string = coordinate_string[:-1]
            coordinate_string += ")"
            logging.debug(coordinate_string)
        for i, measurement in enumerate(chain.from_iterable(self.measurements.values())):
            callpath = measurement.callpath
            metric = measurement.metric
            coordinate = measurement.coordinate
            value_mean = measurement.get_value_mean()
            value_median = measurement.get_value_median()
            logging.debug(
                f"Measurement {i}: {metric}, {callpath}, {coordinate}: {value_mean} (mean), {value_median} (median)")


class ExperimentSchema(Schema):
    _version_ = fields.Constant(extrap.__version__, data_key=extrap.__title__)
    scaling = fields.Str(required=False, allow_none=True, validate=validate.OneOf(['strong', 'weak']))
    parameters = fields.List(fields.Nested(ParameterSchema))
    measurements = TupleKeyDict(keys=(fields.Nested(CallpathSchema), fields.Nested(MetricSchema)),
                                values=fields.List(fields.Nested(MeasurementSchema, exclude=('callpath', 'metric'))))

    modelers = fields.List(fields.Nested(ModelGeneratorSchema), default=[], required=False)

    def create_object(self):
        return Experiment()

    def postprocess_object(self, obj: Experiment):
        for (callpath, metric), measurement in obj.measurements.items():
            obj.add_callpath(callpath)
            obj.add_metric(metric)
            for m in measurement:
                obj.add_coordinate(m.coordinate)
                m.callpath = callpath
                m.metric = metric
        obj.call_tree = io_helper.create_call_tree(obj.callpaths)
        for modeler in obj.modelers:
            for key, model in modeler.models.items():
                model.measurements = obj.measurements[key]

        return obj
