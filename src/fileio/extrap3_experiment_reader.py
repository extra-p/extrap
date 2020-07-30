"""
This file is part of the Extra-P software (https://github.com/extra-p/extrap)

Copyright (c) 2020 Technical University of Darmstadt, Darmstadt, Germany

All rights reserved.

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""
import copy
import logging
import os
import struct
from typing import Optional

from entities.callpath import Callpath
from entities.coordinate import Coordinate
from entities.experiment import Experiment
from entities.functions import Function, SingleParameterFunction, MultiParameterFunction
from entities.hypotheses import Hypothesis
from entities.measurement import Measurement
from entities.metric import Metric
from entities.model import Model
from entities.parameter import Parameter
from entities.terms import CompoundTerm, SimpleTerm, MultiParameterTerm
from fileio import io_helper
from modelers.model_generator import ModelGenerator
from modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler
from util.exceptions import FileFormatError
from util.progress_bar import DUMMY_PROGRESS


class IoTransaction:
    def __init__(self, m_input_stream):
        self._input_stream = m_input_stream
        self._position = m_input_stream.tell()
        self.__commited = False

    def __enter__(self):
        return self

    def commit(self):
        self.__commited = True

    def rollback(self):
        self._input_stream.seek(self._position, 0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None or self.__commited:
            return True
        else:
            self.rollback()
        return isinstance(exc_val, AbortTransaction)


class AbortTransaction(RuntimeError):
    pass


class IoHelper:

    def __init__(self, in_file):
        self.m_input_stream = in_file
        self.byte_order = self.check_system_and_file_endianness()
        self.INT64 = struct.Struct(self.byte_order + 'q')
        self.UINT32 = struct.Struct(self.byte_order + 'I')
        self.DOUBLE = struct.Struct(self.byte_order + 'd')
        self.cached_structs = {}

    def readId(self) -> int:
        s = self.m_input_stream.read(self.INT64.size)
        s = self.INT64.unpack(s)[0]
        return s

    def read_struct(self, compiled_struct):
        s = self.m_input_stream.read(compiled_struct.size)
        s = compiled_struct.unpack(s)
        return s

    def read_pattern(self, pattern):
        if pattern in self.cached_structs:
            compiled_struct = self.cached_structs[pattern]
        else:
            compiled_struct = struct.Struct(self.byte_order + pattern)
            self.cached_structs[pattern] = compiled_struct
        return self.read_struct(compiled_struct)

    def readInt(self) -> int:
        s = self.m_input_stream.read(self.INT64.size)
        s = self.INT64.unpack(s)[0]
        return s

    def read_uint32_t(self) -> Optional[int]:
        s = self.m_input_stream.read(self.UINT32.size)
        if len(s) == 0:
            return None
        s = self.UINT32.unpack(s)[0]
        return s

    def readValue(self) -> float:
        s = self.m_input_stream.read(self.DOUBLE.size)
        s = self.DOUBLE.unpack(s)[0]
        return s

    def readString(self) -> str:
        size = self.read_uint32_t()
        if size is None:
            return ''
        s = self.m_input_stream.read(size)
        s = struct.unpack(self.byte_order + str(size) + 's', s)[0]
        return s.decode("utf-8")

    def check_system_and_file_endianness(self):
        s = self.m_input_stream.read(2)
        s = struct.unpack('<h', s)[0]
        if s == 1:
            return '<'
        else:
            return '>'

    def begin_transaction(self):
        return IoTransaction(self.m_input_stream)


def deserialize_parameter(parameter_mapping, ioHelper):
    id = ioHelper.readId()
    paramName = ioHelper.readString()
    parameter_mapping[paramName] = id
    return Parameter(paramName)


def deserialize_metric(ioHelper):
    id = ioHelper.readId()
    name = ioHelper.readString()
    unit = ioHelper.readString()
    return Metric(name)


def deserialize_region(region_mapping, ioHelper):
    id = ioHelper.readId()
    name = ioHelper.readString()
    sourceFileName = ioHelper.readString()
    sourceFileBeginLine = ioHelper.readInt()
    region_mapping[id] = name


def deserialize_callpath(region_mapping, callpath_mapping, ioHelper):
    id = ioHelper.readId()
    region_id = ioHelper.readId()
    parent_id = ioHelper.readId()

    region_name = region_mapping[region_id]
    if parent_id != -1:
        parent = callpath_mapping[parent_id]
        callpath = Callpath(parent.name + '->' + region_name)
    else:
        callpath = Callpath(region_name)

    callpath_mapping[id] = callpath
    return callpath


def deserialize_coordinate(exp, ioHelper):
    id = ioHelper.readId()
    length = ioHelper.readInt()

    coordinate = [None] * length
    for i in range(length):
        param = Parameter(ioHelper.readString())
        paramIdx = exp.parameters.index(param)
        val = ioHelper.readValue()
        coordinate[paramIdx] = val

    return Coordinate(*coordinate)


def deserialize_modelcomment(ioHelper):
    id = ioHelper.readId()
    comment = ioHelper.readString()


def deserialize_CompoundTerm(ioHelper):
    compoundTerm = CompoundTerm()
    coefficient = ioHelper.readValue()
    compoundTerm.coefficient = coefficient
    length = ioHelper.readInt()
    for i in range(length):
        prefix = ioHelper.readString()
        assert (prefix == 'SimpleTerm')
        term = deserialize_SimpleTerm(ioHelper)
        compoundTerm.add_simple_term(term)

    return compoundTerm


def deserialize_SimpleTerm(ioHelper):
    # Read FunctionType
    functionType = ioHelper.readString()
    exponent = ioHelper.readValue()
    return SimpleTerm(functionType, exponent)


def deserialize_SingleParameterSimpleModelGenerator(exp, ioHelper):
    model_generator = deserialize_SingleParameterModelGenerator(exp, ioHelper)

    # Read CompoundTerms
    length = ioHelper.readInt()
    buildingBlocks = []
    for i in range(0, length):
        prefix = ioHelper.readString()
        assert (prefix == 'CompoundTerm')
        compoundTerm = deserialize_CompoundTerm(ioHelper)

    # Read MaxTermCount
    maxTermCount = ioHelper.readInt()
    return model_generator


def deserialize_SingleParameterModelGenerator(exp, ioHelper):
    userName = ioHelper.readString()
    cvMethod = ioHelper.readString()
    if cvMethod == "CROSSVALIDATION_NONE":
        pass
    elif cvMethod == "CROSSVALIDATION_LEAVE_ONE_OUT":
        pass
    elif cvMethod == "CROSSVALIDATION_LEAVE_P_OUT":
        pass
    elif cvMethod == "CROSSVALIDATION_K_FOLD":
        pass
    elif cvMethod == "CROSSVALIDATION_TWO_FOLD":
        pass
    else:
        logging.error("Invalid Crossvalidation Method found in File. Defaulting to No crossvalidation.")
    eps = ioHelper.readValue()

    # read the options
    generate_strategy = ioHelper.readString()
    # convert generate model options to enum
    if generate_strategy == "GENERATE_MODEL_MEAN":
        use_median = False
    elif generate_strategy == "GENERATE_MODEL_MEDIAN":
        use_median = True
    else:
        use_median = False
        logging.error("Invalid ModelOptions found in File.")

    with ioHelper.begin_transaction():
        min_number_points = ioHelper.readInt()
        single_strategy = ioHelper.readString()
        add_points = ioHelper.readInt()
        number_add_points = ioHelper.readInt()
        multi_strategy = ioHelper.readString()
        # convert ints to bool values
        use_add_points = add_points

        # convert single parameter strategy to enum
        if single_strategy == "FIRST_POINTS_FOUND":
            pass

        elif single_strategy == "MAX_NUMBER_POINTS":
            pass

        elif single_strategy == "CHEAPEST_POINTS":
            pass

        else:
            logging.error("Invalid ModelOptions found in File.")
            raise AbortTransaction
        # convert multi parameter strategy to enum
        if multi_strategy == "INCREASING_COST":
            pass

        elif multi_strategy == "DECREASING_COST":
            pass

        else:
            logging.error("Invalid ModelOptions found in File.")
            raise AbortTransaction

    return ModelGenerator(exp, MultiParameterModeler(), userName, use_median)


def deserialize_MultiParameterModelGenerator(exp, ioHelper):
    userName = ioHelper.readString()

    # read the options
    generate_strategy = ioHelper.readString()

    use_median = False
    # convert generate model options to enum
    if generate_strategy == "GENERATE_MODEL_MEAN":
        use_median = False
    elif generate_strategy == "GENERATE_MODEL_MEDIAN":
        use_median = True
    else:
        logging.error("Invalid ModelOptions found in File.")

    with ioHelper.begin_transaction():

        min_number_points = ioHelper.readInt()
        single_strategy = ioHelper.readString()
        add_points = ioHelper.readInt()
        number_add_points = ioHelper.readInt()
        multi_strategy = ioHelper.readString()

        # convert ints to bool values
        use_add_points = bool(add_points)

        # convert single parameter strategy to enum
        if single_strategy == "FIRST_POINTS_FOUND":
            pass
        elif single_strategy == "MAX_NUMBER_POINTS":
            pass
        elif single_strategy == "CHEAPEST_POINTS":
            pass
        else:
            logging.info("New ModelOptions not found in File.")
            raise AbortTransaction

        # convert multi parameter strategy to enum
        if multi_strategy == "INCREASING_COST":
            pass
        elif multi_strategy == "DECREASING_COST":
            pass
        else:
            logging.info("New ModelOptions not found in File.")
            raise AbortTransaction

    return ModelGenerator(exp, MultiParameterModeler(), userName, use_median)


def deserialize_ExperimentPoint(experiment, callpath_mapping, ioHelper):
    # coordinate_id = ioHelper.readId()
    # sampleCount = ioHelper.readInt()
    # mean = ioHelper.readValue()
    # meanCI_start = ioHelper.readValue()
    # meanCI_end = ioHelper.readValue()
    # standardDeviation = ioHelper.readValue()
    # median = ioHelper.readValue()
    # medianCI_start = ioHelper.readValue()
    # medianCI_end = ioHelper.readValue()
    # minimum = ioHelper.readValue()
    # maximum = ioHelper.readValue()
    # metricId = ioHelper.readId()
    # callpathId = ioHelper.readId()
    coordinate_id, sampleCount, \
    mean, meanCI_start, meanCI_end, \
    standardDeviation, \
    median, medianCI_start, medianCI_end, \
    minimum, maximum, metricId, callpathId = ioHelper.read_pattern('qqdddddddddqq')

    coordinate = experiment.get_coordinate(coordinate_id)
    metric = experiment.get_metric(metricId)
    callpath = callpath_mapping[callpathId]

    point = Measurement(coordinate, callpath, metric, None)
    point.minimum = minimum
    point.maximum = maximum
    point.mean = mean
    point.median = median
    point.std = standardDeviation
    return point


def deserialize_Model(experiment, parameter_mapping, callpath_mapping, ioHelper):
    metricId = ioHelper.readId()
    callpathId = ioHelper.readId()
    metric = experiment.get_metric(metricId)
    callpath = callpath_mapping[callpathId]
    generator_id = ioHelper.readId()

    prefix = ioHelper.readString()
    model_function = deserialize_Function(ioHelper, parameter_mapping, prefix)

    prefix = ioHelper.readString()
    confidence_interval_upper = deserialize_Function(ioHelper, parameter_mapping, prefix)
    SAFE_RETURN_None(confidence_interval_upper)
    prefix = ioHelper.readString()
    confidence_interval_lower = deserialize_Function(ioHelper, parameter_mapping, prefix)
    SAFE_RETURN_None(confidence_interval_lower)

    prefix = ioHelper.readString()
    error_cone_interval_upper = deserialize_Function(ioHelper, parameter_mapping, prefix)
    SAFE_RETURN_None(error_cone_interval_upper)
    prefix = ioHelper.readString()
    error_cone_interval_lower = deserialize_Function(ioHelper, parameter_mapping, prefix)
    SAFE_RETURN_None(error_cone_interval_lower)

    prefix = ioHelper.readString()
    noise_error_interval_upper = deserialize_Function(ioHelper, parameter_mapping, prefix)
    SAFE_RETURN_None(noise_error_interval_upper)
    prefix = ioHelper.readString()
    noise_error_interval_lower = deserialize_Function(ioHelper, parameter_mapping, prefix)
    SAFE_RETURN_None(noise_error_interval_lower)

    RSS = ioHelper.readValue()
    AR2 = ioHelper.readValue()
    SMAPE = ioHelper.readValue()
    RE = ioHelper.readValue()

    length = ioHelper.readInt()
    for i in range(0, length):
        comment_id = ioHelper.readId()

    hypothesis = Hypothesis(model_function, False)
    hypothesis._RSS = RSS
    hypothesis._AR2 = AR2
    hypothesis._SMAPE = SMAPE
    hypothesis._RE = RE
    hypothesis._costs_are_calculated = True
    model = Model(hypothesis, callpath, metric)

    length = ioHelper.readInt()
    for i in range(0, length):
        param = ioHelper.readString()
        num_intervals = ioHelper.readInt()
        for j in range(0, num_intervals):
            interval_start = ioHelper.readValue()
            interval_end = ioHelper.readValue()

    return model, generator_id


def deserialize_Function(ioHelper, parameter_mapping, prefix):
    if prefix == 'SingleParameterFunction':
        # Read a SingleParameterFunction
        f = deserialize_SingleParameterFunction(ioHelper)
        return copy.deepcopy(f)
    elif prefix == 'MultiParameterFunction':
        # Read a MultiParameterFunction
        f = deserialize_MultiParameterFunction(parameter_mapping, ioHelper)
        return f
    elif prefix == 'SimpleTerm':
        # Read a SimpleTerm
        f = deserialize_SimpleTerm(ioHelper)
        return copy.deepcopy(f)
    elif prefix == 'CompoundTerm':
        # Read a CompoundTerm
        f = deserialize_CompoundTerm(ioHelper)
        return copy.deepcopy(f)
    elif prefix == 'Function':
        f = Function()
        return f
    else:
        logging.error("Could not identify Function type: " + prefix)
        return None


def deserialize_SingleParameterFunction(ioHelper):
    function = SingleParameterFunction()
    coefficient = ioHelper.readValue()
    function.constant_coefficient = coefficient
    length = ioHelper.readInt()
    for i in range(0, length):
        prefix = ioHelper.readString()
        assert (prefix == 'CompoundTerm')
        term = deserialize_CompoundTerm(ioHelper)
        function.add_compound_term(term)

    return function


def deserialize_MultiParameterTerm(parameter_mapping, ioHelper):
    new_term = MultiParameterTerm()
    new_term.coefficient = ioHelper.readValue()
    size = ioHelper.readInt()
    for i in range(size):
        name = ioHelper.readString()
        p = parameter_mapping[name]
        prefix = ioHelper.readString()
        assert (prefix == 'CompoundTerm')
        term = deserialize_CompoundTerm(ioHelper)
        new_term.add_parameter_term_pair((p, term))

    return new_term


def deserialize_MultiParameterFunction(parameter_mapping, ioHelper):
    function = MultiParameterFunction()

    function.constant_coefficient = ioHelper.readValue()
    size = ioHelper.readInt()
    for i in range(0, size):
        term = deserialize_MultiParameterTerm(parameter_mapping, ioHelper)
        function.add_multi_parameter_term(term)

    return function


def SAFE_RETURN_None(x):
    if x is None:
        raise FileFormatError()


def read_extrap3_experiment(path, progress_bar=DUMMY_PROGRESS):
    progress_bar.total += os.path.getsize(path)
    with open(path, "rb") as file:
        ioHelper = IoHelper(file)
        qualifier = ioHelper.readString()
        if qualifier != "EXTRAP_EXPERIMENT":
            raise FileFormatError("This is not an Extra-P 3 Experiment File. Qualifier was " + str(qualifier))
        exp = Experiment()
        region_mapping = {}
        callpath_mapping = {}
        parameter_mapping = {}
        versionNumber = ioHelper.readString()
        prefix = ioHelper.readString()
        progress_bar.step('Load Extra-P 3 experiment')
        last_pos = 0
        while prefix:
            pos = file.tell()
            progress_bar.update(pos - last_pos)
            last_pos = pos
            # logging.debug("Deserialize " + str(prefix))
            # noinspection PyNoneFunctionAssignment
            if prefix == 'Parameter':
                p = deserialize_parameter(parameter_mapping, ioHelper)
                exp.add_parameter(p)

            elif prefix == 'Metric':
                m = deserialize_metric(ioHelper)
                SAFE_RETURN_None(m)
                exp.add_metric(m)

            elif prefix == 'Region':
                deserialize_region(region_mapping, ioHelper)

            elif prefix == 'Callpath':
                c = deserialize_callpath(region_mapping, callpath_mapping, ioHelper)
                SAFE_RETURN_None(c)
                exp.add_callpath(c)
                progress_bar.total += 100

            elif prefix == 'Coordinate':
                c = deserialize_coordinate(exp, ioHelper)
                SAFE_RETURN_None(c)
                exp.add_coordinate(c)

            elif prefix == 'ModelComment':
                deserialize_modelcomment(ioHelper)
                # SAFE_RETURN_None(comment)
                # exp.addModelComment(comment)

            elif prefix == 'SingleParameterSimpleModelGenerator':
                generator = deserialize_SingleParameterSimpleModelGenerator(exp, ioHelper)
                SAFE_RETURN_None(generator)
                exp.add_modeler(generator)

            elif prefix == 'SingleParameterRefiningModelGenerator':
                generator = deserialize_SingleParameterModelGenerator(exp, ioHelper)
                SAFE_RETURN_None(generator)
                exp.add_modeler(generator)

            elif prefix == 'MultiParameterSimpleModelGenerator':
                generator = deserialize_MultiParameterModelGenerator(exp, ioHelper)
                SAFE_RETURN_None(generator)
                exp.add_modeler(generator)

            elif prefix == 'MultiParameterSparseModelGenerator':
                generator = deserialize_MultiParameterModelGenerator(exp, ioHelper)
                SAFE_RETURN_None(generator)
                exp.add_modeler(generator)

            elif prefix == 'ExperimentPoint':
                point = deserialize_ExperimentPoint(exp, callpath_mapping, ioHelper)
                SAFE_RETURN_None(point)
                exp.add_measurement(point)

            elif prefix == 'Model':
                model, generator_id = deserialize_Model(exp, parameter_mapping, callpath_mapping, ioHelper)
                SAFE_RETURN_None(model)
                exp.modelers[generator_id].models[(model.callpath, model.metric)] = model

            else:
                raise FileFormatError("Unknown object: " + prefix + ". Can not load experiment.")

            prefix = ioHelper.readString()

        for modeler in exp.modelers:
            for key, model in modeler.models.items():
                model.measurements = exp.measurements.get(key)

        callpaths = exp.callpaths
        call_tree = io_helper.create_call_tree(callpaths, progress_bar, True, progress_scale=100)
        exp.add_call_tree(call_tree)

        io_helper.validate_experiment(exp, progress_bar)
        # new code
        return exp
