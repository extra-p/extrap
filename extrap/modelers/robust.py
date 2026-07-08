import signal
import sys
from enum import Enum
from functools import partial
from numbers import Number
from pathlib import Path
from typing import Optional, Sequence, Tuple, Type

import numpy
import random
import copy
from PySide6.QtCore import *  # @UnusedWildImport
from PySide6.QtGui import *  # @UnusedWildImport
from PySide6.QtWidgets import *  # @UnusedWildImport

import extrap

from extrap.entities.callpath import Callpath
from extrap.entities.hypotheses import SingleParameterHypothesis, ConstantHypothesis, MultiParameterHypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.measurement import Measure
from extrap.entities.metric import Metric
from extrap.util.progress_bar import DUMMY_PROGRESS
from extrap.entities.experiment import Experiment

from extrap.modelers.model_generator import ModelGenerator

class RobustModel():

    @staticmethod
    def checkingEqualCallPath(exp_bb, exp_time):
        if len(exp_bb.callpaths) != len(exp_time.callpaths):
            return False

        for path in exp_bb.callpaths:
            if path not in exp_time.callpaths:
                return False

        return True

    @staticmethod
    def creating_robust_experiment(bb_experiment, experiment_TIME):
        experiment_Robust = copy.deepcopy(bb_experiment)
        experiment_Robust.modelers.clear()
        for i in range(0, len(experiment_Robust.callpaths)):
            if "MPI_" in str(experiment_Robust.callpaths[i]):
                try:
                    measurement_values = experiment_TIME.measurements[
                        (Callpath(str(experiment_TIME.callpaths[i])), Metric('bytes_sent'))]
                    experiment_Robust.measurements.update(
                        {(Callpath(str(experiment_TIME.callpaths[i])), Metric('bytes_sent')): measurement_values})
                except KeyError:
                    try:
                        measurement_values = experiment_TIME.measurements[
                            (Callpath(str(experiment_TIME.callpaths[i])), Metric('bytes_received'))
                        ]
                        experiment_Robust.measurements.update(
                            {(Callpath(str(experiment_TIME.callpaths[i])),Metric('bytes_received')):measurement_values})
                    except KeyError:
                        continue

        return experiment_Robust

    @staticmethod
    def createRobustModel(experiment_BB,experiment_TIME,use_measure=Measure.MEAN):
        is_same_callpath : bool = RobustModel.checkingEqualCallPath(exp_bb=experiment_BB,exp_time=experiment_TIME)

        if not is_same_callpath:
             raise Exception("Not equal callpath")

        experiment_Robust = RobustModel.creating_robust_experiment(experiment_BB,experiment_TIME)
        model_Robust = ModelGenerator(experiment=experiment_Robust, name="Robust", use_measure=use_measure)
        model_Robust.model_all()

        RobustModel.manipulate_all_functions(model_Robust, experiment_TIME.measurements)
        return experiment_Robust

    @staticmethod
    def manipulate_all_functions(model_robust, time_measurements):
        models = list(model_robust.models.items())
        for i in range(0, len(models)):
            cur_model_tuple = models[i]  # Creates a Tuple with (Callpath,Metric),Model
            cur_time_measurement = time_measurements[cur_model_tuple[0]]
            if isinstance(cur_model_tuple[1].hypothesis, ConstantHypothesis):
                RobustModel.modeling_constant(cur_model_tuple[1], cur_time_measurement)
            if isinstance(cur_model_tuple[1].hypothesis, SingleParameterHypothesis):
                RobustModel.modeling_with_param(cur_model_tuple[1], cur_time_measurement)
            if isinstance(cur_model_tuple[1].hypothesis, MultiParameterHypothesis):
                RobustModel.modeling_with_param(cur_model_tuple[1], cur_time_measurement)


    @staticmethod
    def modeling_constant(model, measurements):
        # use all available additional points for modeling the multi-parameter models
        model.hypothesis.compute_coefficients(measurements)

    @staticmethod
    def modeling_with_param(model, measurement):
        model.hypothesis.compute_coefficients(measurement)
        model.hypothesis.compute_cost(measurement)
        values = numpy.fromiter(
            Measurement.select_measure(measurement, model.hypothesis._use_measure),
            float,
            len(measurement)
        )
        meanModel = numpy.mean(values)
        constantCost = numpy.sum((values - meanModel) * (values - meanModel))
        model.hypothesis.compute_adjusted_rsquared(constantCost,measurement)