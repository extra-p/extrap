"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""

import itertools
from typing import Dict, Union, Tuple

from entities.callpath import Callpath
from entities.experiment import Experiment
from entities.metric import Metric
from entities.model import Model
from modelers import multi_parameter
from modelers import single_parameter
from modelers.abstract_modeler import AbstractModeler, MultiParameterModeler
from util.deprecation import deprecated
from util.progress_bar import DUMMY_PROGRESS


class ModelGenerator:
    """
    Counter for global modeler ids
    """
    ID_COUNTER = itertools.count()

    def __init__(self, experiment: Experiment,
                 modeler: Union[AbstractModeler, str] = "Default",
                 name: str = "New Modeler",
                 use_median: bool = False):
        self.experiment = experiment
        self.name = name
        self.id = next(__class__.ID_COUNTER)
        # choose the modeler based on the input data
        self._modeler: AbstractModeler = self._choose_modeler(modeler, use_median)
        # all models modeled with this model generator
        self.models: Dict[Tuple[Callpath, Metric], Model] = {}

    @property
    def modeler(self):
        return self._modeler

    def _choose_modeler(self, modeler: Union[AbstractModeler, str], use_median: bool) -> AbstractModeler:
        if isinstance(modeler, str):
            try:
                if len(self.experiment.parameters) == 1:
                    # single parameter model generator init here...
                    result_modeler = single_parameter.all_modelers[modeler]()
                else:
                    # multi parameter model generator init here...
                    result_modeler = multi_parameter.all_modelers[modeler]()
                result_modeler.use_median = use_median
            except KeyError:
                raise ValueError(
                    f'Modeler with name "{modeler}" does not exist.')
        else:
            if (len(self.experiment.parameters) > 1) == isinstance(modeler, MultiParameterModeler):
                # single parameter model generator init here...
                result_modeler = modeler
                if use_median is not None:
                    result_modeler.use_median = use_median
            elif len(self.experiment.parameters) > 1:
                raise ValueError("Modeler must use multiple parameters.")
            else:
                raise ValueError("Modeler must use one parameter.")
        return result_modeler

    def model_all(self, progress_bar=DUMMY_PROGRESS):
        models = self._modeler.model(list(self.experiment.measurements.values()), progress_bar)
        self.models = {
            k: m for k, m in zip(self.experiment.measurements.keys(), models)
        }
        for (callpath, metric), model in self.models.items():
            model.callpath = callpath
            model.metric = metric
            model.measurements = self.experiment.measurements[(callpath, metric)]
        # add the modeler with the results to the experiment
        self.experiment.add_modeler(self)

    @deprecated("Use property directly.")
    def set_name(self, name):
        self.name = name

    @deprecated("Use property directly.")
    def get_name(self):
        return self.name

    @deprecated("Use property directly.")
    def get_id(self):
        return self.id

    @deprecated("Use indexer.")
    def get_model(self, callpath_id, metric_id):
        callpath = self.experiment.callpaths[callpath_id]
        metric = self.experiment.metrics[metric_id]
        return self.models[(callpath, metric)]
