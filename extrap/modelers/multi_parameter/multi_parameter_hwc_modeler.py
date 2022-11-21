# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
import logging
import warnings
from itertools import accumulate
from typing import Sequence

from extrap.entities.functions import ConstantFunction
from extrap.entities.functions import MultiParameterFunction
from extrap.entities.hypotheses import ConstantHypothesis
from extrap.entities.hypotheses import MultiParameterHypothesis
from extrap.entities.measurement import Measurement
from extrap.entities.model import Model
from extrap.entities.terms import MultiParameterTerm
from extrap.modelers.modeler_options import modeler_options
from extrap.modelers.abstract_modeler import AbstractMultiParameterModeler
from extrap.modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler
from extrap.modelers.single_parameter.basic import SingleParameterModeler
from extrap.modelers.single_parameter.hardware_counter_modeler import AbstractHardwareCounterModeler
from extrap.util.itertools import pairwise
from extrap.util.progress_bar import DUMMY_PROGRESS


@modeler_options
class MpHardwareCounterModeler(AbstractHardwareCounterModeler, AbstractMultiParameterModeler):
    """
    This class represents the modeler for multi parameter functions.
    In order to create a model measurements at least 5 points are needed.
    The result is either a constant function or one based on the PMNF.
    """

    NAME = 'Hardware Counter MP'
    DESCRIPTION = "Modeler for multi-parameter models based on hardware counters; supports full and sparse modeling."

    source_modeler: AbstractMultiParameterModeler

    def __init__(self):
        """
        Initialize SingleParameterModeler object.
        """
        source_modeler = MultiParameterModeler()
        super().__init__(use_median=False, source_modeler=source_modeler,
                         single_parameter_modeler=source_modeler.single_parameter_modeler)

    make_function = MultiParameterFunction
    make_hypothesis = MultiParameterHypothesis

    @property
    def single_parameter_modeler(self):
        return self.source_modeler.single_parameter_modeler

    @single_parameter_modeler.setter
    def single_parameter_modeler(self, value):
        self.source_modeler.single_parameter_modeler = value

    def reset_single_parameter_modeler(self):
        self.source_modeler.reset_single_parameter_modeler()
