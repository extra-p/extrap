# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.entities.functions import MultiParameterFunction
from extrap.entities.hypotheses import MultiParameterHypothesis
from extrap.entities.measurement import Measure
from extrap.modelers.abstract_modeler import MultiParameterModeler as AbstractMultiParameterModeler
from extrap.modelers.modeler_options import modeler_options
from extrap.modelers.multi_parameter.multi_parameter_modeler import MultiParameterModeler
from extrap.modelers.single_parameter.hardware_counter_modeler import AbstractHardwareCounterModeler


@modeler_options
class MpHardwareCounterModeler(AbstractHardwareCounterModeler, AbstractMultiParameterModeler):
    """
    This class represents the hardware counter modeler for multi parameter functions.
    It uses the multi parameter modeler to create models for selected hardware counters and combines them into a
    hypothesis for runtime modeling.
    """

    NAME = 'Hardware Counter MP'
    DESCRIPTION = "Modeler for multi-parameter models based on hardware counters; supports full and sparse modeling."

    source_modeler: AbstractMultiParameterModeler

    def __init__(self):
        """
        Initialize SingleParameterModeler object.
        """
        source_modeler = MultiParameterModeler()
        super().__init__(use_measure=Measure.MEAN, source_modeler=source_modeler,
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
