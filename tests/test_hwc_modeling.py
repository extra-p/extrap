# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.entities.callpath import Callpath
from extrap.entities.metric import Metric
from extrap.fileio.experiment_io import read_experiment
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.multi_parameter.multi_parameter_hwc_modeler import MultiParameterHwcModeler
from extrap.modelers.single_parameter.hardware_counter_modeler import HardwareCounterModeler
from tests.modelling_testcase import TestCaseWithFunctionAssertions


class TestHwcModeler(TestCaseWithFunctionAssertions):
    def test_mp_hwc_modeler(self):
        experiment = read_experiment('data/input/hwc_experiment/no-noise.extra-p')
        hwcm: HardwareCounterModeler = HardwareCounterModeler()
        hwcm.number_used_counters = 1
        hwcm.counter_list = ['PAPI_TOT_INS']
        hwcm.no_negative_coefficients = False
        modeler_test = MultiParameterHwcModeler()
        modeler_test.no_negative_coefficients = False
        modeler_test.single_parameter_modeler = hwcm
        ModelGenerator(experiment, modeler_test).model_all()

        for k in experiment.modelers[1].models:
            self.assertTrue(k in experiment.modelers[0].models)
        for k in experiment.modelers[0].models:
            self.assertTrue(k in experiment.modelers[1].models)
            if k[1] != Metric('time'):
                self.assertApproxModel(experiment.modelers[0].models[k], experiment.modelers[1].models[k])
            else:
                self.assertApprox(experiment.modelers[0].models[k].hypothesis.function.evaluate((1, 1)),
                                  experiment.modelers[1].models[k].hypothesis.function.evaluate((1, 1)), places=1)
