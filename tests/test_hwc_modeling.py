# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
from math import sin, log

from extrap.entities.callpath import Callpath
from extrap.entities.coordinate import Coordinate
from extrap.entities.measurement import Measurement
from extrap.entities.metric import Metric
from extrap.fileio.experiment_io import read_experiment
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.multi_parameter.multi_parameter_hwc_modeler import MultiParameterHwcModeler
from extrap.modelers.single_parameter.hardware_counter_modeler import HardwareCounterModeler
from extrap.util.exceptions import RecoverableError
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

    def test_hwc_modeler(self):

        measurements = self._prepare_measurements()
        for i in range(1, 5):
            hwcm = HardwareCounterModeler()
            hwcm.number_used_counters = i
            models = hwcm.model(measurements)
            self.assertEqual(len(measurements), len(models))
            time_model = None
            for m in models:
                if m.metric.name == 'time':
                    self.assertEqual(None, time_model)
                    time_model = m
            self.assertEqual(i, len(time_model.hypothesis.function.compound_terms))

    def test_warnings(self):
        measurements = self._prepare_measurements()
        hwcm = HardwareCounterModeler()
        hwcm.number_used_counters = 5
        self.assertWarnsRegex(UserWarning, 'The number of measurements is too low for the number of counters.*',
                              hwcm.model, measurements)
        hwcm = HardwareCounterModeler()
        hwcm.number_used_counters = 6
        self.assertWarnsRegex(UserWarning, 'You have selected more counters than available for modeling.*',
                              hwcm.model, measurements)
        hwcm = HardwareCounterModeler()
        hwcm.number_used_counters = 3
        self.assertWarnsRegex(UserWarning, 'Missing counter data completely for at least one callpath.*',
                              hwcm.model, measurements[0:1])
        self.assertWarnsRegex(UserWarning, 'Missing counter data completely for at least one callpath.*',
                              hwcm.model, measurements[0:1] + measurements[-1:])

    def _prepare_measurements(self):
        coordinates = [10, 20, 30, 40, 50]
        main = Callpath('main')
        time = Metric('time')
        PAPI_DP_OPS = Metric('PAPI_DP_OPS')
        PAPI_BR_INS = Metric('PAPI_BR_INS')
        PAPI_LD_INS = Metric('PAPI_LD_INS')
        PAPI_SR_INS = Metric('PAPI_SR_INS')
        PAPI_SP_OPS = Metric('PAPI_SP_OPS')
        PAPI_CA_CLN = Metric('PAPI_CA_CLN')
        PAPI_TOT_INS = Metric('PAPI_TOT_INS')
        measurements = [
            [Measurement(Coordinate(x), main, time,
                         300 + 4 * x ** 2 + 60 * x + 90 * x ** (1 / 2) + 2 * x ** (5 / 4)
                         + 0.5 * x ** (3 / 2) * log(x) / log(2))
             for x in coordinates],
            [Measurement(Coordinate(x), main, PAPI_DP_OPS, 10 + 15 * x) for x in coordinates],
            [Measurement(Coordinate(x), main, PAPI_BR_INS, 120 + 30 * x ** (1 / 2)) for x in coordinates],
            [Measurement(Coordinate(x), main, PAPI_LD_INS, 50 + 5 * x ** 2) for x in coordinates],
            [Measurement(Coordinate(x), main, PAPI_SR_INS, 5 + 2 * x ** (5 / 4)) for x in coordinates],
            [Measurement(Coordinate(x), main, PAPI_SP_OPS, 5 + 0.05 * x ** (3 / 2) * log(x) / log(2)) for x in
             coordinates],
            [Measurement(Coordinate(x), main, PAPI_CA_CLN, 5 + 0.05 * x ** (1 / 2) * log(x) / log(2)) for x in
             coordinates],
        ]
        tot_ins = [Measurement(Coordinate(x), main, PAPI_TOT_INS, 15 + 0.1 * log(x) ** 2 / log(2)) for x in
                   coordinates]
        for measurement_seq in measurements[1:]:
            for index, value in enumerate(measurement_seq):
                tot_ins[index].merge(value)
        measurements.append(tot_ins)
        return measurements

    def _prepare_measurements_tot_ins(self):
        coordinates = [10, 20, 30, 40, 50]
        main = Callpath('main')
        time = Metric('time')
        PAPI_TOT_INS = Metric('PAPI_TOT_INS')
        measurements = [
            [Measurement(Coordinate(x), main, time,
                         300 + 4 * x ** 2 + 60 * x + 90 * x ** (1 / 2) + 2 * x ** (5 / 4)
                         + 0.5 * x ** (3 / 2) * log(x) / log(2))
             for x in coordinates],
            [Measurement(Coordinate(x), main, PAPI_TOT_INS, 10 + 15 * x) for x in coordinates],
        ]
        return measurements

    def test_hwc_modeler_weighted(self):
        measurements = self._prepare_measurements()
        for i in range(1, 6):
            hwcm = HardwareCounterModeler()
            hwcm.weighted_terms = True
            hwcm.number_used_counters = i
            models = hwcm.model(measurements)
            self.assertEqual(len(measurements), len(models))
            time_model = None
            for m in models:
                if m.metric.name == 'time':
                    self.assertEqual(None, time_model)
                    time_model = m
            self.assertEqual(len(measurements) - 2, len(time_model.hypothesis.function.compound_terms))

    def test_warnings_and_errors_weighted(self):
        measurements = self._prepare_measurements()
        hwcm = HardwareCounterModeler()
        hwcm.weighted_terms = True
        hwcm.number_used_counters = 3
        self.assertRaisesRegex(RecoverableError, r'Missing PAPI_TOT_INS, which is necessary for weighting\.',
                               hwcm.model, measurements[0:1])
