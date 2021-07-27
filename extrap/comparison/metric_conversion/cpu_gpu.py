# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.comparison.metric_conversion import AbstractMetricConverter, ConversionMetrics, CalculationElement
from extrap.entities.metric import Metric


class FlopsDP(AbstractMetricConverter):
    NAME = "FLOPs DP"

    @ConversionMetrics("PAPI_DP_OPS")
    def _conversion1(self, ops: CalculationElement) -> CalculationElement:
        return ops

    @ConversionMetrics("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
                       "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
                       "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum")
    def _conversion2(self, inst_executed_op_add: CalculationElement, inst_executed_op_mul: CalculationElement,
                     inst_executed_op_fma: CalculationElement) -> CalculationElement:
        return inst_executed_op_add + inst_executed_op_mul + 2 * inst_executed_op_fma


class FlopsSP(AbstractMetricConverter):
    NAME = "FLOPs SP"

    @ConversionMetrics("PAPI_SP_OPS")
    def _conversion1(self, ops: CalculationElement) -> CalculationElement:
        return ops

    @ConversionMetrics("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                       "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
                       "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
    def _conversion2(self, inst_executed_op_add: CalculationElement, inst_executed_op_mul: CalculationElement,
                     inst_executed_op_fma: CalculationElement) -> CalculationElement:
        return inst_executed_op_add + inst_executed_op_mul + 2 * inst_executed_op_fma
