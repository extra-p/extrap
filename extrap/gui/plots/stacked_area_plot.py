# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from matplotlib.axes import Axes, np

from extrap.gui.plots.BaseGraphWidget import GraphDisplayWindow


class StackedAreaPlot(GraphDisplayWindow):
    def draw_figure(self):
        model_list, selected_callpaths = self._get_models_to_draw()
        if model_list is None:
            return
        widget = self.main_widget
        color_map = widget.model_color_map
        ax: Axes = self.figure.add_subplot()
        width, _ = self.get_width_height()
        maxX, maxY = self.get_max()
        x = np.linspace(1.0, maxX, width)

        ax.stackplot(x, [model.hypothesis.function.evaluate(x) for model in model_list],
                     labels=[callpath.name for callpath in selected_callpaths])
        ax.legend(loc="upper left")
        ax.set_ylabel(self.main_widget.get_selected_metric().name)
        self.figure.tight_layout()
