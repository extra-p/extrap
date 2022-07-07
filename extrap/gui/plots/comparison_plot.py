# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from typing import cast

import numpy
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

from extrap.comparison.entities.comparison_model import ComparisonModel
from extrap.comparison.experiment_comparison import ComparisonExperiment
from extrap.gui.plots.BaseGraphWidget import BaseContourGraph


class ComparisonPlotColorbarFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        result = super(ComparisonPlotColorbarFormatter, self).__call__(x, pos)
        if result.startswith('_'):
            result = result[1:]
        return result


class ComparisonPlot(BaseContourGraph):
    colormap = 'RdBu'

    def __init__(self, *args, **kwargs):
        super(ComparisonPlot, self).__init__(*args, **kwargs)

    def _handle_min_max_update(self, *args):
        self._draw_cross_hair()
        self.fig.canvas.draw_idle()

    def hideEvent(self, evt):
        self.main_widget.min_max_value_updated_event -= self._handle_min_max_update
        return super(ComparisonPlot, self).hideEvent(evt)

    def showEvent(self, evt):
        self.main_widget.min_max_value_updated_event += self._handle_min_max_update
        return super(ComparisonPlot, self).showEvent(evt)

    def _draw_cross_hair(self):
        selected_values = self.main_widget.selector_widget.getParameterValues()
        x_value = selected_values[self.main_widget.data_display.getAxisParameter(0).id]
        y_value = selected_values[self.main_widget.data_display.getAxisParameter(1).id]
        max_x, max_y = self.get_max()
        pixel_gap_x, pixel_gap_y = self._calculate_grid_parameters(max_x, max_y)
        max_x -= pixel_gap_x / 2
        max_y -= pixel_gap_y / 2
        display_x_value = 1 - pixel_gap_x / 2 <= x_value <= max_x
        display_y_value = 1 - pixel_gap_x / 2 <= y_value <= max_y
        for ax in self.fig.axes:
            ax: Axes

            if not hasattr(ax, 'extra_p_type') or getattr(ax, 'extra_p_type') != 'plot':
                continue

            for i, line in reversed(list(enumerate(ax.lines))):
                if not hasattr(ax, 'extra_p_type') or getattr(ax, 'extra_p_type') != 'plot':
                    continue
                del ax.lines[i]

            if display_x_value:
                line = ax.plot([x_value, x_value], [1 - pixel_gap_y / 2, max_y], ':k')
                line[0].extra_p_type = 'crosshair'
            if display_y_value:
                line = ax.plot([1 - pixel_gap_x / 2, max_x], [y_value, y_value], ':k')
                line[0].extra_p_type = 'crosshair'

    def draw_figure(self):
        """
          This function draws the graph
        """

        # Get data
        model_list1, selected_call_nodes1 = self.main_widget.get_selected_models()
        if not model_list1:
            return None, None
        model_list_a = []
        model_list_b = []
        selected_call_nodes = []
        metric = None
        for i, (model, call_node) in enumerate(zip(model_list1, selected_call_nodes1)):
            if isinstance(model, ComparisonModel):
                model_list_a.append(model.models[0])
                model_list_b.append(model.models[-1])
                selected_call_nodes.append(call_node)
                metric = model.metric

        if model_list_a is None or model_list_b is None:
            return

        assert len(model_list_a) == len(model_list_b)

        model_list = model_list_a + model_list_b
        # Get max x and max y value as a initial default value or a value provided by user
        max_x, max_y = self.get_max()

        X, Y, Z_List_raw, z_List = self.calculate_z_models(max_x, max_y, model_list)
        Z_List_raw = numpy.array(Z_List_raw)
        Z_List = Z_List_raw[:len(model_list_a)] - Z_List_raw[len(model_list_a):]

        # Get the callpath color map
        widget = self.main_widget
        dict_callpath_color = widget.model_color_map

        # Adjusting subplots in order to avoid overlapping of labels
        # Reference : https://stackoverflow.com/questions/2418125/matplotlib-subplots-adjust-hspace-so-titles-and-xlabels-dont-overlap
        left = 0.1
        right = 0.9
        bottom = 0.2
        top = 0.9
        wspace = 0.5
        hspace = 0.2
        self.fig.subplots_adjust(
            left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        # Set the x_label and y_label based on parameter selected.
        x_label = self.main_widget.data_display.getAxisParameter(0).name
        if x_label.startswith("_"):
            x_label = x_label[1:]

        y_label = self.main_widget.data_display.getAxisParameter(1).name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        # # Set the axis details for the subplot where we will draw all isolines
        # ax_all = self.fig.add_subplot()
        # ax_all.set_xlabel('\n' + x_label)
        # ax_all.set_ylabel('\n' + y_label)
        # ax_all.set_title(r'All')
        # for item in ([ax_all.title, ax_all.xaxis.label, ax_all.yaxis.label]):
        #     item.set_fontsize(10)

        # Draw isolines
        colormap = get_cmap(self.colormap)
        red_patch = tuple(Patch(color=colormap(p / 100)) for p in range(0, 50, 2))
        blue_patch = tuple(Patch(color=colormap(p / 100)) for p in range(52, 102, 2))
        experiment_names = cast(ComparisonExperiment, self.main_widget.getExperiment()).experiment_names
        labels = [f'{experiment_names[0]}$\prec${experiment_names[-1]}',
                  f'{experiment_names[0]}$\succ${experiment_names[-1]}']

        for i in range(len(Z_List)):
            ax = self.fig.add_subplot(1, len(Z_List), i + 1)
            ax.extra_p_type = 'plot'
            ax.set_title(selected_call_nodes[i].name)
            ax.xaxis.major.formatter._useMathText = True
            ax.yaxis.major.formatter._useMathText = True
            try:
                pcm = ax.pcolormesh(X, Y, Z_List[i], cmap=self.colormap, norm=TwoSlopeNorm(0), shading='auto')
                cb = self.fig.colorbar(pcm, ax=ax, label="Difference " + metric.name,
                                       format=ComparisonPlotColorbarFormatter())

                ax.legend([red_patch, blue_patch], labels,
                          handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})


            except ValueError:  # raised if function selected is constant
                pass
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            self.fig.subplots_adjust(
                left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(10)

        self._draw_cross_hair()
        # draw legend
        # self.draw_legend(ax_all, dict_callpath_color)
