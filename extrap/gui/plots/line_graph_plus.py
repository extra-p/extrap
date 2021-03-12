from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.axes import Axes, np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from extrap.gui.plots.BaseGraphWidget import GraphDisplayWindow


class LineGraphPlus(GraphDisplayWindow):
    def draw_figure(self):
        model_list, selected_callpaths = self.main_widget.get_selected_models()
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
