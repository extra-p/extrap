from abc import abstractmethod, ABC, ABCMeta

import numpy as np
from PySide2.QtWidgets import QSizePolicy
from matplotlib import patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class GraphDisplayWindow(FigureCanvas):
    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):
        self.graphWidget = graphWidget
        self.main_widget = main_widget
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        super().setSizePolicy(QSizePolicy.Expanding,
                              QSizePolicy.Expanding)
        super().updateGeometry()
        self.draw_figure()
        self.fig.tight_layout()

    @abstractmethod
    def draw_figure(self):
        ...

    # noinspection DuplicatedCode
    def _calculate_grid_parameters(self, maxX, maxY):
        if maxX < 10:
            number_of_pixels_x = 45
        elif 10 <= maxX <= 1000:
            number_of_pixels_x = 40
        elif 1000 < maxX <= 1000000000:
            number_of_pixels_x = 15
        else:
            number_of_pixels_x = 5

        if maxY < 10:
            number_of_pixels_y = 45
        elif 10 <= maxY <= 1000:
            number_of_pixels_y = 40
        elif 1000 < maxY <= 1000000000:
            number_of_pixels_y = 15
        else:
            number_of_pixels_y = 5

        pixel_gap_x = self.getPixelGap(0, maxX, number_of_pixels_x)
        pixel_gap_y = self.getPixelGap(0, maxY, number_of_pixels_y)
        return pixel_gap_x, pixel_gap_y

    def getPixelGap(self, lowerlimit, upperlimit, numberOfPixels):
        """
           This function calculate the gap in pixels based on number of pixels and max value
        """
        pixelGap = (upperlimit - lowerlimit) / numberOfPixels
        return pixelGap

    def calculate_z_optimized(self, X, Y, function):
        """
           This function evaluates the function passed to it.
        """
        xs, ys = X.reshape(-1), Y.reshape(-1)
        points = np.ndarray((len(self.main_widget.data_display.parameters), len(xs)))

        parameter_value_list = self.main_widget.data_display.getValues()
        for p, v in parameter_value_list.items():
            points[p] = v
        param1 = self.main_widget.data_display.getAxisParameter(0).id
        param2 = self.main_widget.data_display.getAxisParameter(1).id
        points[param1] = xs
        points[param2] = ys

        z_value = function.evaluate(points)
        return z_value

    def calculate_z_models(self, maxX, maxY, model_list):
        # define grid parameters based on max x and max y value
        pixelGap_x, pixelGap_y = self._calculate_grid_parameters(maxX, maxY)
        # Get the grid of the x and y values
        x = np.arange(1.0, maxX, pixelGap_x)
        y = np.arange(1.0, maxY, pixelGap_y)
        X, Y = np.meshgrid(x, y)
        # Get the z value for the x and y value
        z_List = list()
        Z_List = list()
        for model in model_list:
            function = model.hypothesis.function
            zs = self.calculate_z_optimized(X, Y, function)
            Z = zs.reshape(X.shape)
            z_List.append(zs)
            Z_List.append(Z)
        return X, Y, Z_List, z_List

    def get_selected_models(self):
        selected_metric = self.main_widget.getSelectedMetric()
        selected_callpaths = self.main_widget.getSelectedCallpath()
        if not selected_callpaths:
            return None, None
        model_set = self.main_widget.getCurrentModel().models
        model_list = list()
        for selected_callpath in selected_callpaths:
            model = model_set[selected_callpath.path, selected_metric]
            if model != None:
                model_list.append(model)
        return model_list, selected_callpaths

    def draw_legend(self, ax_all, dict_callpath_color):
        fontSize = self.graphWidget.getFontSize()
        # draw legend
        patches = list()
        for key, value in dict_callpath_color.items():
            labelName = str(key.name)
            if labelName.startswith("_"):
                labelName = labelName[1:]
            patch = mpatches.Patch(color=value, label=labelName)
            patches.append(patch)
        leg = ax_all.legend(handles=patches, fontsize=fontSize,
                            loc="upper right", bbox_to_anchor=(1, 1))
        if leg:
            leg.set_draggable(True)

    def get_max(self, lower_max=2.0):
        # since we are drawing the plots with minimum axis value of 1 to avoid nan values,
        # so the first max-value of parameter could be 2 to calcualte number of subdivisions
        maxX = self.graphWidget.getMaxX()
        maxY = self.graphWidget.getMaxY()
        # define min x and min y value
        if maxX < lower_max:
            maxX = lower_max
        if maxY < lower_max:
            maxY = lower_max
        return maxX, maxY


class BaseContourGraph(GraphDisplayWindow):
    def _calculate_grid_parameters(self, maxX, maxY):
        # define grid parameters based on max x and max y value
        if maxX <= 1000:
            numberOfPixels_x = 100
        elif 1000 < maxX <= 1000000000:
            numberOfPixels_x = 75
        else:
            numberOfPixels_x = 50

        if maxY <= 1000:
            numberOfPixels_y = 100
        elif 1000 < maxY <= 1000000000:
            numberOfPixels_y = 75
        else:
            numberOfPixels_y = 50
        pixelGap_x = self.getPixelGap(0, maxX, numberOfPixels_x)
        pixelGap_y = self.getPixelGap(0, maxY, numberOfPixels_y)
        return pixelGap_x, pixelGap_y
