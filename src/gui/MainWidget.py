"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license.  See the COPYING file in the package base
directory for details.
"""


import signal
try:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    pyqt_version = 4
except ImportError:
    from PyQt5.QtGui import *  # @UnusedWildImport
    from PyQt5.QtCore import *  # @UnusedWildImport
    from PyQt5.QtWidgets import *  # @UnusedWildImport
    pyqt_version = 5
from gui.ColorWidget import ColorWidget
from gui.CubeFileReader import CubeFileReader
from gui.DataDisplay import DataDisplayManager
from gui.ModelerWidget import ModelerWidget
from gui.PlotTypeSelector import PlotTypeSelector
from gui.SelectorWidget import SelectorWidget
from gui.Utils import formatNumber
from fileio.text_file_reader import read_text_file
from fileio.talpas_file_reader import read_talpas_file
from fileio.json_file_reader import read_json_file
from enum import Enum
from modelers.model_generator import ModelGenerator


class CallPathEnum(Enum):
    constant = "constant"
    logarithmic = "logarithmic"
    polynomial = "polynomial"
    exponential = "exponential"


class MainWidget(QMainWindow):

    def __init__(self, *args, **kwargs):
        """
        Initializes the main application widget.
        """
        super(MainWidget, self).__init__(*args, **kwargs)
        self.old_x_pos = 0
        self.experiment = None
        self.graph_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                 '#bcbd22', '#17becf']
        # ['#8B0000', '#00008B', '#006400', '#2F4F4F', '#8B4513', '#556B2F',
        #  '#808000', '#008080', '#FF00FF', '#800000', '#FF0000', '#000080', '#008000', '#00FFFF', '#800080']
        self.font_size = 6
        self.experiment_change = True
        self.initUI()
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # switch for using mean or median measurement values for modeling
        # is used when loading the data from a file and then modeling directly
        self.median = 0

    def initUI(self):
        """
        Initializes the User Interface of the main widget. E.g. the menus.
        """
        self.setWindowTitle('Extra-P')
        # Status bar
        self.statusBar()

        #central_widget = QWidget(self)
        #top_widget = QWidget(central_widget)
        #bottom_widget = QWidget(central_widget)

        # Main splitter
        #hsplitter = QSplitter(top_widget)

        # Left side: Callpath and metric selection
        dock = QDockWidget(self.tr("Selection"), self)
        self.selector_widget = SelectorWidget(self, dock)
        dock.setWidget(self.selector_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        # middle: Graph

        self.data_display = DataDisplayManager(self, self)
        central_widget = self.data_display

        # Right side: Model configurator
        dock = QDockWidget(self.tr("Modeler"), self)
        self.modeler_widget = ModelerWidget(self, dock)
        dock.setWidget(self.modeler_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # Set splitter sizes
        # w = self.width()
        # sizes = [w/3, w/3, w/3]
        # hsplitter.setSizes(sizes)

        # top widget
        # grid = QGridLayout(top_widget)
        # grid.addWidget(hsplitter, 0, 1)

        # bottom widget
        dock = QDockWidget(self.tr("Color Info"), self)
        bottom_widget = QWidget(central_widget)
        grid = QGridLayout(bottom_widget)
        self.min_value = 1
        self.min_value_label = QLabel(formatNumber(str(self.min_value)))
        self.max_value = 1
        self.max_value_label = QLabel(formatNumber(str(self.max_value)))
        self.color_widget = ColorWidget(bottom_widget)
        grid.addWidget(self.min_value_label, 0, 0)
        grid.addWidget(self.max_value_label, 0, 20)
        grid.addWidget(self.color_widget, 1, 0, 1, 21)
        bottom_widget.setLayout(grid)
        dock.setWidget(bottom_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

        # central_widget
        # grid = QGridLayout(central_widget)
        # central_widget.setLayout(grid)
        # grid.addWidget(top_widget, 0, 0, 50, 0)
        # grid.addWidget(bottom_widget, 51, 0, 2, 0)

        # Menu creation

        # File menu
        screenshot_action = QAction(self.tr('Screenshot'), self)
        screenshot_action.setShortcut('Ctrl+P')
        screenshot_action.setStatusTip(self.tr('Creates a screenshot of the Extra-P GUI'))
        screenshot_action.triggered.connect(self.screenshot)

        exit_action = QAction(self.tr('Exit'), self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip(self.tr('Exit application'))
        exit_action.triggered.connect(self.close)

        open_text_file_action = QAction(self.tr('Open text input'), self)
        open_text_file_action.setStatusTip(self.tr('Open text input file'))
        open_text_file_action.triggered.connect(self.open_text_file)

        open_json_file_action = QAction(self.tr('Open json input'), self)
        open_json_file_action.setStatusTip(self.tr('Open json input file'))
        open_json_file_action.triggered.connect(self.open_json_file)

        open_talpas_file_action = QAction(self.tr('Open talpas input'), self)
        open_talpas_file_action.setStatusTip(self.tr('Open talpas input file'))
        open_talpas_file_action.triggered.connect(self.open_talpas_file)

        cube_file_action = QAction(self.tr('Open set of CUBE files'), self)
        cube_file_action.setStatusTip(self.tr('Open a set of CUBE files for single parameter models and generate data points for a new experiment from them'))
        cube_file_action.triggered.connect(self.open_cube_file)

        # View menu
        change_font_action = QAction(self.tr('Legend font size'), self)
        change_font_action.setStatusTip(self.tr('Change the legend font size'))
        change_font_action.triggered.connect(self.open_font_dialog_box)

        select_view_action = QAction(self.tr('Select Plot Type'), self)
        select_view_action.setStatusTip(self.tr('Select the Plots you want to view'))
        select_view_action.triggered.connect(self.open_select_plots_dialog_box)

        # Plots menu
        graphs = ['Line graph', 'Selected models in same surface plot', 'Selected models in different surface plots', 'Dominating models in a 3D Scatter plot',
                  'Max z as a single surface plot', 'Dominating models and max z as heat map', 'Selected models in contour plot', 'Selected models in interpolated contour plots', 'Measurement points']
        graph_actions = [QAction(self.tr(g), self) for g in graphs]
        for i, g in enumerate(graph_actions):
            slot = (lambda k: lambda: self.data_display.reloadTabs((k,)))(i)
            g.triggered.connect(slot)

        # Model menu
        model_delete_action = QAction(self.tr('Delete model'), self)
        model_delete_action.setShortcut('Ctrl+D')
        model_delete_action.setStatusTip(self.tr('Delete the current model'))
        model_delete_action.triggered.connect(self.selector_widget.model_delete)

        model_rename_action = QAction(self.tr('Rename model'), self)
        model_rename_action.setShortcut('Ctrl+R')
        model_rename_action.setStatusTip(self.tr('Rename the current model'))
        model_rename_action.triggered.connect(self.selector_widget.model_rename)

        # Filter menu
        filter_callpath_action = QAction(self.tr('Filter Callpaths'), self)
        filter_callpath_action.setShortcut('Ctrl+F')
        filter_callpath_action.setStatusTip(self.tr('Select the callpath you want to hide'))
        filter_callpath_action.triggered.connect(self.hide_callpath_dialog_box)

        # Main menu bar
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        file_menu = menubar.addMenu(self.tr('&File'))
        file_menu.addAction(cube_file_action)
        file_menu.addAction(open_text_file_action)
        file_menu.addAction(open_json_file_action)
        file_menu.addAction(open_talpas_file_action)
        file_menu.addAction(screenshot_action)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu(self.tr('View'))
        view_menu.addAction(change_font_action)
        view_menu.addAction(select_view_action)

        plots_menu = menubar.addMenu(self.tr('Plots'))
        for g in graph_actions:
            plots_menu.addAction(g)

        model_menu = menubar.addMenu(self.tr('Model'))
        model_menu.addAction(model_delete_action)
        model_menu.addAction(model_rename_action)

        filter_menu = menubar.addMenu(self.tr('Filter'))
        filter_menu.addAction(filter_callpath_action)

        # Main window
        self.resize(1200, 800)
        self.setCentralWidget(central_widget)
        self.experiment_change = False
        self.show()

    def get_file_name(self, fileName):
        """
        Extracts the file name of the return value of the
        QFileDialog functions getOpenFileName and getSaveFileName.
        In Qt5 and Qt4 they have different return values.
        This function should hide the differences.
        """
        if pyqt_version == 5:
            return fileName[0]
        else:
            return fileName

    def setExperiment(self, experiment):
        self.experiment_change = True
        self.experiment = experiment
        self.selector_widget.updateModelList()
        self.selector_widget.fillMetricList()
        self.selector_widget.createParameterSliders()
        self.selector_widget.fillCalltree()
        self.updateMinMaxValue()
        self.selector_widget.tree_model.valuesChanged()
        self.data_display.experimentChange()
        self.experiment_change = False
        self.update()

    def updateAllWidget(self):
        if not self.experiment_change:
            self.data_display.updateWidget()
            self.update()

    def metricIndexChanged(self):
        if not self.experiment_change:
            self.data_display.updateWidget()
            self.update()
            self.updateMinMaxValue()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, self.tr('Quit'), self.tr(
            "Are you sure to quit?"), QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def getExperiment(self):
        return self.experiment

    def getSelectedMetric(self):
        return self.selector_widget.getSelectedMetric()

    def getSelectedCallpath(self):
        return self.selector_widget.getSelectedCallpath()

    def getCurrentModel(self):
        return self.selector_widget.getCurrentModel()

    def open_font_dialog_box(self):
        fontSizeItems = list()
        for i in range(4, 9, +1):
            fontSizeItems.append(str(i))

        fontSize, ok = QInputDialog.getItem(
            self, "Font Size", "Select the font Size:", fontSizeItems, 0, False)
        if(ok):
            self.font_size = fontSize
            self.data_display.updateWidget()
            self.update()

    def open_select_plots_dialog_box(self):
        dialog = PlotTypeSelector(self, self.data_display)
        dialog.exec_()

    def hide_callpath_dialog_box(self):
        callpathList = list()
        for callpath in CallPathEnum:
            callpathList.append(callpath.value)
            _, _ = QInputDialog.getItem(
                self, "Callpath Filter", "Select the call path to hide:", callpathList, 0, True)

    def getFontSize(self):
        return self.font_size

    def screenshot(self):
        """
        This function creates a screenshot of this widget
        and stores it into a file. It opens a file dialog to
        specify the file name.
        """
        if pyqt_version == 4:
            pixmap = QPixmap.grabWidget(self)
        else:
            pixmap = self.grab()
            image = pixmap.toImage()

        initial_path = QDir.currentPath() + "/extrap.png"
        filetype = 'png'
        file_filter = filetype.upper() + ' Files (*.' + filetype + ')'
        file_name = QFileDialog.getSaveFileName(self,
                                                self.tr("Save As"),
                                                initial_path,
                                                file_filter)
        file_name = self.get_file_name(file_name)
        if file_name:
            image.save(file_name, filetype)
        self.statusBar().showMessage(self.tr("Ready"))

    def model_experiment(self, experiment):
        # initialize model generator
        model_generator = ModelGenerator(experiment, use_median=self.median)
        # create models from data
        model_generator.model_all()
        self.setExperiment(experiment)
        # self.selector_widget.selectLastModel()
        # self.selector_widget.renameCurrentModel("Default Model")

    def open_text_file(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open a text input file')
        file_name = self.get_file_name(file_name)
        if file_name:
            experiment = read_text_file(file_name)
            # call the modeler and create a function model
            self.model_experiment(experiment)

    def open_json_file(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open a json input file')
        file_name = self.get_file_name(file_name)
        if file_name:
            experiment = read_json_file(file_name)
            # call the modeler and create a function model
            self.model_experiment(experiment)

    def open_talpas_file(self):
        file_name = QFileDialog.getOpenFileName(
            self, 'Open a talpas input file')
        file_name = self.get_file_name(file_name)
        if file_name:
            experiment = read_talpas_file(file_name)
            # call the modeler and create a function model
            self.model_experiment(experiment)

    def open_cube_file(self):
        dir_name = QFileDialog.getExistingDirectory(
            self, 'Select a directory with a set of CUBE files', "", QFileDialog.ReadOnly)
        if not dir_name:
            return
        dialog = CubeFileReader(self, dir_name)
        dialog.exec_()
        if dialog.valid:
            self.setExperiment(dialog.experiment)
            self.selector_widget.selectLastModel()
            self.modeler_widget.remodel()
            self.selector_widget.renameCurrentModel("Default Model")

    def updateMinMaxValue(self):
        if not self.experiment_change:
            updated_value_list = self.selector_widget.getMinMaxValue()
            updated_max_value = max(updated_value_list)
            # don't allow values < 0
            updated_min_value = max(0.0, min(updated_value_list))
            self.min_value = updated_min_value
            self.max_value = updated_max_value
            self.min_value_label.setText(formatNumber(str(updated_min_value)))
            self.max_value_label.setText(formatNumber(str(updated_max_value)))

    def populateCallPathColorMap(self, callpaths):
        callpaths = list(set(callpaths))
        current_index = 0
        size_of_color_list = len(self.graph_color_list)
        self.dict_callpath_color = {}
        for callpath in callpaths:
            if(current_index < size_of_color_list):
                self.dict_callpath_color[callpath] = self.graph_color_list[current_index]
            else:
                offset = (current_index-size_of_color_list) % size_of_color_list
                multiple = int(current_index / size_of_color_list)
                color = self.graph_color_list[offset]
                newcolor = color[:-1]+str(multiple)
                self.dict_callpath_color[callpath] = newcolor
            current_index = current_index + 1

    def getColorForCallPath(self, callpath):
        return self.dict_callpath_color[callpath]

    def get_callpath_color_map(self):
        return self.dict_callpath_color
