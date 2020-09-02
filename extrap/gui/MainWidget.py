"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the package base
directory for details.
"""
import signal
from enum import Enum
from functools import partial
from pathlib import Path

from PySide2.QtCore import *  # @UnusedWildImport
from PySide2.QtGui import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport

import extrap
from extrap.fileio.experiment_io import read_experiment, write_experiment
from extrap.fileio.extrap3_experiment_reader import read_extrap3_experiment
from extrap.fileio.json_file_reader import read_json_file
from extrap.fileio.talpas_file_reader import read_talpas_file
from extrap.fileio.text_file_reader import read_text_file
from extrap.gui.ColorWidget import ColorWidget
from extrap.gui.CubeFileReader import CubeFileReader
from extrap.gui.DataDisplay import DataDisplayManager, GraphLimitsWidget
from extrap.gui.LogWidget import LogWidget
from extrap.gui.ModelerWidget import ModelerWidget
from extrap.gui.PlotTypeSelector import PlotTypeSelector
from extrap.gui.ProgressWindow import ProgressWindow
from extrap.gui.SelectorWidget import SelectorWidget
from extrap.modelers.model_generator import ModelGenerator

pyqt_version = 5


class CallPathEnum(Enum):
    constant = "constant"
    logarithmic = "logarithmic"
    polynomial = "polynomial"
    exponential = "exponential"


class MainWidget(QMainWindow):

    def __init__(self, *args, **kwargs):
        """
        Initializes the extrap application widget.
        """
        super(MainWidget, self).__init__(*args, **kwargs)
        self.max_value = 1
        self.min_value = 1
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
        self.median = False

    # noinspection PyAttributeOutsideInit
    def initUI(self):
        """
        Initializes the User Interface of the extrap widget. E.g. the menus.
        """
        self.setWindowTitle(extrap.__title__)
        # Status bar
        # self.statusBar()

        # Main splitter
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setDockNestingEnabled(True)

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

        # bottom widget
        dock = QDockWidget(self.tr("Color Info"), self)
        self.color_widget = ColorWidget()
        self.color_widget.update_min_max(self.min_value, self.max_value)
        dock.setWidget(self.color_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        dock = QDockWidget(self.tr("Graph Limits"), self)
        self.graph_limits_widget = GraphLimitsWidget(self, self.data_display)
        dock.setWidget(self.graph_limits_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock, Qt.Horizontal)

        dock2 = QDockWidget(self.tr("Log"), self)
        self.log_widget = LogWidget(self)
        dock2.setWidget(self.log_widget)
        self.tabifyDockWidget(dock, dock2)
        dock2.hide()
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

        file_imports = [
            ('Open set of CUBE files', 'Open a set of CUBE files for single parameter models and generate data points '
                                       'for a new experiment from them', self.open_cube_file),
            ('Open text input', 'Open text input file',
             self._make_import_func('Open a text input file', read_text_file, filter="Text (*.txt);;All Files (*)")),
            ('Open json input', 'Open json input file',
             self._make_import_func('Open a json input file', read_json_file,
                                    filter="JSON (*.json), JSON Lines (*.jsonl);;All Files (*)")),
            ('Open talpas input', 'Open talpas input file',
             self._make_import_func('Open a talpas input file', read_talpas_file,
                                    filter="Talpas (*.txt);;All Files (*)")),
            ('Open Extra-P 3 experiment', 'Opens legacy experiment file',
             self._make_import_func('Open an Extra-P 3 experiment file', read_extrap3_experiment, model=False))
        ]

        open_experiment_action = QAction(self.tr('Open experiment'), self)
        open_experiment_action.setStatusTip(self.tr('Opens experiment file'))
        open_experiment_action.triggered.connect(self.open_experiment)

        save_experiment_action = QAction(self.tr('Save experiment'), self)
        save_experiment_action.setStatusTip(self.tr('Saves experiment file'))
        save_experiment_action.triggered.connect(self.save_experiment)

        # View menu
        change_font_action = QAction(self.tr('Legend font size'), self)
        change_font_action.setStatusTip(self.tr('Change the legend font size'))
        change_font_action.triggered.connect(self.open_font_dialog_box)

        select_view_action = QAction(self.tr('Select Plot Type'), self)
        select_view_action.setStatusTip(self.tr('Select the Plots you want to view'))
        select_view_action.triggered.connect(self.open_select_plots_dialog_box)

        # Plots menu
        graphs = ['Line graph', 'Selected models in same surface plot', 'Selected models in different surface plots',
                  'Dominating models in a 3D Scatter plot',
                  'Max z as a single surface plot', 'Dominating models and max z as heat map',
                  'Selected models in contour plot', 'Selected models in interpolated contour plots',
                  'Measurement points']
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
        # filter_callpath_action = QAction(self.tr('Filter Callpaths'), self)
        # filter_callpath_action.setShortcut('Ctrl+F')
        # filter_callpath_action.setStatusTip(self.tr('Select the callpath you want to hide'))
        # filter_callpath_action.triggered.connect(self.hide_callpath_dialog_box)

        # Main menu bar
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        file_menu = menubar.addMenu(self.tr('&File'))
        for name, tooltip, command in file_imports:
            action = QAction(self.tr(name), self)
            action.setStatusTip(self.tr(tooltip))
            action.triggered.connect(command)
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(open_experiment_action)
        file_menu.addAction(save_experiment_action)
        file_menu.addSeparator()
        file_menu.addAction(screenshot_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu(self.tr('View'))
        view_menu.addAction(change_font_action)
        view_menu.addAction(select_view_action)
        ui_parts_menu = self.createPopupMenu()
        if ui_parts_menu:
            ui_parts_menu_action = view_menu.addMenu(ui_parts_menu)
            ui_parts_menu_action.setText('Tool Windows')

        plots_menu = menubar.addMenu(self.tr('Plots'))
        for g in graph_actions:
            plots_menu.addAction(g)

        model_menu = menubar.addMenu(self.tr('Model'))
        model_menu.addAction(model_delete_action)
        model_menu.addAction(model_rename_action)

        # filter_menu = menubar.addMenu(self.tr('Filter'))
        # filter_menu.addAction(filter_callpath_action)

        # Main window
        self.resize(1200, 800)
        self.setCentralWidget(central_widget)
        self.experiment_change = False
        self.show()

    @staticmethod
    def get_file_name(fileName):
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

        self.selector_widget.tree_model.valuesChanged()
        self.data_display.experimentChange()
        self.modeler_widget.experimentChanged()
        self.experiment_change = False
        self.updateMinMaxValue()
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
        if ok:
            self.font_size = fontSize
            self.data_display.updateWidget()
            self.update()

    def open_select_plots_dialog_box(self):
        dialog = PlotTypeSelector(self, self.data_display)
        dialog.setModal(True)
        dialog.open()

    # def hide_callpath_dialog_box(self):
    #     callpathList = list()
    #     for callpath in CallPathEnum:
    #         callpathList.append(callpath.value)
    #     answer,ok = QInputDialog.getItem(
    #         self, "Callpath Filter", "Select the call path to hide:", callpathList, 0, True)

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
        with ProgressWindow(self, 'Modeling') as pbar:
            # create models from data
            model_generator.model_all(pbar)
        self.setExperiment(experiment)

    def _make_import_func(self, title, reader_func, **kwargs):
        return partial(self.import_file, reader_func, title, **kwargs)

    def import_file(self, reader_func, title='Open file', filter='', model=True, progress_text="Loading file",
                    file_name=None):
        if not file_name:
            file_name = QFileDialog.getOpenFileName(self, title, filter=filter)
            file_name = self.get_file_name(file_name)
        if file_name:
            with ProgressWindow(self, progress_text) as pw:
                experiment = reader_func(file_name, pw)
                self._set_opened_file_name(file_name)
            # call the modeler and create a function model
            if model:
                self.model_experiment(experiment)
            else:
                self.setExperiment(experiment)

    def _set_opened_file_name(self, file_name):
        if file_name:
            self.setWindowTitle(extrap.__title__ + " - " + Path(file_name).name)
        else:
            self.setWindowTitle(extrap.__title__)

    def open_experiment(self):
        self.import_file(read_experiment, 'Open experiment',
                         filter='Experiment (*.extra-p)',
                         model=False,
                         progress_text="Loading experiment")

    def save_experiment(self):
        file_name = QFileDialog.getSaveFileName(
            self, 'Save experiment', filter='Experiment (*.extra-p)')
        file_name = self.get_file_name(file_name)
        if file_name:
            with ProgressWindow(self, "Saving experiment") as pw:
                write_experiment(self.getExperiment(), file_name, pw)

    def open_cube_file(self):
        dir_name = QFileDialog.getExistingDirectory(
            self, 'Select a directory with a set of CUBE files', "", QFileDialog.ReadOnly)
        if not dir_name:
            return
        dialog = CubeFileReader(self, dir_name)
        dialog.setModal(True)
        dialog.open()
        if dialog.valid:
            self._set_opened_file_name(dir_name)
            self.model_experiment(dialog.experiment)

    def updateMinMaxValue(self):
        if not self.experiment_change:
            updated_value_list = self.selector_widget.getMinMaxValue()
            # don't allow values < 0
            updated_max_value = max(0.0, max(updated_value_list))
            updated_min_value = max(0.0, min(updated_value_list))
            self.min_value = updated_min_value
            self.max_value = updated_max_value
            self.color_widget.update_min_max(self.min_value, self.max_value)

    def populateCallPathColorMap(self, callpaths):
        callpaths = list(set(callpaths))
        current_index = 0
        size_of_color_list = len(self.graph_color_list)
        self.dict_callpath_color = {}
        for callpath in callpaths:
            if current_index < size_of_color_list:
                self.dict_callpath_color[callpath] = self.graph_color_list[current_index]
            else:
                offset = (current_index - size_of_color_list) % size_of_color_list
                multiple = int(current_index / size_of_color_list)
                color = self.graph_color_list[offset]
                newcolor = color[:-1] + str(multiple)
                self.dict_callpath_color[callpath] = newcolor
            current_index = current_index + 1

    def getColorForCallPath(self, callpath):
        return self.dict_callpath_color[callpath]

    def get_callpath_color_map(self):
        return self.dict_callpath_color
