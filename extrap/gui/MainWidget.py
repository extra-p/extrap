# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import signal
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

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
from extrap.gui.components.model_color_map import ModelColorMap
from extrap.modelers.model_generator import ModelGenerator


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
        self.model_color_map = ModelColorMap()
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
        dock = QDockWidget("Selection", self)
        self.selector_widget = SelectorWidget(self, dock)
        dock.setWidget(self.selector_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        # middle: Graph

        self.data_display = DataDisplayManager(self, self)
        central_widget = self.data_display

        # Right side: Model configurator
        dock = QDockWidget("Modeler", self)
        self.modeler_widget = ModelerWidget(self, dock)
        dock.setWidget(self.modeler_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # bottom widget
        dock = QDockWidget("Color Info", self)
        self.color_widget = ColorWidget()
        self.color_widget.update_min_max(self.min_value, self.max_value)
        dock.setWidget(self.color_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        dock = QDockWidget("Graph Limits", self)
        self.graph_limits_widget = GraphLimitsWidget(self, self.data_display)
        dock.setWidget(self.graph_limits_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock, Qt.Horizontal)

        dock2 = QDockWidget("Log", self)
        self.log_widget = LogWidget(self)
        dock2.setWidget(self.log_widget)
        self.tabifyDockWidget(dock, dock2)
        dock2.hide()
        # Menu creation

        # File menu
        screenshot_action = QAction('S&creenshot', self)
        screenshot_action.setShortcut('Ctrl+I')
        screenshot_action.setStatusTip('Creates a screenshot of the Extra-P GUI')
        screenshot_action.triggered.connect(self.screenshot)

        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)

        file_imports = [
            ('Open set of &CUBE files', 'Open a set of CUBE files for single-parameter models and generate data points '
                                        'for a new experiment from them', self.open_cube_file),
            ('Open &text input', 'Open text input file',
             self._make_import_func('Open a Text Input File', read_text_file,
                                    filter="Text Files (*.txt);;All Files (*)")),
            ('Open &JSON input', 'Open JSON or JSON Lines input file',
             self._make_import_func('Open a JSON or JSON Lines Input File', read_json_file,
                                    filter="JSON (Lines) Files (*.json *.jsonl);;All Files (*)")),
            ('Open Tal&pas input', 'Open Talpas input file',
             self._make_import_func('Open a Talpas Input File', read_talpas_file,
                                    filter="Talpas Files (*.txt);;All Files (*)")),
            ('Open Extra-P &3 experiment', 'Opens legacy experiment file',
             self._make_import_func('Open an Extra-P 3 Experiment', read_extrap3_experiment, model=False))
        ]

        open_experiment_action = QAction('&Open experiment', self)
        open_experiment_action.setStatusTip('Opens experiment file')
        open_experiment_action.setShortcut(QKeySequence.Open)
        open_experiment_action.triggered.connect(self.open_experiment)

        save_experiment_action = QAction('&Save experiment', self)
        save_experiment_action.setStatusTip('Saves experiment file')
        save_experiment_action.setShortcut(QKeySequence.Save)
        save_experiment_action.triggered.connect(self.save_experiment)
        save_experiment_action.setEnabled(False)
        self.save_experiment_action = save_experiment_action

        # View menu
        change_font_action = QAction('Legend &font size', self)
        change_font_action.setStatusTip('Change the legend font size')
        change_font_action.triggered.connect(self.open_font_dialog_box)

        select_view_action = QAction('Select plot &type', self)
        select_view_action.setStatusTip('Select the plots you want to view')
        select_view_action.triggered.connect(self.open_select_plots_dialog_box)

        # Plots menu
        graphs = ['&Line graph', 'Selected models in same &surface plot', 'Selected models in &different surface plots',
                  'Dominating models in a 3D S&catter plot',
                  'Max &z as a single surface plot', 'Dominating models and max z as &heat map',
                  'Selected models in c&ontour plot', 'Selected models in &interpolated contour plots',
                  '&Measurement points']
        graph_actions = [QAction(g, self) for g in graphs]
        for i, g in enumerate(graph_actions):
            slot = (lambda k: lambda: self.data_display.reloadTabs((k,)))(i)
            g.triggered.connect(slot)

        # Model menu
        model_delete_action = QAction('&Delete model', self)
        model_delete_action.setShortcut('Ctrl+D')
        model_delete_action.setStatusTip('Delete the current model')
        model_delete_action.triggered.connect(self.selector_widget.model_delete)

        model_rename_action = QAction('&Rename model', self)
        model_rename_action.setShortcut('Ctrl+R')
        model_rename_action.setStatusTip('Rename the current model')
        model_rename_action.triggered.connect(self.selector_widget.model_rename)

        # Filter menu
        # filter_callpath_action = QAction('Filter Callpaths', self)
        # filter_callpath_action.setShortcut('Ctrl+F')
        # filter_callpath_action.setStatusTip('Select the callpath you want to hide')
        # filter_callpath_action.triggered.connect(self.hide_callpath_dialog_box)

        # Main menu bar
        menubar = self.menuBar()
        menubar.setNativeMenuBar(True)

        file_menu = menubar.addMenu('&File')
        for name, tooltip, command in file_imports:
            action = QAction(name, self)
            action.setStatusTip(tooltip)
            action.triggered.connect(command)
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(open_experiment_action)
        file_menu.addAction(save_experiment_action)
        file_menu.addSeparator()
        file_menu.addAction(screenshot_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu('&View')
        view_menu.addAction(change_font_action)
        view_menu.addAction(select_view_action)
        ui_parts_menu = self.createPopupMenu()
        if ui_parts_menu:
            ui_parts_menu_action = view_menu.addMenu(ui_parts_menu)
            ui_parts_menu_action.setText('Tool &windows')

        plots_menu = menubar.addMenu('&Plots')
        for g in graph_actions:
            plots_menu.addAction(g)

        model_menu = menubar.addMenu('&Model')
        model_menu.addAction(model_delete_action)
        model_menu.addAction(model_rename_action)

        # filter_menu = menubar.addMenu('Filter')
        # filter_menu.addAction(filter_callpath_action)

        # Help menue
        help_menu = menubar.addMenu('&Help')

        doc_action = QAction('&Documentation', self)
        doc_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(extrap.__documentation_link__)))
        help_menu.addAction(doc_action)

        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        # Main window
        self.resize(1200, 800)
        self.setCentralWidget(central_widget)
        self.experiment_change = False
        self.show()

    def setExperiment(self, experiment):
        self.experiment_change = True
        self.experiment = experiment
        self.selector_widget.on_experiment_changed()
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
        if not self.windowFilePath():
            event.accept()
            return
        msg_box = QMessageBox(QMessageBox.Question, 'Quit', "Are you sure to quit?",
                              QMessageBox.No | QMessageBox.Yes, self, Qt.Sheet)
        msg_box.setDefaultButton(QMessageBox.No)

        if msg_box.exec_() == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def getExperiment(self):
        return self.experiment

    def getSelectedMetric(self):
        return self.selector_widget.getSelectedMetric()

    def getSelectedCallpath(self):
        return self.selector_widget.getSelectedCallpath()

    def getCurrentModel(self) -> Optional[ModelGenerator]:
        return self.selector_widget.getCurrentModel()

    def open_font_dialog_box(self):
        fontSizeItems = list()
        for i in range(4, 9, +1):
            fontSizeItems.append(str(i))

        fontSize, ok = QInputDialog.getItem(
            self, "Font Size", "Select the font size:", fontSizeItems, 0, False, Qt.Sheet)
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

    def screenshot(self, _checked=False, target=None, name_addition=""):
        """
        This function creates a screenshot of this or the target widget
        and stores it into a file. It opens a file dialog to
        specify the file name and type.
        """
        if not target:
            target = self
        pixmap = target.grab()
        image = pixmap.toImage()

        def _save(file_name):
            with ProgressWindow(self, "Saving Screenshot"):
                image.save(file_name)

        initial_path = Path(self.windowFilePath()).stem + name_addition
        file_filter = ';;'.join(
            [f"{str(f, 'utf-8').upper()} image (*.{str(f, 'utf-8')})" for f in QImageWriter.supportedImageFormats() if
             str(f, 'utf-8') not in ['icns', 'cur', 'ico']])
        dialog = self._file_dialog(_save, "Save Screenshot", initial_path,
                                   file_filter, accept_mode=QFileDialog.AcceptSave)
        dialog.selectNameFilter("PNG image (*.png)")

    def model_experiment(self, experiment):
        # initialize model generator
        model_generator = ModelGenerator(experiment, use_median=self.median, name="Default Model")
        with ProgressWindow(self, 'Modeling') as pbar:
            # create models from data
            model_generator.model_all(pbar)
        self.setExperiment(experiment)

    def _make_import_func(self, title, reader_func, **kwargs):
        return partial(self.import_file, reader_func, title, **kwargs)

    def import_file(self, reader_func, title='Open File', filter='', model=True, progress_text="Loading File",
                    file_name=None):
        def _import_file(file_name):
            with ProgressWindow(self, progress_text) as pw:
                experiment = reader_func(file_name, pw)
                self._set_opened_file_name(file_name)
                # call the modeler and create a function model
                if model:
                    self.model_experiment(experiment)
                else:
                    self.setExperiment(experiment)

        if file_name:
            _import_file(file_name)
        else:
            self._file_dialog(_import_file, title, filter=filter)

    def _file_dialog(self, on_accept, caption='', directory='', filter='', file_mode=None,
                     accept_mode=QFileDialog.AcceptOpen):
        if file_mode is None:
            file_mode = QFileDialog.ExistingFile if accept_mode == QFileDialog.AcceptOpen else QFileDialog.AnyFile
        f_dialog = QFileDialog(self, caption, directory, filter)
        f_dialog.setAcceptMode(accept_mode)
        f_dialog.setFileMode(file_mode)

        def _on_accept():
            file_list = f_dialog.selectedFiles()
            if file_list:
                if len(file_list) > 1:
                    on_accept(file_list)
                else:
                    on_accept(file_list[0])

        f_dialog.accepted.connect(_on_accept)
        f_dialog.open()
        return f_dialog

    def _set_opened_file_name(self, file_name):
        if file_name:
            self.save_experiment_action.setEnabled(True)
            self.setWindowFilePath(file_name)
            self.setWindowTitle(Path(file_name).name + " – " + extrap.__title__)

        else:
            self.save_experiment_action.setEnabled(False)
            self.setWindowFilePath("")
            self.setWindowTitle(extrap.__title__)

    def open_experiment(self):
        self.import_file(read_experiment, 'Open Experiment',
                         filter='Experiments (*.extra-p)',
                         model=False,
                         progress_text="Loading experiment")

    def save_experiment(self):
        def _save(file_name):
            with ProgressWindow(self, "Saving Experiment") as pw:
                write_experiment(self.getExperiment(), file_name, pw)
                self._set_opened_file_name(file_name)

        self._file_dialog(_save,
                          'Save Experiment', filter='Experiments (*.extra-p)', accept_mode=QFileDialog.AcceptSave)

    def open_cube_file(self):
        def _process_cube(dir_name):
            dialog = CubeFileReader(self, dir_name)
            dialog.setWindowFlag(Qt.Sheet, True)
            dialog.setModal(True)
            dialog.exec_()  # do not use open, wait for loading to finish
            if dialog.valid:
                self._set_opened_file_name(dir_name)
                self.model_experiment(dialog.experiment)

        self._file_dialog(_process_cube,
                          'Select a Directory with a Set of CUBE Files', "", file_mode=QFileDialog.Directory)

    def updateMinMaxValue(self):
        if not self.experiment_change:
            updated_value_list = self.selector_widget.getMinMaxValue()
            # don't allow values < 0
            updated_max_value = max(0.0, max(updated_value_list))
            updated_min_value = max(0.0, min(updated_value_list))
            self.min_value = updated_min_value
            self.max_value = updated_max_value
            self.color_widget.update_min_max(self.min_value, self.max_value)

    def show_about_dialog(self):
        QMessageBox.about(self, "About " + extrap.__title__,
                          f"""<h1>{extrap.__title__}</h1>
<p>Version {extrap.__version__}</p>
<p>{extrap.__description__}</p>
<p>{extrap.__copyright__}</p>
"""
                          )

    activate_event_handlers = []

    def event(self, e: QEvent) -> bool:
        if e.type() == QEvent.WindowActivate:
            for h in self.activate_event_handlers:
                h(e)
        return super().event(e)
