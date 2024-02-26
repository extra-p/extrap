# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools
import logging
import signal
import sys
from enum import Enum
from functools import partial
from numbers import Number
from pathlib import Path
from typing import Optional, Sequence, Tuple, Type
from urllib.error import URLError, HTTPError

from PySide6.QtCore import *  # @UnusedWildImport
from PySide6.QtGui import *  # @UnusedWildImport
from PySide6.QtWidgets import *  # @UnusedWildImport

import extrap
from extrap.comparison.experiment_comparison import ComparisonExperiment
from extrap.entities.calltree import Node
from extrap.entities.experiment import Experiment
from extrap.entities.model import Model
from extrap.entities.scaling_type import ScalingType
from extrap.fileio import io_helper
from extrap.fileio.experiment_io import read_experiment, write_experiment
from extrap.fileio.file_reader import all_readers, FileReader
from extrap.fileio.file_reader.abstract_directory_reader import AbstractScalingConversionReader
from extrap.fileio.file_reader.cube_file_reader2 import CubeFileReader2
from extrap.gui.ColorWidget import ColorWidget
from extrap.gui.CoordinateTransformation import CoordinateTransformationDialog
from extrap.gui.DataDisplay import DataDisplayManager, GraphLimitsWidget
from extrap.gui.ImportOptionsDialog import ImportOptionsDialog
from extrap.gui.LogWidget import LogWidget
from extrap.gui.ModelerWidget import ModelerWidget
from extrap.gui.PlotTypeSelector import PlotTypeSelector
from extrap.gui.PostProcessingWidget import PostProcessingWidget
from extrap.gui.RankingWidget import RankingWidget
from extrap.gui.SelectorWidget import SelectorWidget
from extrap.gui.StrongScalingConversionDialog import StrongScalingConversionDialog
from extrap.gui.comparison.comparison_wizard import ComparisonWizard
from extrap.gui.components import file_dialog
from extrap.gui.components.ProgressWindow import ProgressWindow
from extrap.gui.components.model_color_map import ModelColorMap
from extrap.gui.components.plot_formatting_options import PlotFormattingOptions, PlotFormattingDialog
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.deprecation import deprecated
from extrap.util.dynamic_options import DynamicOptions
from extrap.util.event import Event

_SETTING_CHECK_FOR_UPDATES_ON_STARTUP = 'check_for_updates_on_startup'

DEFAULT_MODEL_NAME = "Default Model"


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

        self.settings = QSettings(QSettings.Scope.UserScope, "Extra-P", "Extra-P GUI")

        self.min_value = 0
        self.developer_mode = False
        self._experiment = None
        self.model_color_map = ModelColorMap()
        self.plot_formatting_options = PlotFormattingOptions()
        self.experiment_change = True
        self.min_max_value_updated_event = Event(Number, int)
        self._init_ui()

        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # switch for using mean or median measurement values for modeling
        # is used when loading the data from a file and then modeling directly
        self.median = False

        if sys.platform.startswith('darwin'):
            self._macos_update_title_bar()

    # noinspection PyAttributeOutsideInit
    def _init_ui(self):
        """
        Initializes the User Interface of the extrap widget. E.g. the menus.
        """

        self.setWindowTitle(extrap.__title__)
        # Status bar
        # self.statusBar()

        # Main splitter
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setDockNestingEnabled(True)

        # Left side: Callpath and metric selection
        dock = QDockWidget("Selection", self)
        self.selector_widget = SelectorWidget(self, dock)
        dock.setWidget(self.selector_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # middle: Graph

        self.data_display = DataDisplayManager(self, self)
        central_widget = self.data_display

        # Right side: Model configurator
        dock = QDockWidget("Modeler", self)
        self.modeler_widget = ModelerWidget(self, dock)
        dock.setWidget(self.modeler_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        dock = QDockWidget("Aggregation and Analysis", self)
        self.postprocessing_widget = PostProcessingWidget(self, dock)
        dock.setWidget(self.postprocessing_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        # bottom widget
        dock = QDockWidget("Color Info", self)
        self.color_widget = ColorWidget()
        self.min_max_value_updated_event += self.color_widget.update_min_max
        dock.setWidget(self.color_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        dock = QDockWidget("Graph Limits", self)
        self.graph_limits_widget = GraphLimitsWidget(self, self.data_display)
        dock.setWidget(self.graph_limits_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock, Qt.Orientation.Horizontal)

        dock2 = QDockWidget("Log", self)
        self.log_widget = LogWidget(self)
        dock2.setWidget(self.log_widget)
        self.tabifyDockWidget(dock, dock2)
        dock2.hide()

        dock2 = QDockWidget("Ranking", self)
        self.ranking_widget = RankingWidget(self)
        dock2.setWidget(self.ranking_widget)
        self.tabifyDockWidget(dock, dock2)
        dock2.hide()
        # Menu creation

        # File menu
        screenshot_action = QAction('S&creenshot', self)
        screenshot_action.setShortcut('Ctrl+I')
        screenshot_action.setStatusTip('Creates a screenshot of the Extra-P GUI')
        screenshot_action.triggered.connect(self.screenshot)

        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)

        file_imports = []
        for reader in all_readers.values():
            file_imports.append((reader.GUI_ACTION, reader.DESCRIPTION, self._make_import_func(reader)))

        open_experiment_action = QAction('&Open experiment', self)
        open_experiment_action.setStatusTip('Opens experiment file')
        open_experiment_action.setShortcut(QKeySequence.StandardKey.Open)
        open_experiment_action.triggered.connect(self.open_experiment)

        save_experiment_action = QAction('&Save experiment', self)
        save_experiment_action.setStatusTip('Saves experiment file')
        save_experiment_action.setShortcut(QKeySequence.StandardKey.Save)
        save_experiment_action.triggered.connect(self.save_experiment)
        save_experiment_action.setEnabled(False)
        self.save_experiment_action = save_experiment_action

        # View menu
        change_font_action = QAction('Plot &formatting options', self)
        change_font_action.setStatusTip('Change the formatting of the plots')
        change_font_action.triggered.connect(self.open_plot_format_dialog_box)

        toggle_developer_mode = QAction('Developer mode', self)
        toggle_developer_mode.setCheckable(True)
        toggle_developer_mode.toggled.connect(self._toggle_developer_mode)

        select_view_action = QAction('Select plot &type', self)
        select_view_action.setStatusTip('Select the plots you want to view')
        select_view_action.triggered.connect(self.open_select_plots_dialog_box)

        # Plots menu
        graphs = ['&Line graph', 'Selected models in same &surface plot', 'Selected models in &different surface plots',
                  'Dominating models in a 3D S&catter plot',
                  'Max &z as a single surface plot', 'Dominating models and max z as &heat map',
                  'Selected models in c&ontour plot', 'Selected models in &interpolated contour plots',
                  '&Measurement points', 'Stacked &area plot', '&Comparison plot', '&Expectation plot 3D']
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

        metric_delete_action = QAction('Dele&te metrics', self)
        metric_delete_action.triggered.connect(self.selector_widget.delete_metric)

        coordinates_transform_dialog = CoordinateTransformationDialog(self)
        coordinates_transform = QAction('Transform Coordinates', self)
        coordinates_transform.triggered.connect(coordinates_transform_dialog.show)
        # compare menu
        compare_action = QAction('&Compare with experiment', self)
        compare_action.setStatusTip('Compare the current models with ')
        compare_action.triggered.connect(self._compare_experiment)
        compare_action.setEnabled(False)
        self.compare_action = compare_action

        # Filter menu
        # filter_callpath_action = QAction('Filter Callpaths', self)
        # filter_callpath_action.setShortcut('Ctrl+F')
        # filter_callpath_action.setStatusTip('Select the callpath you want to hide')
        # filter_callpath_action.triggered.connect(self.hide_callpath_dialog_box)

        # Main menu bar
        menubar = self.menuBar()
        menubar.setContentsMargins(-4, 0, 0, 0)
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
        file_menu.addAction(compare_action)
        file_menu.addSeparator()
        file_menu.addAction(screenshot_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu('&View')
        view_menu.addAction(change_font_action)
        view_menu.addAction(select_view_action)
        view_menu.addAction(toggle_developer_mode)
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
        model_menu.addSeparator()
        model_menu.addAction(metric_delete_action)
        model_menu.addSeparator()
        model_menu.addAction(coordinates_transform)

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

        if self.settings.value(_SETTING_CHECK_FOR_UPDATES_ON_STARTUP, True, bool):
            update_available = None
            try:
                update_available = self.update_available()
            except Exception as e:
                logging.error("Check for updates: " + str(e))

            if update_available:
                update_menu = menubar.addMenu("UPDATE AVAILABLE")
                update_action = QAction(
                    f'Version {update_available[0]} is available here: {update_available[1]}',
                    self)
                update_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(update_available[1])))
                update_menu.addAction(update_action)

                ignore_action = QAction("Ignore", self)
                ignore_action.triggered.connect(lambda: update_menu.menuAction().setVisible(False))
                update_menu.addAction(ignore_action)
                update_menu.addSeparator()
                auto_update_toggle = QAction(f'Check for updates on startup', self)
                auto_update_toggle.setChecked(True)
                auto_update_toggle.setCheckable(True)
                update_menu.addAction(auto_update_toggle)
                auto_update_toggle.toggled.connect(
                    lambda toggled: self.settings.setValue(_SETTING_CHECK_FOR_UPDATES_ON_STARTUP,
                                                           toggled))

        # Main window
        self.resize(1200, 800)
        self.setCentralWidget(central_widget)
        self.experiment_change = False
        self.show()

    def set_experiment(self, experiment, file_name="", *, compared=False):
        if experiment is None:
            raise ValueError("Experiment cannot be none.")
        self.experiment_change = True
        self._experiment = experiment
        if file_name is not None:
            self._set_opened_file_name(file_name, compared=compared)
        self.save_experiment_action.setEnabled(True)
        self.compare_action.setEnabled(not isinstance(experiment, ComparisonExperiment))
        self.selector_widget.on_experiment_changed()
        self.data_display.experimentChange()
        self.modeler_widget.experimentChanged()
        self.postprocessing_widget.on_experiment_changed(experiment)
        self.experiment_change = False
        self.updateMinMaxValue()
        self.update()

    def on_selection_changed(self):
        if not self.experiment_change:
            self.data_display.updateWidget()
            self.update()
            self.updateMinMaxValue()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if not self.windowFilePath():
            event.accept()
            return
        msg_box = QMessageBox(QMessageBox.Icon.Question, 'Quit', "Are you sure to quit?",
                              QMessageBox.StandardButton.No | QMessageBox.StandardButton.Yes, self, Qt.WindowType.Sheet)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        if msg_box.exec() == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

    def getExperiment(self) -> Experiment:
        return self._experiment

    def get_selected_metric(self):
        return self.selector_widget.getSelectedMetric()

    def get_selected_call_tree_nodes(self) -> Sequence[Node]:
        return self.selector_widget.get_selected_call_tree_nodes()

    def get_current_model_gen(self) -> Optional[ModelGenerator]:
        return self.selector_widget.getCurrentModel()

    def get_selected_models(self) -> Tuple[Optional[Sequence[Model]], Optional[Sequence[Node]]]:
        return self.selector_widget.get_selected_models()

    def open_plot_format_dialog_box(self):
        dialog = PlotFormattingDialog(self.plot_formatting_options, self, Qt.WindowType.Sheet, self.model_color_map)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.data_display.updateWidget()
            self.update()

    def open_select_plots_dialog_box(self):
        dialog = PlotTypeSelector(self, self.data_display)
        dialog.setModal(True)
        dialog.open()

    @Slot()
    def _compare_experiment(self, experiment=None):
        exp_name = 'exp1'
        if self.windowFilePath():
            exp_name = Path(self.windowFilePath()).stem
        cw = ComparisonWizard(self.getExperiment(), experiment, exp_name)

        def on_accept():
            self.set_experiment(cw.experiment,
                                cw.experiment.experiment_names[0] + " <> " + cw.experiment.experiment_names[1],
                                compared=True)

        cw.accepted.connect(on_accept)
        cw.open()

    # def hide_callpath_dialog_box(self):
    #     callpathList = list()
    #     for callpath in CallPathEnum:
    #         callpathList.append(callpath.value)
    #     answer,ok = QInputDialog.getItem(
    #         self, "Callpath Filter", "Select the call path to hide:", callpathList, 0, True)

    @deprecated
    def getFontSize(self):
        return self.plot_formatting_options.font_size

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
        dialog = file_dialog.showSave(self, _save, "Save Screenshot", initial_path, file_filter)
        dialog.selectNameFilter("PNG image (*.png)")

    def model_experiment(self, experiment, file_name=""):
        # initialize model generator
        model_generator = ModelGenerator(experiment, use_median=self.median, name=DEFAULT_MODEL_NAME)
        with ProgressWindow(self, 'Modeling') as pbar:
            # create models from data
            model_generator.model_all(pbar)
        self.set_experiment(experiment, file_name)

    def _make_import_func(self, reader_class: Type[FileReader]):
        file_mode = QFileDialog.FileMode.Directory if reader_class.LOADS_FROM_DIRECTORY else None
        title = reader_class.DESCRIPTION if reader_class.DESCRIPTION else 'Open File'

        if issubclass(reader_class, DynamicOptions):
            def _import_function():
                def _process_with_settings(path):
                    reader: FileReader = reader_class()
                    dialog = ImportOptionsDialog(self, reader, path)
                    dialog.setWindowFlag(Qt.WindowType.Sheet, True)
                    dialog.setModal(True)
                    # do not use open, wait for loading to finish
                    if dialog.exec() == QDialog.DialogCode.Accepted:
                        if reader_class.GENERATE_MODELS_AFTER_LOAD:
                            experiment = self._check_and_convert_scaling(dialog.experiment, path, reader)
                            self.model_experiment(experiment, path)
                        else:
                            self.set_experiment(dialog.experiment, path)

                file_dialog.show(self, _process_with_settings, title, filter=reader_class.FILTER, file_mode=file_mode)

            return _import_function

        else:
            reader: FileReader = reader_class()
            return partial(self.import_file, reader.read_experiment, title, filter=reader_class.FILTER,
                           model=reader_class.GENERATE_MODELS_AFTER_LOAD, file_mode=file_mode, reader=reader)

    def _check_and_convert_scaling(self, experiment, path, reader=None):
        check_res = io_helper.check_for_strong_scaling(experiment)
        if not check_res or max(check_res) == 0:
            return experiment
        if isinstance(reader, AbstractScalingConversionReader):
            if StrongScalingConversionDialog.pose_question_for_readers_with_scaling_conversion(
                    self) == QMessageBox.StandardButton.Yes:
                reader.scaling_type = ScalingType.STRONG
                dialog = ImportOptionsDialog(self, reader, path)
                dialog.setWindowFlag(Qt.WindowType.Sheet, True)
                dialog.setModal(True)
                dialog.open()
                dialog.accept()

                return dialog.experiment
        else:
            dialog = StrongScalingConversionDialog(experiment, check_res, self)
            dialog.setWindowFlag(Qt.WindowType.Sheet, True)
            dialog.setModal(True)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                return dialog.experiment
        return experiment

    def import_file(self, reader_func, title='Open File', filter='', model=True, progress_text="Loading File",
                    file_name=None, file_mode=None, reader=None):
        def _import_file(file_name):
            with ProgressWindow(self, progress_text) as pw:
                experiment = reader_func(file_name, pw)
                # call the modeler and create a function model
                if model:
                    experiment = self._check_and_convert_scaling(experiment, file_name, reader)
                    self.model_experiment(experiment, file_name)
                else:
                    self.set_experiment(experiment, file_name)

        if file_name:
            _import_file(file_name)
        else:
            file_dialog.show(self, _import_file, title, filter=filter, file_mode=file_mode)

    def _set_opened_file_name(self, file_name, *, compared=False):
        if file_name:
            self.setWindowFilePath(file_name if not compared else "")
            self.setWindowTitle(Path(file_name).name + " – " + extrap.__title__)
        else:
            self.setWindowFilePath("")
            self.setWindowTitle(extrap.__title__)

    def open_experiment(self, file_name=None):
        self.import_file(read_experiment, 'Open Experiment',
                         filter='Experiments (*.extra-p)',
                         model=False,
                         progress_text="Loading experiment",
                         file_name=file_name)

    def save_experiment(self):
        def _save(file_name):
            with ProgressWindow(self, "Saving Experiment") as pw:
                write_experiment(self.getExperiment(), file_name, pw)
                self._set_opened_file_name(file_name)

        file_dialog.showSave(self, _save, 'Save Experiment', filter='Experiments (*.extra-p)')

    @deprecated
    def open_cube_file(self):
        self._make_import_func(CubeFileReader2)()

    def updateMinMaxValue(self):
        if not self.experiment_change:
            self.min_max_value_updated_event(*self.selector_widget.update_min_max_value())

    def show_about_dialog(self):
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About " + extrap.__title__)
        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        layout.setSpacing(4)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 1)

        row = itertools.count(0)

        columnSpan = 4
        layout.addWidget(QLabel(f"<h1>{extrap.__title__}</h1>"), next(row), 0, 1, columnSpan)
        layout.addWidget(QLabel(f"<b>{extrap.__description__}</b>"), next(row), 0, 1, columnSpan)
        layout.addItem(QSpacerItem(0, 2), next(row), 0, 1, columnSpan)

        icon_label = QLabel()
        text_label = QLabel()
        text_label.setOpenExternalLinks(True)
        same_row = next(row)
        layout.addWidget(icon_label, same_row, 0, 1, 1)
        layout.addWidget(text_label, same_row, 1, 1, columnSpan - 1)
        try:
            update_available = self.update_available()
            if not update_available:
                icon_label.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton).pixmap(16))
                text_label.setText(f"{extrap.__title__} is up to date")
            else:
                icon_label.setPixmap(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation).pixmap(16))
                text_label.setText(f'Version {update_available[0]} is available. '
                                   f'Get it here: <a href="{update_available[1]}">{update_available[1]}</a>')
        except HTTPError as e:
            icon_label.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(16))
            text_label.setText(f"Could not check for updates: " + str(e))
        except URLError as e:
            icon_label.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(16))
            text_label.setText(f"Could not check for updates: " + str(e.reason))
        except Exception as e:
            icon_label.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(16))
            text_label.setText(f"Could not check for updates: " + str(e))

        same_row = next(row)
        layout.addWidget(QLabel(f"Version {extrap.__version__}"), same_row, 1, 1, 1)
        layout.addWidget(QLabel(' — '), same_row, 2, 1, 1)

        check_for_updates = QCheckBox(self)
        check_for_updates.setChecked(bool(self.settings.value(_SETTING_CHECK_FOR_UPDATES_ON_STARTUP, True, bool)))
        check_for_updates.setText("Check for updates on start up")
        check_for_updates.toggled.connect(
            lambda status: self.settings.setValue(_SETTING_CHECK_FOR_UPDATES_ON_STARTUP, status))
        layout.addWidget(check_for_updates, same_row, 3, 1, 1)

        layout.addItem(QSpacerItem(0, 2), next(row), 0, 1, columnSpan)

        creators = QLabel(extrap.__developed_by_html__)
        creators.setOpenExternalLinks(True)
        layout.addWidget(creators, next(row), 0, 1, columnSpan)
        layout.addItem(QSpacerItem(0, 2), next(row), 0, 1, columnSpan)
        support = QLabel(f'Do you have questions or suggestions?<br>'
                         f'Write us: <a href="mailto:{extrap.__support_email__}">{extrap.__support_email__}</a>')
        support.setOpenExternalLinks(True)
        layout.addWidget(support, next(row), 0, 1, columnSpan)

        layout.addItem(QSpacerItem(0, 10), next(row), 0, 1, columnSpan)

        layout.addWidget(QLabel(extrap.__copyright__), next(row), 0, 1, columnSpan)

        layout.addItem(QSpacerItem(0, 10), next(row), 0, 1, columnSpan)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(about_dialog.accept)

        # button_box.addButton(check_for_updates, QDialogButtonBox.ButtonRole.ResetRole)
        # ResetRole is a hack to achieve left alignment
        layout.addWidget(button_box, next(row), 0, 1, columnSpan)
        about_dialog.setLayout(layout)

        about_dialog.open()

    activate_event_handlers = []

    def event(self, e: QEvent) -> bool:
        if e.type() == QEvent.Type.WindowActivate:
            for h in self.activate_event_handlers:
                h(e)
        elif e.type() == QEvent.Type.LayoutRequest or e.type() == QEvent.Type.WinIdChange:
            if sys.platform.startswith('darwin'):
                self._macos_update_title_bar()
        return super().event(e)

    def _macos_update_title_bar(self):
        try:
            import objc
            from AppKit import NSWindow, NSView, NSColor, NSColorSpace
            ns_view = objc.objc_object(c_void_p=int(self.winId()))
            ns_window = ns_view.window()
            if ns_window is not None:
                ns_window.setTitlebarAppearsTransparent_(True)
                ns_window.setColorSpace_(NSColorSpace.sRGBColorSpace())
                c = self.palette().window().color()
                ns_window_color = NSColor.colorWithDeviceRed_green_blue_alpha_(c.redF(), c.greenF(), c.blueF(),
                                                                               c.alphaF())
                ns_window.setBackgroundColor_(ns_window_color)
        except ImportError:
            pass

    @staticmethod
    def update_available():
        import json
        import urllib.request
        from packaging.version import Version

        with urllib.request.urlopen(extrap.__current_version_api__) as response:
            data = json.loads(response.read().decode('utf-8'))
            info = data['info']

            if Version(info['version']) > Version(extrap.__version__):
                return info['version'], info['release_url']
            else:
                return False

    @Slot(bool)
    def _toggle_developer_mode(self, enabled):
        self.developer_mode = enabled
