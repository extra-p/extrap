# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import json
import math
import warnings
from asyncio import Event
from functools import partial
from itertools import chain
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, Type, cast

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QCommandLinkButton, QFileDialog, QFormLayout,
                               QLabel, QLineEdit, QSizePolicy, QSpacerItem,
                               QWizard, QWizardPage, QComboBox, QGridLayout, QButtonGroup, QHBoxLayout, QWidget,
                               QRadioButton, QListWidget, QGroupBox, QDoubleSpinBox, QPushButton, QSpinBox,
                               QListWidgetItem, QAbstractItemView, QVBoxLayout)

from extrap.comparison import matchers
from extrap.comparison.entities.projection_info import RooflineData
from extrap.comparison.experiment_comparison import ComparisonExperiment
from extrap.comparison.matchers import AbstractMatcher
from extrap.entities.parameter import Parameter
from extrap.fileio.experiment_io import ExperimentReader
from extrap.fileio.file_reader import FileReader, all_readers
from extrap.gui.comparison.interactive_matcher import InteractiveMatcher
from extrap.gui.components import file_dialog
from extrap.gui.components.dynamic_options import DynamicOptionsWidget
from extrap.gui.components.wizard_pages import ProgressPage, ScrollAreaPage
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.dynamic_options import DynamicOptions
from extrap.util.exceptions import FileFormatError
from extrap.util.unique_list import UniqueList


class ComparisonWizard(QWizard):
    file_reader: Type[FileReader]
    file_name: str

    def __init__(self, experiment1, experiment2=None, name1='exp1', name2='exp2'):
        super().__init__()

        self.setWindowTitle("Compare With Other Experiment")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        if not experiment2:
            self.addPage(FileSelectionPage(self))
            self.addPage(FileReaderOptionsPage(self))
            self.file_loading_page_id = self.addPage(FileLoadingPage(self))
        self.addPage(NamingPage(self))
        self.addPage(ModelSelectionPage(self))
        self.addPage(ParameterMappingPage(self))
        self.addPage(MatcherSelectionPage(self))
        self.comparing_page_id = self.addPage(ComparingPage(self))
        self.matcher: Optional[AbstractMatcher] = None
        self.exp_names = [name1, name2]
        self.experiment1 = experiment1
        self.experiment2 = experiment2
        self.experiment: Optional[ComparisonExperiment] = None
        self.model_mapping = {}
        self.parameter_mapping = {}
        self.is_cancelled = Event()
        self.file_reader = None
        self.rejected.connect(self.on_reject)

    def on_reject(self):
        self.is_cancelled.set()

    def back(self) -> None:
        self.restart()


class FileSelectionPage(ScrollAreaPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Select input format')
        layout = self.scroll_layout
        for reader in chain([ExperimentReader], all_readers.values()):
            def _(reader):
                btn = QCommandLinkButton(reader.GUI_ACTION.replace('&', ''))
                # btn.setDescription(reader.DESCRIPTION)
                btn.clicked.connect(
                    lambda: file_dialog.show(self, partial(self.open_file, reader), reader.DESCRIPTION,
                                             filter=reader.FILTER,
                                             file_mode=QFileDialog.FileMode.Directory if reader.LOADS_FROM_DIRECTORY else None))
                layout.addWidget(btn)

            _(reader)
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        self.scroll_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))

    def open_file(self, reader, name):
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        wizard.file_name = name
        wizard.exp_names[1] = Path(name).stem
        wizard.file_reader = reader()
        wizard.next()

    def nextId(self) -> int:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        if not isinstance(wizard.file_reader, DynamicOptions):
            return wizard.file_loading_page_id
        else:
            return super().nextId()

    def isComplete(self) -> bool:
        return False


class FileReaderOptionsPage(QWizardPage):
    def __init__(self, parent: ComparisonWizard):
        super().__init__(parent)
        self.setTitle('File import options')
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.dynamic_options_widget = DynamicOptionsWidget(self, None)
        layout.addWidget(self.dynamic_options_widget)

    def initializePage(self) -> None:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        self.dynamic_options_widget.update_object_with_options(wizard.file_reader)


class MatcherSelectionPage(ScrollAreaPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Select mapping provider')
        layout = self.scroll_layout
        for matcher in chain(matchers.all_matchers.values(), [InteractiveMatcher]):
            def _(matcher):
                btn = QCommandLinkButton(matcher.NAME)
                btn.setDescription(matcher.DESCRIPTION)
                btn.clicked.connect(lambda: self.select_matcher(matcher))
                layout.addWidget(btn)

            _(matcher)
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))

    def select_matcher(self, matcher):
        self.wizard().matcher = matcher
        self.wizard().next()

    def isComplete(self) -> bool:
        return False


class FileLoadingPage(ProgressPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Loading experiment')

    def do_process(self, pbar):
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        wizard.experiment2 = wizard.file_reader.read_experiment(wizard.file_name, pbar)
        if wizard.file_reader.GENERATE_MODELS_AFTER_LOAD:
            from extrap.gui.MainWidget import DEFAULT_MODEL_NAME
            ModelGenerator(wizard.experiment2, name=DEFAULT_MODEL_NAME).model_all(pbar)


class ComparingPage(ProgressPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Comparing experiments')

    def cleanupPage(self) -> None:
        self.next_id = None
        super().cleanupPage()

    def do_process(self, pbar):
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        if wizard.matcher == InteractiveMatcher:
            matcher = wizard.matcher(wizard)
        else:
            matcher = wizard.matcher()

        wizard.experiment = ComparisonExperiment(wizard.experiment1, wizard.experiment2, matcher=matcher)
        wizard.experiment.experiment_names = wizard.exp_names
        wizard.experiment.modelers_match = wizard.model_mapping
        wizard.experiment.parameter_mapping = wizard.parameter_mapping
        if wizard.matcher == InteractiveMatcher:
            self.next_id = matcher.determine_next_page_id()
        else:
            wizard.experiment.do_comparison(pbar)

    def once_after_shown(self):
        super().once_after_shown()
        btn = QCommandLinkButton("Project estimate based on hardware capabilities")
        btn.clicked.connect(self.switch_to_projection)
        self.layout().addWidget(btn)

    def switch_to_projection(self):
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        self.next_id = wizard.addPage(ProjectionConfigurationPage(wizard))
        wizard.addPage(ProjectionProgressPage(wizard))
        wizard.next()


class NamingPage(QWizardPage):
    def __init__(self, parent: ComparisonWizard):
        super().__init__(parent)
        self.setTitle('Name compared experiments')
        layout = QFormLayout(self)
        self.setLayout(layout)
        self._tb_name1 = QLineEdit()
        self._tb_name2 = QLineEdit()
        layout.addRow(QLabel("These names are used in Extra-P to show the sources of the compared models."))
        layout.addRow("Name of experiment 1", self._tb_name1)
        layout.addRow("Name of experiment 2", self._tb_name2)

    def initializePage(self) -> None:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        self._tb_name1.setText(wizard.exp_names[0])
        self._tb_name2.setText(wizard.exp_names[1])

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        wizard.exp_names[0] = self._tb_name1.text()
        if not wizard.exp_names[0]:
            wizard.exp_names[0] = 'exp1'
        wizard.exp_names[1] = self._tb_name2.text()
        if not wizard.exp_names[1]:
            wizard.exp_names[1] = 'exp2'
        if wizard.exp_names[0] == wizard.exp_names[1]:
            wizard.exp_names[0] += '1'
            wizard.exp_names[1] += '2'
        return super().validatePage()


class ModelSelectionPage(QWizardPage):
    def __init__(self, parent: ComparisonWizard):
        super().__init__(parent)
        self.setTitle('Choose Models for Comparison')
        self.layout = QFormLayout(self)
        self.setLayout(self.layout)
        self.model_lists = []

    def initializePage(self) -> None:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        for modeler in wizard.experiment1.modelers:
            if len(self.model_lists) < len(wizard.experiment1.modelers):
                model_list = QComboBox()
                model_list.addItems([model.name for model in wizard.experiment2.modelers])
                default_value = model_list.findText(modeler.name)
                if default_value >= 0:
                    model_list.setCurrentIndex(default_value)
                self.model_lists.append(model_list)
                self.layout.addRow(f"Compare {modeler.name} with: ", model_list)

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        exp1_models = wizard.experiment1.modelers
        exp2_models = wizard.experiment2.modelers
        wizard.model_mapping = {
            m.name: [m, exp2_models[self.model_lists[i].currentIndex()]] for i, m in enumerate(exp1_models)
        }

        return super().validatePage()


class ParameterMappingPage(QWizardPage):
    def __init__(self, parent: ComparisonWizard):
        super().__init__(parent)
        self.setTitle('Apply parameter mapping')
        self._layout = QGridLayout(self)
        self._param_lists = []
        self._name_edits = []
        self.setLayout(self._layout)

    def initializePage(self) -> None:
        self._clear_layout()
        self._name_edits.clear()
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        r_ctr = 0
        self._layout.addWidget(QLabel('Parameter Experiment 1'), r_ctr, 0)
        self._layout.addWidget(QLabel('Parameter Experiment 2'), r_ctr, 1)
        self._layout.addWidget(QLabel('New Parameter Name'), r_ctr, 2)
        r_ctr = 3
        for param in wizard.experiment1.parameters:
            self._layout.addWidget(QLabel(param.name + ':'), r_ctr, 0)
            param_list = QComboBox()
            param_list.addItems([param2.name for param2 in wizard.experiment2.parameters])
            default_value = param_list.findText(param.name)
            if default_value >= 0:
                param_list.setCurrentIndex(default_value)
            self._param_lists.append(param_list)
            self._layout.addWidget(param_list, r_ctr, 1)

            name_edit = QLineEdit(param.name)
            self._name_edits.append(name_edit)
            self._layout.addWidget(name_edit, r_ctr, 2)
            r_ctr += 1

    def _clear_layout(self):
        for i in reversed(range(self._layout.count())):
            widget = self._layout.itemAt(i).widget()
            self._layout.removeWidget(widget)
            widget.setParent(None)

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        old_param_names = [p.name for p in wizard.experiment1.parameters]
        new_param_names = [name_edit.text().strip() for name_edit in self._name_edits]

        for i, param_name in enumerate(new_param_names):
            if not param_name:
                warnings.warn("Parameters cannot be empty.")
                return False
            if old_param_names[i] != param_name \
                    and Parameter(param_name) in wizard.experiment1.parameters:
                warnings.warn(f"Parameter {param_name} already exists, "
                              "you cannot have two parameters with the same name.")
                return False
            if new_param_names.count(param_name) > 1:
                warnings.warn(f"Parameter {param_name} already exists, "
                              "you cannot have two parameters with the same name.")
                return False

        wizard.parameter_mapping = {
            name: [str(param1), cb_param2.currentText()] for name, param1, cb_param2 in
            zip(new_param_names, wizard.experiment1.parameters, self._param_lists)
        }

        return super().validatePage()


class ProjectionConfigurationPage(QWizardPage):
    def __init__(self, parent: ComparisonWizard):
        super().__init__(parent)
        self.setTitle('Projection')
        self._layout = QFormLayout(self)
        self.setLayout(self._layout)
        self._init_UI()

    def _init_UI(self):
        experiment_selection_buttons = QWidget(self)
        experiment_selection_buttons.setLayout(QHBoxLayout(self))
        experiment_selection_buttons.layout().setContentsMargins(0, 0, 0, 0)

        self._target_experiment_selection = QButtonGroup(self)
        experiment_selection = self._target_experiment_selection
        experiment_selection.idClicked.connect(self.update_arithmetic_intensity_metrics)
        self._target_exp_rb_1 = QRadioButton("Exp1", experiment_selection_buttons)
        experiment_selection_buttons.layout().addWidget(self._target_exp_rb_1)
        experiment_selection.addButton(self._target_exp_rb_1, 0)

        self._target_exp_rb_2 = QRadioButton("Exp2", experiment_selection_buttons)
        experiment_selection_buttons.layout().addWidget(self._target_exp_rb_2)
        experiment_selection.addButton(self._target_exp_rb_2, 1)

        experiment_selection_buttons.layout().addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self._target_exp_rb_2.setChecked(True)

        self._layout.addRow("Target experiment", experiment_selection_buttons)

        self._metrics_to_project_box = QListWidget(self)
        self._metrics_to_project_box.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._layout.addRow("Metrics to project", self._metrics_to_project_box)

        roofline_group = QGroupBox(self)
        self._layout.addRow(roofline_group)

        roofline_layout = QGridLayout(roofline_group)
        roofline_group.setLayout(roofline_layout)
        roofline_group.setTitle("Roofline model information")

        roofline_layout.addWidget(QLabel("Memory bandwidth"), 0, 1)
        roofline_layout.addWidget(QLabel("Peak performance"), 0, 2)

        self._exp_label_1 = QLabel("Exp1", roofline_group)
        roofline_layout.addWidget(self._exp_label_1, 1, 0)
        self._bw_sb_1 = QDoubleSpinBox(roofline_group)
        self._bw_sb_1.setSuffix(" GBytes/s")
        self._bw_sb_1.setMaximum(math.inf)
        roofline_layout.addWidget(self._bw_sb_1, 1, 1)
        self._pp_sb_1 = QDoubleSpinBox(roofline_group)
        self._pp_sb_1.setSuffix(" GFlops/s")
        self._pp_sb_1.setMaximum(math.inf)
        roofline_layout.addWidget(self._pp_sb_1, 1, 2)
        self._rm_load_btn_1 = QPushButton("Load ERT JSON file...", roofline_group)
        roofline_layout.addWidget(self._rm_load_btn_1, 1, 3)
        self._rm_load_btn_1.clicked.connect(lambda: self._load_ert(self._bw_sb_1, self._pp_sb_1))

        self._exp_label_2 = QLabel("Exp2", roofline_group)
        roofline_layout.addWidget(self._exp_label_2, 2, 0)
        self._bw_sb_2 = QDoubleSpinBox(roofline_group)
        self._bw_sb_2.setSuffix(" GBytes/s")
        self._bw_sb_2.setMaximum(math.inf)
        roofline_layout.addWidget(self._bw_sb_2, 2, 1)
        self._pp_sb_2 = QDoubleSpinBox(roofline_group)
        self._pp_sb_2.setSuffix(" GFlops/s")
        self._pp_sb_2.setMaximum(math.inf)
        roofline_layout.addWidget(self._pp_sb_2, 2, 2)
        self._rm_load_btn_2 = QPushButton("Load ERT JSON file...", roofline_group)
        roofline_layout.addWidget(self._rm_load_btn_2, 2, 3)
        self._rm_load_btn_2.clicked.connect(lambda: self._load_ert(self._bw_sb_2, self._pp_sb_2))

        self._arithmetic_intensity_group = QGroupBox(self)
        self._layout.addRow(self._arithmetic_intensity_group)
        self._arithmetic_intensity_group.setTitle("Use arithmetic intensity (operation intensity) to improve accuracy")
        self._arithmetic_intensity_group.setCheckable(True)
        arithmetic_intensity_layout = QFormLayout(self._arithmetic_intensity_group)
        self._arithmetic_intensity_group.setLayout(arithmetic_intensity_layout)

        self._fp_dp_metric_cb = QComboBox(self._arithmetic_intensity_group)
        arithmetic_intensity_layout.addRow("Floating point operations (double precision)", self._fp_dp_metric_cb)
        # self._fp_sp_metric_cb = QComboBox(self._arithmetic_intensity_group)
        # arithmetic_intensity_layout.addRow("Floating point operations (single precision)", self._fp_sp_metric_cb)
        self._num_mem_transfers_metric_cb = QComboBox(self._arithmetic_intensity_group)
        arithmetic_intensity_layout.addRow("Number of memory transfers", self._num_mem_transfers_metric_cb)
        self._bytes_per_mem_transfer_sb = QSpinBox(self._arithmetic_intensity_group)
        arithmetic_intensity_layout.addRow("Bytes per memory transfer", self._bytes_per_mem_transfer_sb)

    def initializePage(self) -> None:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())

        self._target_exp_rb_1.setText(wizard.exp_names[0])
        self._target_exp_rb_2.setText(wizard.exp_names[1])
        self._exp_label_1.setText(wizard.exp_names[0])
        self._exp_label_2.setText(wizard.exp_names[1])

        self._target_exp_rb_2.setChecked(True)
        self.update_arithmetic_intensity_metrics(1)

        self._metrics_to_project_box.clear()

        for m in wizard.experiment.metrics:
            item = QListWidgetItem(m.name)
            item.setData(Qt.UserRole, m)
            self._metrics_to_project_box.addItem(item)
            if m.name == "time":
                self._metrics_to_project_box.setCurrentItem(item)

    @Slot(int)
    def update_arithmetic_intensity_metrics(self, target_id):
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        self._fp_dp_metric_cb.clear()
        # self._fp_sp_metric_cb.clear()
        self._num_mem_transfers_metric_cb.clear()
        self._bytes_per_mem_transfer_sb.setValue(8)

        # self._fp_sp_metric_cb.addItem("Not selected", None)
        self._fp_dp_metric_cb.addItem("Not selected", None)
        self._num_mem_transfers_metric_cb.addItem("Not selected", None)

        metrics = UniqueList()
        for id, experiment in enumerate(wizard.experiment.compared_experiments):
            if id == target_id:
                continue
            metrics.extend(experiment.metrics)

        found_ai_flops, found_ai_mem = False, False
        for m in metrics:
            # self._fp_sp_metric_cb.addItem(m.name, m)
            self._fp_dp_metric_cb.addItem(m.name, m)
            self._num_mem_transfers_metric_cb.addItem(m.name, m)
            if m.name == "PAPI_DP_OPS":
                found_ai_flops = True
                self._fp_dp_metric_cb.setCurrentText(m.name)
            # elif m.name == "PAPI_SP_OPS":
            #     found_ai_flops = True
            #     self._fp_sp_metric_cb.setCurrentText(m.name)
            elif m.name == "UNC_M_CAS_COUNT:ALL":
                found_ai_mem = True
                self._num_mem_transfers_metric_cb.setCurrentText(m.name)

        self._arithmetic_intensity_group.setChecked(found_ai_flops and found_ai_mem)

    def validatePage(self) -> bool:
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        if len(self._metrics_to_project_box.selectedIndexes()) < 1:
            warnings.warn("At least one metric to project must be selected.")
            return False
        if self._pp_sb_1.value() <= 0 or self._pp_sb_2.value() <= 0 or \
                self._bw_sb_1.value() <= 0 or self._bw_sb_2.value() <= 0:
            warnings.warn("You have to provide the peak performance and peak memory bandwidth for all experiments.")
            return False

        if self._arithmetic_intensity_group.isChecked():
            if self._bytes_per_mem_transfer_sb.value() < 1:
                warnings.warn("Bytes per memory transfer must be at least one.")
                return False
            # self._fp_sp_metric_cb.currentIndex() <= 0 and
            if self._fp_dp_metric_cb.currentIndex() <= 0:
                warnings.warn(
                    "At least one floating point metric must be selected to calculate the arithmetic intensity.")
                return False

        proj_info = wizard.experiment.projection_info
        proj_info.peak_performance_in_gflops_per_s[0] = self._pp_sb_1.value()
        proj_info.peak_mem_bandwidth_in_gbytes_per_s[0] = self._bw_sb_1.value()
        proj_info.target_experiment_id = self._target_experiment_selection.checkedId()

        proj_info.peak_performance_in_gflops_per_s[1] = self._pp_sb_2.value()
        proj_info.peak_mem_bandwidth_in_gbytes_per_s[1] = self._bw_sb_2.value()

        proj_info.metrics_to_project = [i.data(Qt.UserRole) for i in self._metrics_to_project_box.selectedItems()]

        if self._arithmetic_intensity_group.isChecked():
            proj_info.fp_dp_metric = self._fp_dp_metric_cb.currentData()
            # proj_info.fp_sp_metric = self._fp_sp_metric_cb.currentData()
            proj_info.num_mem_transfers_metric = self._num_mem_transfers_metric_cb.currentData()
            proj_info.bytes_per_mem = self._bytes_per_mem_transfer_sb.value()
        else:
            proj_info.fp_dp_metric = None
            proj_info.fp_sp_metric = None
            proj_info.num_mem_transfers_metric = None
            proj_info.bytes_per_mem = 0
        return super().validatePage()

    def _load_ert(self, _bw_sb_2, _pp_sb_2):
        def _load_and_parse(filename):
            try:
                with open(filename, 'r') as file:
                    ert_data: RooflineData = json.load(file)['empirical']
                    _bw_sb_2.setValue(ert_data['gbytes']["data"][-1][1])
                    _pp_sb_2.setValue(ert_data['gflops']["data"][0][1])
            except JSONDecodeError as err:
                raise FileFormatError(str(err)) from err
            except KeyError as err:
                raise FileFormatError(str(err)) from err
            except IndexError as err:
                raise FileFormatError(str(err)) from err

        file_dialog.show(self, _load_and_parse, "Open ERT JSON file", filter="ERT JSON Files (*.json);;All Files (*)",
                         file_mode=QFileDialog.FileMode.ExistingFile)


class ProjectionProgressPage(ProgressPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Projecting estimate')

    def do_process(self, pbar):
        wizard: ComparisonWizard = cast(ComparisonWizard, self.wizard())
        # wizard.projection_info
        experiment = wizard.experiment
        projection_info = experiment.projection_info
        experiment.project_expected_performance(projection_info, pbar)
