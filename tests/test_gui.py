# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import sys
import unittest
import warnings

from PySide6.QtCore import QRect, QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

from extrap.extrap import extrapgui
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.gui.AdvancedPlotWidget import AdvancedPlotWidget
from extrap.gui.MainWidget import MainWidget, QCoreApplication
from extrap.modelers.model_generator import ModelGenerator
from extrap.modelers.single_parameter.segmented import SegmentedModeler

_qapp_instance = None


class GuiTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        global _qapp_instance
        if _qapp_instance is None:
            _qapp_instance = QApplication([])
            _qapp_instance.setStyle('Fusion')

        self.app_instance = _qapp_instance
        self.tabs = list(range(12))

    def tearDown(self):
        del self.app_instance
        super(GuiTestCase, self).tearDown()


class TestGuiCommon(GuiTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.window = MainWidget()
        # self.window.hide()

    def tearDown(self):
        # if not app_thread:
        #     raise unittest.SkipTest("GUI could not start.")
        QCoreApplication.processEvents()
        self.window.closeEvent = lambda e: e.accept()
        self.window.close()
        QCoreApplication.processEvents()

    def test_line_graph(self):
        data_display = self.window.data_display
        self.assertTrue(data_display.is_tab_already_opened("Line graph"))
        data_display.display_widget.tabBar().removeTab(0)
        self.assertFalse(data_display.is_tab_already_opened("Line graph"))
        data_display.reloadTabs([0])
        self.assertTrue(data_display.is_tab_already_opened("Line graph"))

    def test_start_modeling(self):
        modeler_widget = self.window.modeler_widget
        modeler_widget.model_mean_radio.click()
        modeler_widget._model_button.click()
        modeler_widget.remodel()
        modeler_widget._model_other_radio.click()
        modeler_widget._model_button.click()
        modeler_widget.remodel()


class TestGuiExperimentLoaded(TestGuiCommon):

    def setUp(self) -> None:
        super().setUp()

        exp = TextFileReader().read_experiment('data/text/one_parameter_6.txt')
        self.window.model_experiment(exp)

    def test_graph_model_multiple_selected(self):
        data_display = self.window.data_display
        self.window.selector_widget.tree_view.selectAll()
        # check graphs
        data_display.reloadTabs(self.tabs)
        for i in self.tabs:
            data_display.display_widget.setCurrentIndex(i)
            p = data_display.display_widget.currentWidget()
            if isinstance(p, AdvancedPlotWidget):
                self.assertIsNotNone(p.graphDisplayWindow)
                p.drawGraph()
                self.assertIsNotNone(p.graphDisplayWindow)
                QCoreApplication.processEvents()

    def test_graph_model_one_selected(self):
        data_display = self.window.data_display
        self.window.selector_widget.tree_view.setSelection(QRect(0, 0, 1, 1), QItemSelectionModel.SelectionFlag.Rows)
        # check graphs
        data_display.reloadTabs(self.tabs)
        for i in self.tabs:
            data_display.display_widget.setCurrentIndex(i)
            p = data_display.display_widget.currentWidget()
            if isinstance(p, AdvancedPlotWidget):
                self.assertIsNotNone(p.graphDisplayWindow)
                p.drawGraph()
                self.assertIsNotNone(p.graphDisplayWindow)
                QCoreApplication.processEvents()

    def test_graph_no_model_selected(self):
        data_display = self.window.data_display
        # check graphs
        data_display.reloadTabs(self.tabs)
        for i in self.tabs:
            data_display.display_widget.setCurrentIndex(i)
            p = data_display.display_widget.currentWidget()
            if isinstance(p, AdvancedPlotWidget):
                self.assertIsNotNone(p.graphDisplayWindow)
                p.drawGraph()
                self.assertIsNotNone(p.graphDisplayWindow)
                QCoreApplication.processEvents()

    def test_model_display(self):
        tree_view = self.window.selector_widget.tree_view

        rows = tree_view.model().rowCount()
        columns = list(range(tree_view.model().columnCount()))
        columns.remove(1)  # Severity column
        columns.remove(2)  # Annotations column

        for row in range(rows):
            for col in columns:
                index = tree_view.model().index(row, col)
                data = tree_view.model().data(index, Qt.DisplayRole)
                self.assertIsNotNone(data, f"Missing content in ({row}, {col})")
                self.assertNotEqual(str(data).strip(), "", f"Empty string in ({row}, {col})")
            # Check severity column
            index = tree_view.model().index(row, 1)
            data = tree_view.model().data(index, Qt.ItemDataRole.DecorationRole)
            self.assertIsNotNone(data, f"Missing content in ({row}, {1})")
            self.assertNotEqual(str(data).strip(), "", f"Empty string in ({row}, {1})")

    def test_modeler_options_reset(self):
        modeler_widget = self.window.modeler_widget
        modeler_widget._options_container.toggle(False)
        QCoreApplication.processEvents()
        checkbox = None
        reset_button = None
        for child in modeler_widget._options_container.content().children():
            QCoreApplication.processEvents()
            if not reset_button and isinstance(child, QPushButton):
                reset_button = child
            elif not checkbox and isinstance(child, QCheckBox):
                checkbox = child
            elif reset_button and checkbox:
                break
        if checkbox:
            old_state = checkbox.isChecked()
            checkbox.toggle()
            QCoreApplication.processEvents()
            self.assertNotEqual(old_state, checkbox.isChecked())
            reset_button.click()
            for child in modeler_widget._options_container.content().children():
                QCoreApplication.processEvents()
                if isinstance(child, QCheckBox):
                    checkbox = child
                    break

            self.assertEqual(old_state, checkbox.isChecked())

    def test_start_modeling(self):
        super().test_start_modeling()


class TestGuiLoadExperiment(GuiTestCase):
    def test_load_experiment(self):
        _old_warnings_handler = warnings.showwarning
        _old_exception_handler = sys.excepthook
        try:
            window, app = extrapgui.main(test=True, args=[])
            exp = TextFileReader().read_experiment('data/text/one_parameter_1.txt')
            self.assertIsNone(window.getExperiment())

            window.model_experiment(exp)
            QCoreApplication.processEvents()
            self.assertIsNotNone(window.getExperiment())
            window.closeEvent = lambda e: e.accept()
            window.close()
        finally:
            warnings.showwarning = _old_warnings_handler
            sys.excepthook = _old_exception_handler

    def test_load_experiment_keep_values(self):
        _old_warnings_handler = warnings.showwarning
        _old_exception_handler = sys.excepthook
        try:
            window, app = extrapgui.main(test=True, args=[])
            tfr = TextFileReader()
            tfr.keep_values = True
            exp = tfr.read_experiment('data/text/one_parameter_1.txt')
            self.assertIsNone(window.getExperiment())

            window.model_experiment(exp)
            QCoreApplication.processEvents()
            self.assertIsNotNone(window.getExperiment())
            window.closeEvent = lambda e: e.accept()
            window.close()
        finally:
            warnings.showwarning = _old_warnings_handler
            sys.excepthook = _old_exception_handler


class TestGuiNoExperiment(TestGuiCommon):
    def test_generator_button(self):
        self.assertFalse(self.window.modeler_widget._model_button.isEnabled())
        self.assertTrue(self.window.modeler_widget.model_name_edit)

    def test_graph_no_model_selected(self):
        data_display = self.window.data_display
        # check graphs
        data_display.reloadTabs(self.tabs)
        for i in self.tabs:
            data_display.display_widget.setCurrentIndex(i)
            p = data_display.display_widget.currentWidget()
            if isinstance(p, AdvancedPlotWidget):
                self.assertIsNone(p.graphDisplayWindow)
                QCoreApplication.processEvents()


class TestGuiSegmented(TestGuiExperimentLoaded):
    def setUp(self) -> None:
        super().setUp()

        exp = TextFileReader().read_experiment('data/text/one_parameter_segmented_6.txt')
        model_generator = ModelGenerator(exp, modeler=SegmentedModeler())
        model_generator.model_all()
        self.window.set_experiment(exp)

    def test_start_modeling(self):
        modeler_widget = self.window.modeler_widget
        modeler_widget.model_mean_radio.click()
        num_modelers = modeler_widget._model_selector.count()
        for i in range(num_modelers):
            if modeler_widget._model_selector.itemText(i).lower() == 'segmented':
                modeler_widget._model_selector.setCurrentIndex(i)
        modeler_widget._model_button.click()
        modeler_widget.remodel()
        modeler_widget._model_other_radio.click()
        for i in range(num_modelers):
            if modeler_widget._model_selector.itemText(i).lower() == 'segmented':
                modeler_widget._model_selector.setCurrentIndex(i)
        modeler_widget._model_button.click()
        modeler_widget.remodel()


class TestGuiSelectedThenNoModelSelected(TestGuiExperimentLoaded):
    def setUp(self) -> None:
        super().setUp()
        self.test_graph_model_one_selected()
        self.window.selector_widget.get_current_model_gen = lambda: None


if __name__ == '__main__':
    unittest.main()
