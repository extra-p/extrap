# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import sys
import unittest
import warnings
from threading import Thread

from PySide6.QtCore import QRect, QItemSelectionModel
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

from extrap.extrap import extrapgui
from extrap.fileio.file_reader.text_file_reader import TextFileReader
from extrap.gui.AdvancedPlotWidget import AdvancedPlotWidget
from extrap.gui.MainWidget import MainWidget, QCoreApplication

try:
    APP = QApplication()
    APP.setStyle('Fusion')
    app_thread = Thread(target=APP.exec)
except:
    app_thread = None
    pass


class TestGuiCommon(unittest.TestCase):

    def setUp(self) -> None:
        global app_thread
        if not app_thread:
            raise unittest.SkipTest("GUI could not start.")
        if not app_thread.is_alive():
            app_thread = Thread(target=APP.exec)
            app_thread.start()
        self.window = MainWidget()
        self.window.hide()

    def tearDown(self):
        if not app_thread:
            raise unittest.SkipTest("GUI could not start.")
        self.window.closeEvent = lambda e: e.accept()
        self.window.close()

    def test_line_graph(self):
        data_display = self.window.data_display
        self.assertTrue(data_display.is_tab_already_opened("Line graph"))
        data_display.display_widget.tabBar().removeTab(0)
        self.assertFalse(data_display.is_tab_already_opened("Line graph"))
        data_display.reloadTabs([0])
        self.assertTrue(data_display.is_tab_already_opened("Line graph"))


class TestGuiExperimentLoaded(TestGuiCommon):

    def setUp(self) -> None:
        super().setUp()

        exp = TextFileReader().read_experiment('data/text/one_parameter_6.txt')
        self.window.model_experiment(exp)

    def test_graph_model_multiple_selected(self):
        data_display = self.window.data_display
        self.window.selector_widget.tree_view.selectAll()
        # check graphs
        tabs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        data_display.reloadTabs(tabs)
        for i in tabs:
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
        tabs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        data_display.reloadTabs(tabs)
        for i in tabs:
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
        tabs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        data_display.reloadTabs(tabs)
        for i in tabs:
            data_display.display_widget.setCurrentIndex(i)
            p = data_display.display_widget.currentWidget()
            if isinstance(p, AdvancedPlotWidget):
                self.assertIsNotNone(p.graphDisplayWindow)
                p.drawGraph()
                self.assertIsNotNone(p.graphDisplayWindow)
                QCoreApplication.processEvents()

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


class TestGuiLoadExperiment(unittest.TestCase):
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


class TestGuiNoExperiment(TestGuiCommon):
    def test_generator_button(self):
        self.assertFalse(self.window.modeler_widget._model_button.isEnabled())
        self.assertTrue(self.window.modeler_widget.model_name_edit)

    def test_graph_no_model_selected(self):
        data_display = self.window.data_display
        # check graphs
        tabs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        data_display.reloadTabs(tabs)
        for i in tabs:
            data_display.display_widget.setCurrentIndex(i)
            p = data_display.display_widget.currentWidget()
            if isinstance(p, AdvancedPlotWidget):
                self.assertIsNone(p.graphDisplayWindow)
                QCoreApplication.processEvents()


class TestGuiSelectedThenNoModelSelected(TestGuiExperimentLoaded):
    def setUp(self) -> None:
        super().setUp()
        self.test_graph_model_one_selected()
        self.window.selector_widget.get_current_model_gen = lambda: None


if __name__ == '__main__':
    unittest.main()
