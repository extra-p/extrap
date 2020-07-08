# import logging

import sys
import warnings

from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtGui import QPalette, QColor
from PySide2.QtCore import Qt
from gui.MainWidget import MainWidget
from fileio.text_file_reader import read_text_file
from fileio.json_file_reader import read_json_file


def main():
    # TODO: add logging to the gui application

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    # app.setStyleSheet('QWidget{background:#333;color:#eee}')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(190, 190, 190))
    palette.setColor(QPalette.WindowText, Qt.black)
    palette.setColor(QPalette.Base, QColor(220, 220, 220))
    palette.setColor(QPalette.AlternateBase, QColor(10, 10, 10))
    palette.setColor(QPalette.Text, Qt.black)
    palette.setColor(QPalette.Button, QColor(220, 220, 220))
    palette.setColor(QPalette.ButtonText, Qt.black)
    palette.setColor(QPalette.Highlight, QColor(31, 119, 180))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(40, 40, 40))
    palette.setColor(QPalette.Disabled, QPalette.Button, QColor(150, 150, 150))
    app.setPalette(palette)

    window = MainWidget()

    _old_warnings_handler = warnings.showwarning

    def _warnings_handler(message: Warning, category, filename, lineno, file=None, line=None):
        msgBox = QMessageBox(window)
        msgBox.setWindowTitle('Warning')
        msgBox.setIcon(QMessageBox.Icon.Warning)
        msgBox.setText(str(message))
        msgBox.open()
        return _old_warnings_handler(message, category, filename, lineno, file, line)

    warnings.showwarning = _warnings_handler

    window.show()

    if len(sys.argv) == 3 and '--text' == sys.argv[1]:
        experiment = read_text_file(sys.argv[2])
        # call the modeler and create a function model
        window.model_experiment(experiment)
    elif len(sys.argv) == 3 and '--json' == sys.argv[1]:
        experiment = read_json_file(sys.argv[2])
        # call the modeler and create a function model
        window.model_experiment(experiment)

    app.exec_()


if __name__ == "__main__":
    main()
