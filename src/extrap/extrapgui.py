import argparse
import logging
import sys
import threading
import traceback
import warnings

from PySide2.QtCore import Qt
from PySide2.QtGui import QPalette, QColor
from PySide2.QtWidgets import QApplication, QMessageBox
from matplotlib import font_manager

import __info__
from fileio.cube_file_reader2 import read_cube_file
from fileio.experiment_io import read_experiment
from fileio.extrap3_experiment_reader import read_extrap3_experiment
from fileio.json_file_reader import read_json_file
from fileio.talpas_file_reader import read_talpas_file
from fileio.text_file_reader import read_text_file
from gui.MainWidget import MainWidget
from util.exceptions import RecoverableError


def _preload_common_fonts():
    common_fonts = [
        font_manager.FontProperties('sans\\-serif:style=normal:variant=normal:weight=normal:stretch=normal:size=10.0'),
        'STIXGeneral', 'STIXGeneral:italic', 'STIXGeneral:weight=bold',
        'STIXNonUnicode', 'STIXNonUnicode:italic', 'STIXNonUnicode:weight=bold',
        'STIXSizeOneSym', 'STIXSizeTwoSym', 'STIXSizeThreeSym', 'STIXSizeFourSym', 'STIXSizeFiveSym',
        'cmsy10', 'cmr10', 'cmtt10', 'cmmi10', 'cmb10', 'cmss10', 'cmex10',
        'DejaVu Sans', 'DejaVu Sans:italic', 'DejaVu Sans:weight=bold', 'DejaVu Sans Mono', 'DejaVu Sans Display',
        font_manager.FontProperties('sans\\-serif:style=normal:variant=normal:weight=normal:stretch=normal:size=12.0'),
        font_manager.FontProperties('sans\\-serif:style=normal:variant=normal:weight=normal:stretch=normal:size=6.0')
    ]

    def _thread(fonts):
        for f in fonts:
            font_manager.findfont(f)

    thread = threading.Thread(target=_thread, args=(common_fonts,))
    thread.start()
    return thread


def main(*, test=False):
    # preload fonts for matplotlib
    font_preloader = _preload_common_fonts()
    arguments = parse_arguments()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=arguments.log_level)

    app = QApplication(sys.argv) if not test else QApplication.instance()
    app.setStyle('Fusion')
    app.setPalette(make_palette())

    window = MainWidget()

    _old_warnings_handler = warnings.showwarning
    _old_exception_handler = sys.excepthook

    _seen_warnings = set()

    def _warnings_handler(message: Warning, category, filename, lineno, file=None, line=None):
        nonlocal _seen_warnings
        message_str = str(message)
        if message_str not in _seen_warnings:
            _warn_box = QMessageBox(window)
            _warn_box.setWindowTitle('Warning')
            _warn_box.setIcon(QMessageBox.Icon.Warning)
            _warn_box.finished.connect(lambda x: _seen_warnings.remove(message_str))
            _warn_box.setText(message_str)
            if not test:
                _warn_box.open()
            _seen_warnings.add(message_str)
        logging.warning(message_str)
        QApplication.processEvents()
        return _old_warnings_handler(message, category, filename, lineno, file, line)

    def _exception_handler(type, value, traceback_):
        msgBox = QMessageBox(window)
        if hasattr(value, 'NAME'):
            msgBox.setWindowTitle(__info__.__title__)
        else:
            msgBox.setWindowTitle('Error')
        msgBox.setIcon(QMessageBox.Icon.Critical)
        msgBox.setText(str(value))
        traceback_lines = traceback.extract_tb(traceback_).format()
        msgBox.setDetailedText(''.join(traceback_lines))
        logging.error(str(value))
        if test:
            return _old_exception_handler(type, value, traceback_)
        if issubclass(type, RecoverableError):
            _old_exception_handler(type, value, traceback_)
            msgBox.open()
        else:
            _old_exception_handler(type, value, traceback_)
            msgBox.exec_()  # ensures waiting
            exit(1)

    warnings.showwarning = _warnings_handler
    sys.excepthook = _exception_handler

    window.show()

    load_from_command(arguments, window)

    if not test:
        app.exec_()
        font_preloader.join()
    else:
        font_preloader.join()
        return window, app


def parse_arguments():
    parser = argparse.ArgumentParser(description=__info__.__description__)
    parser.add_argument("--log", action="store", dest="log_level", type=str.upper, choices=['INFO', 'DEBUG'],
                        default='INFO', help="set program's log level [INFO (default), DEBUG]")
    parser.add_argument("--version", action="version", version=__info__.__title__ + " " + __info__.__version__)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--cube", action="store_true", default=False, dest="cube", help="load data from cube files")
    group.add_argument("--text", action="store_true", default=False, dest="text", help="load data from text files")
    group.add_argument("--talpas", action="store_true", default=False, dest="talpas",
                       help="load data from talpas data format")
    group.add_argument("--json", action="store_true", default=False, dest="json", help="load data from json file")
    group.add_argument("--extra-p-3", action="store_true", default=False, dest="extrap3",
                       help="load data from Extra-P 3 experiment")
    parser.add_argument("path", metavar="FILEPATH", type=str, action="store", nargs='?',
                        help="specify a file path for Extra-P to work with")

    parser.add_argument("--scaling", action="store", dest="scaling_type", default="weak", type=str.lower,
                        choices=["weak", "strong"],
                        help="set weak or strong scaling when loading data from cube files [weak (default), strong]")
    arguments = parser.parse_args()
    return arguments


def load_from_command(arguments, window):
    if arguments.path:
        if arguments.text:
            window.import_file(read_text_file, file_name=arguments.path)
        elif arguments.json:
            window.import_file(read_json_file, file_name=arguments.path)
        elif arguments.talpas:
            window.import_file(read_talpas_file, file_name=arguments.path)
        elif arguments.cube:
            window.import_file(lambda x, y: read_cube_file(x, arguments.scaling_type, y), file_name=arguments.path)
        elif arguments.extrap3:
            window.import_file(read_extrap3_experiment, model=False, file_name=arguments.path)
        else:
            window.import_file(read_experiment, model=False, file_name=arguments.path)


def make_palette():
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
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(80, 80, 80))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(80, 80, 80))
    palette.setColor(QPalette.Disabled, QPalette.Button, QColor(150, 150, 150))
    return palette


if __name__ == "__main__":
    main()
