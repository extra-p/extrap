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

import extrap
from extrap.fileio.cube_file_reader2 import read_cube_file
from extrap.fileio.experiment_io import read_experiment
from extrap.fileio.extrap3_experiment_reader import read_extrap3_experiment
from extrap.fileio.json_file_reader import read_json_file
from extrap.fileio.talpas_file_reader import read_talpas_file
from extrap.fileio.text_file_reader import read_text_file
from extrap.gui.MainWidget import MainWidget
from extrap.util.exceptions import RecoverableError, CancelProcessError

TRACEBACK = logging.DEBUG - 1
logging.addLevelName(TRACEBACK, 'TRACEBACK')


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


def main(*, args=None, test=False):
    # preload fonts for matplotlib
    font_preloader = _preload_common_fonts()
    arguments = parse_arguments(args)

    # configure logging
    log_level = min(logging.getLevelName(arguments.log_level), logging.INFO)
    if arguments.log_file:
        logging.basicConfig(format="%(levelname)s: %(asctime)s: %(message)s", level=log_level,
                            filename=arguments.log_file)
    else:
        logging.basicConfig(format="%(levelname)s: %(asctime)s: %(message)s", level=log_level)
    logging.getLogger().handlers[0].setLevel(logging.getLevelName(arguments.log_level))

    app = QApplication(sys.argv) if not test else QApplication.instance()
    app.setStyle('Fusion')
    app.setPalette(make_palette())

    window = MainWidget()

    _current_warnings = set()
    _old_warnings_handler = warnings.showwarning
    _old_exception_handler = sys.excepthook

    def _warnings_handler(message: Warning, category, filename, lineno, file=None, line=None):
        nonlocal _current_warnings
        message_str = str(message)
        if message_str not in _current_warnings:
            _warn_box = QMessageBox(window)
            _warn_box.setWindowTitle('Warning')
            _warn_box.setIcon(QMessageBox.Icon.Warning)
            _warn_box.finished.connect(lambda x: _current_warnings.remove(message_str))
            _warn_box.setText(message_str)
            if not test:
                _warn_box.open()
            _current_warnings.add(message_str)
        logging.warning(message_str)
        logging.log(TRACEBACK, ''.join(traceback.format_stack()))
        QApplication.processEvents()
        return _old_warnings_handler(message, category, filename, lineno, file, line)

    def _exception_handler(type, value, traceback_):
        traceback_text = ''.join(traceback.extract_tb(traceback_).format())

        if issubclass(type, CancelProcessError):
            logging.log(TRACEBACK, str(value))
            logging.log(TRACEBACK, traceback_text)
            return

        msgBox = QMessageBox(window)
        print()
        if hasattr(value, 'NAME'):
            msgBox.setWindowTitle(getattr(value, 'NAME'))
        else:
            msgBox.setWindowTitle('Error')
        msgBox.setIcon(QMessageBox.Icon.Critical)
        msgBox.setText(str(value))
        msgBox.setDetailedText(traceback_text)

        logging.error(str(value))
        logging.log(TRACEBACK, traceback_text)

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

    try:
        load_from_command(arguments, window)
    except CancelProcessError:
        pass

    if not test:
        app.exec_()
        font_preloader.join()
    else:
        font_preloader.join()
        return window, app


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description=extrap.__description__)
    parser.add_argument("--log", action="store", dest="log_level", type=str.upper, default='CRITICAL',
                        choices=['TRACEBACK', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="set program's log level (default: CRITICAL)")
    parser.add_argument("--logfile", action="store", dest="log_file",
                        help="set path of log file")
    parser.add_argument("--version", action="version", version=extrap.__title__ + " " + extrap.__version__)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--cube", action="store_true", default=False, dest="cube", help="load data from cube files")
    group.add_argument("--text", action="store_true", default=False, dest="text", help="load data from text files")
    group.add_argument("--talpas", action="store_true", default=False, dest="talpas",
                       help="load data from talpas data format")
    group.add_argument("--json", action="store_true", default=False, dest="json",
                       help="load data from json or jsonlines file")
    group.add_argument("--extra-p-3", action="store_true", default=False, dest="extrap3",
                       help="load data from Extra-P 3 experiment")
    parser.add_argument("path", metavar="FILEPATH", type=str, action="store", nargs='?',
                        help="specify a file path for Extra-P to work with")

    parser.add_argument("--scaling", action="store", dest="scaling_type", default="weak", type=str.lower,
                        choices=["weak", "strong"],
                        help="set weak or strong scaling when loading data from cube files [weak (default), strong]")
    arguments = parser.parse_args(args)
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
