# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import argparse
import logging
import sys
import threading
import traceback
import warnings

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication, QMessageBox, QToolTip
from matplotlib import font_manager

import extrap
from extrap.fileio.experiment_io import read_experiment
from extrap.fileio.file_reader import all_readers
from extrap.fileio.file_reader.cube_file_reader2 import CubeFileReader2
from extrap.gui.MainWidget import MainWidget
from extrap.util.exceptions import RecoverableError, CancelProcessError

TRACEBACK = logging.DEBUG - 1
logging.addLevelName(TRACEBACK, 'TRACEBACK')


def main(*, args=None, test=False):
    _update_mac_app_info()
    # preload fonts for matplotlib
    font_preloader = _preload_common_fonts()
    arguments = parse_arguments(args)

    # configure logging
    log_level = min(logging.getLevelName(arguments.log_level.upper()), logging.INFO)
    if arguments.log_file:
        logging.basicConfig(format="%(levelname)s: %(asctime)s: %(message)s", level=log_level,
                            filename=arguments.log_file)
    else:
        logging.basicConfig(format="%(levelname)s: %(asctime)s: %(message)s", level=log_level)
    logging.getLogger().handlers[0].setLevel(logging.getLevelName(arguments.log_level.upper()))

    app = QApplication(sys.argv) if not test else QApplication.instance()
    apply_style(app)

    window = MainWidget()

    _init_warning_system(window, test)

    window.show()

    try:
        load_from_command(arguments, window)
    except CancelProcessError:
        pass

    if not test:
        app.exec()
        font_preloader.join()
    else:
        font_preloader.join()
        return window, app


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description=extrap.__description__)
    parser.add_argument("--log", action="store", dest="log_level", type=str.lower, default='critical',
                        choices=['traceback', 'debug', 'info', 'warning', 'error', 'critical'],
                        help="set program's log level (default: critical)")
    parser.add_argument("--logfile", action="store", dest="log_file",
                        help="set path of log file")
    parser.add_argument("--version", action="version", version=extrap.__title__ + " " + extrap.__version__)

    group = parser.add_mutually_exclusive_group(required=False)
    for reader in all_readers.values():
        group.add_argument(reader.CMD_ARGUMENT, action="store_true", default=False, dest=reader.NAME,
                           help=reader.DESCRIPTION)

    parser.add_argument("path", metavar="FILEPATH", type=str, action="store", nargs='?',
                        help="specify a file path for Extra-P to work with")

    parser.add_argument("--scaling", action="store", dest="scaling_type", default="weak", type=str.lower,
                        choices=["weak", "strong"],
                        help="set weak or strong scaling when loading data from cube files [weak (default), strong]")
    arguments = parser.parse_args(args)
    return arguments


def load_from_command(arguments, window):
    if arguments.path:
        for reader in all_readers.values():
            if getattr(arguments, reader.NAME):
                file_reader = reader()
                if file_reader is CubeFileReader2:
                    file_reader.scaling_type = arguments.scaling_type
                window.import_file(file_reader.read_experiment, file_name=arguments.path,
                                   model=file_reader.GENERATE_MODELS_AFTER_LOAD)
                return
        window.import_file(read_experiment, file_name=arguments.path, model=False)


def _init_warning_system(window, test=False):
    open_message_boxes = []
    current_warnings = set()

    # save old handlers
    _old_warnings_handler = warnings.showwarning
    _old_exception_handler = sys.excepthook

    def activate_box(box):
        box.raise_()
        box.activateWindow()

    def display_messages(event):
        for w in open_message_boxes:
            w.raise_()
            w.activateWindow()

    if sys.platform.startswith('darwin'):
        window.activate_event_handlers.append(display_messages)

    def _warnings_handler(message: Warning, category, filename, lineno, file=None, line=None):
        nonlocal current_warnings
        message_str = str(message)
        if message_str not in current_warnings:
            warn_box = QMessageBox(QMessageBox.Warning, 'Warning', message_str, QMessageBox.Ok, window)
            warn_box.setModal(False)
            warn_box.setAttribute(Qt.WA_DeleteOnClose)
            warn_box.destroyed.connect(
                lambda x: (current_warnings.remove(message_str), open_message_boxes.remove(warn_box)))

            if not test:
                warn_box.show()
                activate_box(warn_box)
                open_message_boxes.append(warn_box)

            current_warnings.add(message_str)
            _old_warnings_handler(message, category, filename, lineno, file, line)

        logging.warning(message_str)
        logging.log(TRACEBACK, ''.join(traceback.format_stack()))
        QApplication.processEvents()

    def _exception_handler(type, value, traceback_):
        traceback_text = ''.join(traceback.extract_tb(traceback_).format())

        if issubclass(type, CancelProcessError):
            logging.log(TRACEBACK, str(value))
            logging.log(TRACEBACK, traceback_text)
            return

        parent, modal = _parent(window)
        msg_box = QMessageBox(QMessageBox.Critical, 'Error', str(value), QMessageBox.Ok, parent)
        print()
        if hasattr(value, 'NAME'):
            msg_box.setWindowTitle(getattr(value, 'NAME'))
        msg_box.setDetailedText(traceback_text)
        open_message_boxes.append(msg_box)

        logging.error(str(value))
        logging.log(TRACEBACK, traceback_text)

        if test:
            return _old_exception_handler(type, value, traceback_)
        _old_exception_handler(type, value, traceback_)
        if issubclass(type, RecoverableError):
            msg_box.open()
            activate_box(msg_box)
        else:
            activate_box(msg_box)
            msg_box.exec()  # ensures waiting
            exit(1)

    warnings.showwarning = _warnings_handler
    sys.excepthook = _exception_handler
    warnings.simplefilter('always', UserWarning)


def apply_style(app):
    app.setStyle('Fusion')

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
    palette.setColor(QPalette.ToolTipBase, QColor(230, 230, 230))
    palette.setColor(QPalette.ToolTipText, Qt.black)
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(80, 80, 80))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(80, 80, 80))
    palette.setColor(QPalette.Disabled, QPalette.Button, QColor(150, 150, 150))
    app.setPalette(palette)
    QToolTip.setPalette(palette)


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


def _parent(window):
    if not sys.platform.startswith('darwin'):
        return window, False
    modal = QApplication.activeModalWidget()
    parent = modal if modal else window
    return parent, bool(modal)


def _update_mac_app_info():
    if sys.platform.startswith('darwin'):
        try:
            from Foundation import NSBundle  # noqa
            bundle = NSBundle.mainBundle()
            if bundle:
                app_info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
                if app_info:
                    app_info['CFBundleName'] = extrap.__title__
            from AppKit import NSWindow
            NSWindow.setAllowsAutomaticWindowTabbing_(False)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
