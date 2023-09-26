# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import argparse
import logging
import sys
import threading
import traceback
import warnings

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication, QMessageBox, QToolTip
from matplotlib import font_manager

import extrap
from extrap.entities.scaling_type import ScalingType
from extrap.fileio.experiment_io import read_experiment
from extrap.fileio.file_reader import all_readers
from extrap.fileio.file_reader.abstract_directory_reader import AbstractScalingConversionReader
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

    QCoreApplication.setApplicationName(extrap.__title__)
    QCoreApplication.setApplicationVersion(extrap.__version__)
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
    parser = argparse.ArgumentParser(description=extrap.__description__, add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit')
    parser.add_argument("--log", action="store", dest="log_level", type=str.lower, default='critical',
                        choices=['traceback', 'debug', 'info', 'warning', 'error', 'critical'],
                        help="Set program's log level (default: %(default)s)")
    parser.add_argument("--logfile", action="store", dest="log_file",
                        help="Set path of log file")
    parser.add_argument("--version", action="version", version=extrap.__title__ + " " + extrap.__version__,
                        help="Show program's version number and exit")

    group = parser.add_mutually_exclusive_group(required=False)
    for reader in all_readers.values():
        group.add_argument(reader.CMD_ARGUMENT, action="store_true", default=False, dest=reader.NAME,
                           help=reader.DESCRIPTION)

    parser.add_argument("path", metavar="FILEPATH", type=str, action="store", nargs='?',
                        help="Specify a file path for Extra-P to work with")

    names_of_scaling_conversion_readers = ", ".join(reader.NAME + " files" for reader in all_readers.values()
                                                    if issubclass(reader, AbstractScalingConversionReader))

    parser.add_argument("--scaling", action="store", dest="scaling_type", default=ScalingType.WEAK,
                        type=ScalingType, choices=ScalingType,
                        help="Set scaling type when loading data from per-thread/per-rank files (" +
                             names_of_scaling_conversion_readers + ") (default: %(default)s)")
    arguments = parser.parse_args(args)
    return arguments


def load_from_command(arguments, window):
    if arguments.path:
        for reader in all_readers.values():
            if getattr(arguments, reader.NAME):
                file_reader = reader()
                if issubclass(reader, AbstractScalingConversionReader):
                    file_reader.scaling_type = arguments.scaling_type
                elif arguments.scaling_type != ScalingType.WEAK:
                    warnings.warn(
                        f"Scaling type {arguments.scaling_type} is not supported by the {reader.NAME} reader.")
                window.import_file(file_reader.read_experiment, file_name=arguments.path,
                                   model=file_reader.GENERATE_MODELS_AFTER_LOAD, reader=file_reader)
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

    def display_messages(_event):
        for w in open_message_boxes:
            w.raise_()
            w.activateWindow()

    if sys.platform.startswith('darwin'):
        window.activate_event_handlers.append(display_messages)

    def _warnings_handler(message: Warning, category, filename, lineno, file=None, line=None):
        nonlocal current_warnings
        message_str = str(message)
        if message_str not in current_warnings:
            warn_box = QMessageBox(QMessageBox.Icon.Warning, 'Warning', message_str, QMessageBox.StandardButton.Ok,
                                   window)
            warn_box.setModal(False)
            warn_box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
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

    def _exception_handler(type_, value, traceback_):
        traceback_text = ''.join(traceback.extract_tb(traceback_).format())

        if issubclass(type_, CancelProcessError):
            logging.log(TRACEBACK, str(value))
            logging.log(TRACEBACK, traceback_text)
            return

        parent, modal = _parent(window)
        msg_box = QMessageBox(QMessageBox.Icon.Critical, 'Error', str(value), QMessageBox.StandardButton.Ok, parent)
        print()
        if hasattr(value, 'NAME'):
            msg_box.setWindowTitle(getattr(value, 'NAME'))
        msg_box.setDetailedText(traceback_text)
        open_message_boxes.append(msg_box)

        logging.error(str(value))
        logging.log(TRACEBACK, traceback_text)

        if test:
            return _old_exception_handler(type_, value, traceback_)
        _old_exception_handler(type_, value, traceback_)
        if issubclass(type_, RecoverableError):
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
    palette.setColor(QPalette.ColorRole.Window, QColor(200, 200, 200))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Base, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(10, 10, 10))
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Button, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(31, 119, 180))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(80, 80, 80))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(80, 80, 80))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor(150, 150, 150))
    palette.setColor(QPalette.ColorRole.Link, QColor(21, 83, 123))
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
            # noinspection PyPackageRequirements
            from AppKit import NSWindow
            NSWindow.setAllowsAutomaticWindowTabbing_(False)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
