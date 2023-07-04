# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from PySide6.QtWidgets import QFileDialog


def show(parent, on_accept, caption='', directory='', filter='', file_mode=None, accept_mode=QFileDialog.AcceptOpen):
    if file_mode is None:
        file_mode = QFileDialog.ExistingFile if accept_mode == QFileDialog.AcceptOpen else QFileDialog.AnyFile
    f_dialog = QFileDialog(parent, caption, directory, filter)
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


showOpen = show


def showSave(parent, on_accept, caption='', directory='', filter=''):
    return show(parent, on_accept, caption, directory, filter, accept_mode=QFileDialog.AcceptSave)


def showOpenDirectory(parent, on_accept, caption='', directory='', filter=''):
    return show(parent, on_accept, caption, directory, filter, file_mode=QFileDialog.Directory)
