
# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from PySide6.QtWidgets import QLayout

def clear_layout(layout: QLayout):
    if not layout:
        return

    while (child := layout.takeAt(0)) is not None:
        if child_layout := child.layout():
            clear_layout(child_layout)
            del child_layout
        if widget := child.widget():
            widget.deleteLater()
        del child
