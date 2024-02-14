from typing import Sequence, Any, Union, Optional, Callable

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex
from PySide6.QtGui import Qt


class BasicTableModel(QAbstractTableModel):

    def __init__(self, header: Sequence[str], table: Sequence[Sequence[Any]], parent=None):
        super().__init__(parent)
        self._header = header
        self._data = table
        self.formatters: list[Optional[Callable[[Any], str]]] = [None] * len(header)

    def columnCount(self, parent: Union[
        QModelIndex, QPersistentModelIndex
    ] = ...) -> int:
        if parent.isValid():
            return 0
        return len(self._header)

    def data(self, index: Union[QModelIndex, QPersistentModelIndex], role: int = ...) -> Any:
        if role != Qt.DisplayRole and role != Qt.ToolTipRole:
            return None
        value = self._data[index.row()][index.column()]
        if self.formatters[index.column()]:
            return self.formatters[index.column()](value)
        else:
            return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> Any:
        if role != Qt.DisplayRole and role != Qt.ToolTipRole:
            return None
        if orientation == Qt.Orientation.Vertical:
            return None
        return self._header[section]

    def rowCount(self, parent: Union[QModelIndex, QPersistentModelIndex] = ...) -> int:
        if parent.isValid():
            return 0
        return len(self._data)

    def updateData(self):
        self.dataChanged.emit(self.index(0, 0), self.index(len(self._data), 1))
