from PySide2.QtCore import Slot, QRunnable, QObject


class Worker(QRunnable):
    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.function = function
        self.args = args
        self.kwargs = kwargs

    @Slot()  # QtCore.Slot
    def run(self):
        self.function(*self.args, **self.kwargs)


class MTWorker(QObject):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        # Store constructor arguments (re-used for processing)
        self.function = function
        self.args = args
        self.kwargs = kwargs

    @Slot()  # QtCore.Slot
    def run(self):
        self.function(*self.args, **self.kwargs)
