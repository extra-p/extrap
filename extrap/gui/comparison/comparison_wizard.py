from asyncio import Event
from functools import partial
from itertools import chain
from typing import Optional, Type

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QWizard, QCommandLinkButton, QSpacerItem, QSizePolicy, QFileDialog

from extrap.comparison import matchers
from extrap.comparison.experiment_comparison import ComparisonExperiment
from extrap.fileio.experiment_io import ExperimentReader
from extrap.fileio.file_reader import all_readers, FileReader
from extrap.gui.comparison.interactive_matcher import InteractiveMatcher
from extrap.gui.components import file_dialog
from extrap.gui.components.wizard_pages import ScrollAreaPage, ProgressPage
from extrap.modelers.model_generator import ModelGenerator


class ComparisonWizard(QWizard):
    file_reader: Type[FileReader]
    file_name: str

    def __init__(self, experiment1, experiment2=None):
        super().__init__()
        self.setWindowTitle("Compare With Other Experiment")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        if not experiment2:
            self.addPage(FileSelectionPage(self))
            self.addPage(FileLoadingPage(self))
        self.addPage(MatcherSelectionPage(self))
        self.comparing_page_id = self.addPage(ComparingPage(self))
        self.matcher = None
        self.experiment1 = experiment1
        self.experiment2 = experiment2
        self.experiment: Optional[ComparisonExperiment] = None
        self.is_cancelled = Event()
        self.rejected.connect(self.on_reject)

    def on_reject(self):
        self.is_cancelled.set()

    def back(self) -> None:
        self.restart()


class FileSelectionPage(ScrollAreaPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Select input format')
        layout = self.scroll_layout
        for reader in chain([ExperimentReader], all_readers.values()):
            def _(reader):
                btn = QCommandLinkButton(reader.GUI_ACTION.replace('&', ''))
                # btn.setDescription(reader.DESCRIPTION)
                btn.clicked.connect(
                    lambda: file_dialog.show(self, partial(self.open_file, reader), reader.DESCRIPTION,
                                             filter=reader.FILTER,
                                             file_mode=QFileDialog.Directory if reader.LOADS_FROM_DIRECTORY else None))
                layout.addWidget(btn)

            _(reader)
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.scroll_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))

    def open_file(self, reader, name):
        self.wizard().file_name = name
        self.wizard().file_reader = reader
        self.wizard().next()

    def isComplete(self) -> bool:
        return False


class MatcherSelectionPage(ScrollAreaPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Select mapping provider')
        layout = self.scroll_layout
        for matcher in chain(matchers.all_matchers.values(), [InteractiveMatcher]):
            def _(matcher):
                btn = QCommandLinkButton(matcher.NAME)
                btn.setDescription(matcher.DESCRIPTION)
                btn.clicked.connect(lambda: self.select_matcher(matcher))
                layout.addWidget(btn)

            _(matcher)
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))

    def select_matcher(self, matcher):
        self.wizard().matcher = matcher
        self.wizard().next()

    def isComplete(self) -> bool:
        return False


class FileLoadingPage(ProgressPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Loading experiment')

    def do_process(self, pbar):
        wizard: ComparisonWizard = self.wizard()
        wizard.experiment2 = wizard.file_reader().read_experiment(wizard.file_name, pbar)
        if wizard.file_reader.GENERATE_MODELS_AFTER_LOAD:
            ModelGenerator(wizard.experiment2).model_all(pbar)


class ComparingPage(ProgressPage):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTitle('Comparing experiments')

    def cleanupPage(self) -> None:
        self._override_next_id = None
        super().cleanupPage()

    def do_process(self, pbar):
        wizard = self.wizard()
        if wizard.matcher == InteractiveMatcher:
            matcher = wizard.matcher(wizard)
        else:
            matcher = wizard.matcher()
        wizard.experiment = ComparisonExperiment(wizard.experiment1, wizard.experiment2, matcher=matcher)
        if wizard.matcher == InteractiveMatcher:
            self._override_next_id = matcher.determine_next_page_id()
        else:
            wizard.experiment.do_comparison(pbar)
