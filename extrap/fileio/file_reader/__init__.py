from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from extrap.entities.experiment import Experiment
from extrap.util.classproperty import classproperty
from extrap.util.extension_loader import load_extensions
from extrap.util.progress_bar import DUMMY_PROGRESS, ProgressBar


class FileReader(ABC):

    @classproperty
    @abstractmethod
    def NAME(cls) -> str:  # noqa
        raise NotImplementedError

    @classproperty
    def SHORT_NAME(cls) -> str:  # noqa
        raise NotImplementedError

    @classproperty
    def LONG_NAME(cls) -> str:  # noqa
        raise NotImplementedError

    @classproperty
    def FILTER(cls) -> str:  # noqa
        return ""

    @classproperty
    def CMD_ARGUMENT(cls) -> bool:  # noqa
        raise NotImplementedError

    @classproperty
    def IS_MODEL(cls) -> bool:  # noqa
        return True

    @abstractmethod
    def read_experiment(self, path: Union[Path, str], progress_bar: ProgressBar = DUMMY_PROGRESS) -> Experiment:
        raise NotImplementedError


all_reader = load_extensions(__path__, __name__, FileReader)
