from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from extrap.entities.experiment import Experiment
from extrap.util.classproperty import classproperty
from extrap.util.extension_loader import load_extensions
from extrap.util.progress_bar import DUMMY_PROGRESS, ProgressBar


class FileReader(ABC):

    @classproperty
    def NAME(cls) -> str:  # noqa
        """ This attribute is the unique display name of the FileReader.

        It is used for identifying the filetype internally.
        You must override this only in concrete FileReaders, you should do so by setting the class variable NAME.
        """
        raise NotImplementedError

    @classproperty
    def CMD_ARGUMENT(cls) -> bool:  # noqa
        """ This attribute is the unique cmd argument of the FileReader.

        It is used to determine the the cmd command to call this FileReader. You must override this only in concrete
        FileReaders, you should do so by setting the class variable CMD_ARGUMENT.
        """
        raise NotImplementedError

    @classproperty
    def SHORT_DESCRIPTION(cls) -> Optional[str]:  # noqa
        """ This attribute is the short description of the FileReader.

        It is used for selecting the FileReader in the GUI. You should override this only in concrete
        FileReaders, you should do so by setting the class variable SHORT_DESCRIPTION.
        """
        return ""

    @classproperty
    def EXTENDED_DESCRIPTION(cls) -> Optional[str]:  # noqa
        """ This attribute is the short description of the FileReader.

        It will be shown in Tooltips. You should override this only in concrete
        FileReaders, you should do so by setting the class variable EXTENDED_DESCRIPTION.
        """
        return ""

    @classproperty
    def FILTER(cls) -> Optional[str]:  # noqa
        """ This attribute is the filter for the files of the FileReader.

        It will be used to determine the files that are associated with this Reader. You should override this only in
        concrete FileReaders, you should do so by setting the class variable FILTER.
        """
        return ""

    @classproperty
    def IS_MODEL(cls) -> Optional[bool]:  # noqa
        return True

    @abstractmethod
    def read_experiment(self, path: Union[Path, str], progress_bar: ProgressBar = DUMMY_PROGRESS) -> Experiment:
        """ Reads a Experiment for a given path."""
        raise NotImplementedError


all_reader = load_extensions(__path__, __name__, FileReader)
