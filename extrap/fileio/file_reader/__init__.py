# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

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
        """ This attribute is the unique display name of the FileReader.

        It is used for identifying the filetype internally.
        You must override this only in concrete FileReaders, you should do so by setting the class variable NAME.
        """
        raise NotImplementedError()

    @classproperty
    @abstractmethod
    def CMD_ARGUMENT(cls) -> bool:  # noqa
        """ This attribute is the unique cmd argument of the FileReader.

        It is used to determine the the cmd command to call this FileReader. You must override this only in concrete
        FileReaders, you should do so by setting the class variable CMD_ARGUMENT.
        """
        raise NotImplementedError()

    @classproperty
    @abstractmethod
    def GUI_ACTION(cls) -> str:  # noqa
        """ This attribute is the text of the FileReader's GUI action.

        It is used for selecting the FileReader in the GUI. You should override this only in concrete
        FileReaders, you should do so by setting the class variable GUI_ACTION.
        """
        raise NotImplementedError()

    @classproperty
    def DESCRIPTION(cls) -> str:  # noqa
        """ This attribute is the short description of the FileReader.

        It will be shown in Tooltips and the help message. You should override this only in concrete
        FileReaders, you should do so by setting the class variable DESCRIPTION.
        """
        return ""

    @classproperty
    def FILTER(cls) -> str:  # noqa
        """ This attribute is the filter for the files of the FileReader.

        It will be used to determine the files that are associated with this Reader. You should override this only in
        concrete FileReaders, you should do so by setting the class variable FILTER.
        """
        return ""

    @classproperty
    def GENERATE_MODELS_AFTER_LOAD(cls) -> bool:  # noqa
        return True

    @classproperty
    def LOADS_FROM_DIRECTORY(cls) -> bool:  # noqa
        return False

    @abstractmethod
    def read_experiment(self, path: Union[Path, str], progress_bar: ProgressBar = DUMMY_PROGRESS) -> Experiment:
        """ Reads a Experiment for a given path."""
        raise NotImplementedError


all_readers = load_extensions(__path__, __name__, FileReader)
