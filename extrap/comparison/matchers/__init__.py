# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from abc import ABC, abstractmethod
from typing import Sequence, Mapping, Tuple, Optional

from extrap.comparison.matches import MutableAbstractMatches, AbstractMatches
from extrap.comparison.metric_conversion import AbstractMetricConverter
from extrap.entities.calltree import CallTree, Node
from extrap.entities.metric import Metric
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.classproperty import classproperty
from extrap.util.extension_loader import load_extensions
from extrap.util.progress_bar import DUMMY_PROGRESS


class AbstractMatcher(ABC):
    @abstractmethod
    def match_metrics(self, *metric: Sequence[Metric], progress_bar=DUMMY_PROGRESS) -> Tuple[
        Sequence[Metric], AbstractMatches[Metric], Sequence[AbstractMetricConverter]]:
        pass

    @abstractmethod
    def match_call_tree(self, *call_tree: CallTree, progress_bar=DUMMY_PROGRESS) -> Tuple[
        CallTree, MutableAbstractMatches[Node]]:
        pass

    @abstractmethod
    def match_modelers(self, *mg: Sequence[ModelGenerator], progress_bar=DUMMY_PROGRESS) -> Mapping[
        str, Sequence[ModelGenerator]]:
        pass

    @classproperty
    @abstractmethod
    def NAME(cls) -> str:  # noqa
        """ This attribute is the unique display name of the matcher.

        It is used for selecting the matcher in the GUI and CLI.
        You must override this only in concrete matchers, you should do so by setting the class variable NAME."""
        raise NotImplementedError

    @classproperty
    def DESCRIPTION(cls) -> Optional[str]:  # noqa
        """ This attribute is the description of the matcher.

        It is shown as additional information in the GUI and CLI.
        You should override this by setting the class variable DESCRIPTION."""
        return None


all_matchers = load_extensions(__path__, __name__, AbstractMatcher)
