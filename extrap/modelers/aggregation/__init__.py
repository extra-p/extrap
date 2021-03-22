# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from abc import abstractmethod, ABC
from typing import Sequence, Dict, Tuple, Optional

from extrap.entities.callpath import Callpath
from extrap.entities.calltree import CallTree
from extrap.entities.metric import Metric
from extrap.entities.model import Model
from extrap.util.classproperty import classproperty
from extrap.util.extension_loader import load_extensions
from extrap.util.progress_bar import DUMMY_PROGRESS


class Aggregation(ABC):

    @abstractmethod
    def aggregate(self, models: Dict[Tuple[Callpath, Metric], Model], calltree: CallTree, metrics: Sequence[Metric],
                  progress_bar=DUMMY_PROGRESS) -> Dict[Tuple[Callpath, Metric], Model]:
        """ Creates an aggregated model for each model.

        This method is the core of the aggregation system.
        It receives a thea call-tree and all models and returns aggregated versions of the models.
        """
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def NAME(cls) -> str:  # noqa
        """ This attribute is the unique display name of the aggregation.

        It is used for selecting the aggregation in the GUI and CLI.
        You must override this only in concrete aggregations, you should do so by setting the class variable NAME."""
        raise NotImplementedError

    @classproperty
    def DESCRIPTION(cls) -> Optional[str]:  # noqa
        """ This attribute is the description of the aggregation.

        It is shown as additional information in the GUI and CLI.
        You should override this by setting the class variable DESCRIPTION."""
        return None


all_aggregations = load_extensions(__path__, __name__, Aggregation)