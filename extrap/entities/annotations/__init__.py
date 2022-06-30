# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

from abc import abstractmethod, ABC
from typing import NewType

from extrap.util.classproperty import classproperty
from extrap.util.serialization_schema import BaseSchema


class Annotation(ABC):
    @classproperty
    @abstractmethod
    def NAME(cls) -> str:  # noqa
        """
        Name of the annotation
        """
        raise NotImplementedError

    @abstractmethod
    def title(self, **context) -> str:
        """
        Title of the annotation
        :param context: Context for improved visualization, should contain the names of the parameters
        """
        raise NotImplementedError

    @abstractmethod
    def icon(self, **context) -> AnnotationIconSVG:
        """
        Icon to visualize the annotation

        :param context: Context for improved visualization, should contain the names of the parameters
        :returns: An icon in form of an SVG string that can be displayed as an icon to visualize the annotation
        """
        raise NotImplementedError

    @abstractmethod
    def content(self, **context):
        """
        Content of the annotation, which will be shown to the user
        :param context: Context for improved visualization, should contain the names of the parameters
        """
        raise NotImplementedError


AnnotationIconSVG = NewType('AnnotationIconSVG', str)


class AnnotationSchema(BaseSchema):
    def create_object(self):
        return None
