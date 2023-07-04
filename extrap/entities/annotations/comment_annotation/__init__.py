# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import importlib.resources

from extrap.entities.annotations import Annotation


class CommentsAnnotation(Annotation):
    NAME = "Comments"

    def __init__(self):
        self.comments: list[str] = []

    def title(self, **context):
        if len(self.comments) == 1:
            return "1 Comment"
        else:
            return f"{len(self.comments)} Comments"

    def icon(self, **context):
        data = importlib.resources.read_text(__name__, 'comment_base.svg')
        count = len(self.comments)
        if count > 99:
            count = '*'
        return data.format(count=count)

    def content(self, **context):
        return "\n\n".join(self.comments)
