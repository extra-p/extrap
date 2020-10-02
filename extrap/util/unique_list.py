# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Sequence, Iterable, TypeVar

T = TypeVar('T')


class UniqueList(list, Sequence[T]):
    def __init__(self, iterable=...):
        super().__init__()
        self._set = set()
        if isinstance(iterable, Iterable):
            self.extend(iterable)

    def __setitem__(self, i, value):
        raise NotImplementedError

    def __contains__(self, item):
        return item in self._set

    def append(self, item):
        if item in self._set:
            return False
        self._set.add(item)
        super().append(item)
        return True

    def extend(self, items):
        for i in items:
            if i in self._set:
                continue
            else:
                self.append(i)

    def remove(self, item):
        super().remove(item)
        self._set.remove(item)

    def __iadd__(self, items):
        self.extend(items)
        return self

    def __delitem__(self, i):
        item = super().__getitem__(i)
        super().__delitem__(i)
        self._set.remove(item)
