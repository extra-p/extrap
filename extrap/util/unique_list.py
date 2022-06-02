# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import copy
from typing import Sequence, Iterable, TypeVar

T = TypeVar('T')


class UniqueList(list, Sequence[T]):
    def __init__(self, iterable=...):
        super().__init__()
        self._set = set()
        if isinstance(iterable, Iterable):
            self.extend(iterable)

    def __setitem__(self, i, value):
        if isinstance(i, slice):
            raise NotImplementedError()

        old_value = self[i]
        if old_value == value:
            return
        elif value in self._set:

            raise ValueError('Value already exists')
        else:
            self._set.remove(old_value)
            self._set.add(value)
            super().__setitem__(i, value)

    def insert(self, index, item):
        if item in self._set:
            return False
        self._set.add(item)
        super().insert(index, item)
        return True

    def __contains__(self, item):
        return item in self._set

    def count(self, item) -> int:
        return 1 if item in self._set else 0

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

    def pop(self, index: int = ...):
        e = super().pop(index)
        self._set.remove(e)
        return e

    def __iadd__(self, items):
        self.extend(items)
        return self

    def __delitem__(self, i):
        item = super().__getitem__(i)
        super().__delitem__(i)
        self._set.remove(item)

    def clear(self) -> None:
        super(UniqueList, self).clear()
        self._set.clear()

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        obj = UniqueList()
        super(UniqueList, obj).extend(copy.deepcopy(e, memodict) for e in self)
        obj._set = copy.deepcopy(self._set, memodict)
        return obj
