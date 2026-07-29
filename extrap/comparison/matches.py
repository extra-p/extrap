# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from abc import ABC, abstractmethod
from typing import TypeVar, Callable, Sequence, Mapping, MutableMapping, Optional

_T = TypeVar('_T')

AbstractMatches = Mapping[_T, Sequence[_T]]

MutableAbstractMatches = MutableMapping[_T, Sequence[Optional[_T]]]


class IdentityMatches(AbstractMatches[_T]):
    def __init__(self, num_matches: int, items):
        self._items = items
        self.num_matches = num_matches

    def __getitem__(self, target: _T) -> Sequence[_T]:
        return [target] * self.num_matches

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

# class DictMatches(Matches[_T]):
#     def __init__(self, matches: Dict[_T,Sequence[_T]]):
#         self.num_matches = num_matches
#
#     def __call__(self, target: _T) -> Sequence[_T]:
#         return [target] * self.num_matches
#
#     def get(self, target: _T, idx: int) -> _T:
#         return target
