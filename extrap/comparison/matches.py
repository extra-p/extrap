from abc import ABC, abstractmethod
from typing import TypeVar, Callable, Sequence, Mapping

_T = TypeVar('_T')

AbstractMatches = Mapping[_T, Sequence[_T]]


class IdentityMatches(AbstractMatches[_T]):
    def __init__(self, num_matches: int, items):
        self.items = items
        self.num_matches = num_matches

    def __getitem__(self, target: _T) -> Sequence[_T]:
        return [target] * self.num_matches

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

# class DictMatches(Matches[_T]):
#     def __init__(self, matches: Dict[_T,Sequence[_T]]):
#         self.num_matches = num_matches
#
#     def __call__(self, target: _T) -> Sequence[_T]:
#         return [target] * self.num_matches
#
#     def get(self, target: _T, idx: int) -> _T:
#         return target
