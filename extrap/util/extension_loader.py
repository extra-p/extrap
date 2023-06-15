# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import importlib
import inspect
import pkgutil
from typing import TypeVar, MutableMapping, Iterator, Type, Callable

_VT = TypeVar("_VT")


class CaseInsensitiveStringDict(MutableMapping[str, _VT]):
    def __init__(self):
        self.case_mapping = {}
        self.data = {}

    def __delitem__(self, v: str) -> None:
        v = self.resolve_key(v)
        del self.data[v]
        del self.case_mapping[v.lower()]

    def resolve_key(self, k):
        if k not in self.data:
            return self.case_mapping[k.lower()]
        return k

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator:
        return iter(sorted(self.data))

    def __getitem__(self, k: str) -> _VT:
        k = self.resolve_key(k)
        return self.data[k]

    def __setitem__(self, k: str, value: _VT):
        self.case_mapping[k.lower()] = k
        self.data[k] = value


def load_extensions(path: str, pkg_name: str, type_: Type[_VT], post_process: Callable[[_VT], None] = None) -> \
        MutableMapping[str, Type[_VT]]:
    def is_modeler(x):
        return inspect.isclass(x) \
               and issubclass(x, type_) \
               and not inspect.isabstract(x)

    modelers = CaseInsensitiveStringDict()
    for _, modname, _ in pkgutil.walk_packages(path=path, prefix=pkg_name + '.', onerror=lambda x: None):
        # Do not use the returned importer. This may create two different classes with the same name and members.
        module = importlib.import_module(modname)
        for name, clazz in inspect.getmembers(module, is_modeler):
            clazz: Type[_VT]
            name = clazz.NAME
            modelers[name] = clazz
            if post_process:
                post_process(clazz)
    return modelers
