# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import inspect
import pkgutil
from typing import Mapping, Type, MutableMapping, TypeVar, Iterator

from marshmallow import fields, validate

from extrap.util.serialization_schema import NumberField
from .abstract_modeler import AbstractModeler, ModelerSchema
from .modeler_options import ModelerOption, modeler_options

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
        return iter(self.data)

    def __getitem__(self, k: str) -> _VT:
        k = self.resolve_key(k)
        return self.data[k]

    def __setitem__(self, k: str, value: _VT):
        self.case_mapping[k.lower()] = k
        self.data[k] = value


def load_modelers(path, pkg_name) -> MutableMapping[str, Type[AbstractModeler]]:
    def is_modeler(x):
        return inspect.isclass(x) \
               and issubclass(x, AbstractModeler) \
               and not inspect.isabstract(x)

    modelers = CaseInsensitiveStringDict()
    for importer, modname, is_pkg in pkgutil.walk_packages(path=path,
                                                           prefix=pkg_name + '.',
                                                           onerror=lambda x: None):
        module = importer.find_module(modname).load_module(modname)
        for name, clazz in inspect.getmembers(module, is_modeler):
            clazz: Type[AbstractModeler]
            name = clazz.NAME
            modelers[name] = clazz
            create_schema(clazz)
    return modelers


def _determine_field(option: ModelerOption):
    if option.range:
        if isinstance(option.range, range):
            if range.step == 1:
                validation = validate.Range(option.range.start, option.range.stop, max_inclusive=False)
            else:
                validation = validate.OneOf(option.range)
        elif isinstance(option.range, Mapping):
            validation = validate.OneOf(list(option.range.values()), labels=list(option.range.keys()))
        else:
            validation = validate.OneOf(option.range)
    else:
        validation = None

    kwargs = {
        'validation': validation,
        'default': option.value,
        'required': False,
        'allow_none': True
    }

    if option.type is int:
        return fields.Int(**kwargs)
    elif option.type is float:
        return NumberField(**kwargs)
    elif option.type is bool:
        return fields.Bool(**kwargs)
    elif option.type is str:
        return fields.Str(**kwargs)
    else:
        return fields.Function(serialize=str, deserialize=option.type, **kwargs)


def create_schema(cls):
    attribute_fields = {'create_object': lambda self: cls()}
    for o in modeler_options.iter(cls):
        attribute_fields[o.field] = _determine_field(o)
    cls_schema = type(cls.__name__ + 'Schema', (ModelerSchema,), attribute_fields)
    globals()[cls_schema.__name__] = cls_schema
