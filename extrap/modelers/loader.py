# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Mapping, Type, MutableMapping

from marshmallow import fields, validate

from extrap.util.extension_loader import load_extensions
from extrap.util.serialization_schema import NumberField
from .abstract_modeler import AbstractModeler, ModelerSchema
from .modeler_options import ModelerOption, modeler_options


def load_modelers(path, pkg_name) -> MutableMapping[str, Type[AbstractModeler]]:
    return load_extensions(path, pkg_name, AbstractModeler, create_schema)


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
