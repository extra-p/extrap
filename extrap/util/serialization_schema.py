# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import collections
import math
from abc import abstractmethod, ABC
from collections.abc import Mapping
from typing import Union, Sequence

from marshmallow import post_load, fields, ValidationError, EXCLUDE
from marshmallow.fields import Field
from marshmallow.schema import Schema as _Schema

from extrap.entities.fraction import Fraction


class SchemaMeta(type(_Schema), type(ABC)):
    pass


class Schema(_Schema, ABC, metaclass=SchemaMeta):
    class Meta:
        ordered = True
        unknown = EXCLUDE

    @abstractmethod
    def create_object(self):
        raise NotImplementedError()

    def postprocess_object(self, obj):
        return obj

    @post_load
    def unpack_to_object(self, data, **kwargs):
        obj = self.create_object()
        try:
            for k, v in data.items():
                setattr(obj, k, v)
        except AttributeError as e:
            print(e)
        return self.postprocess_object(obj)


class BaseSchema(Schema):
    _subclasses = None
    type_field = '$type'

    def create_object(self):
        raise NotImplementedError(f"{type(self)} has no create object method.")

    def __init_subclass__(cls, **kwargs):
        if not cls.__is_direct_subclass(cls, BaseSchema):
            obj = cls().create_object()
            if isinstance(obj, tuple) and obj[0] == NotImplemented:
                cls._subclasses[obj[1].__name__] = cls
            else:
                cls._subclasses[type(cls().create_object()).__name__] = cls
        else:
            cls._subclasses = {}
        super().__init_subclass__(**kwargs)

    @staticmethod
    def __is_direct_subclass(subclass, classs_):
        return subclass.mro()[1] == classs_

    def load(self, data, **kwargs):
        if self.__is_direct_subclass(type(self), BaseSchema) and self.type_field in data:
            type_ = data[self.type_field]
            del data[self.type_field]
            try:
                schema = self._subclasses[type_]()
            except KeyError:
                raise ValidationError(f'No subschema found for {type_} in {type(self).__name__}')
            return schema.load(data, **kwargs)
        else:
            return super(BaseSchema, self).load(data, **kwargs)

    def dump(self, obj, **kwargs):
        if self.__is_direct_subclass(type(self), BaseSchema) and type(obj).__name__ in self._subclasses:
            return self._subclasses[type(obj).__name__]().dump(obj, **kwargs)
        else:
            result = super(BaseSchema, self).dump(obj, **kwargs)
            if type(obj).__name__ in self._subclasses:
                result[self.type_field] = type(obj).__name__
            return result


def make_value_schema(class_, value):
    class Schema(_Schema):
        def dump(self, obj, *, many: bool = None):
            return getattr(obj, value)

        def load(self, data, *, many: bool = None, partial=None, unknown: str = None):
            return class_(data)

    return Schema


class TupleKeyDict(fields.Mapping):
    mapping_type = dict

    def __init__(self,
                 keys: Sequence[Union[Field, type]],
                 values: Union[Field, type] = None,
                 **kwargs):
        super(TupleKeyDict, self).__init__(fields.Tuple(keys), values, **kwargs)

    # noinspection PyProtectedMember
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None

        # Serialize keys
        keys = {
            k: self.key_field._serialize(k, None, None, **kwargs)
            for k in value.keys()
        }

        # Serialize values
        result = self.mapping_type()
        if self.value_field is None:
            for mk, v in value.items():
                curr_dict = result
                for k in keys[mk][:-1]:
                    if k not in curr_dict:
                        curr_dict[k] = self.mapping_type()
                    curr_dict = curr_dict[k]
                curr_dict[keys[mk][-1]] = v
        else:
            for mk, v in value.items():
                curr_dict = result
                for k in keys[mk][:-1]:
                    if k not in curr_dict:
                        curr_dict[k] = self.mapping_type()
                    curr_dict = curr_dict[k]
                curr_dict[keys[mk][-1]] = self.value_field._serialize(v, None, None, **kwargs)

        return result

    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, Mapping):
            raise self.make_error("invalid")

        errors = collections.defaultdict(dict)

        def flatten(d, agg, parent_key, k_fields):
            field, *k_fields = k_fields
            if not isinstance(d, dict):
                raise ValidationError(f'Expected dict found: {d}')
            for key, v in d.items():
                try:
                    new_key = parent_key + [field.deserialize(key, **kwargs)]
                    if k_fields:
                        flatten(v, agg, new_key, k_fields)
                    else:
                        agg[tuple(new_key)] = v
                except ValidationError as error:
                    errors[key]["key"] = error.messages
            return agg

        value_dict = flatten(value, self.mapping_type(), [], list(self.key_field.tuple_fields))

        # Deserialize values
        result = self.mapping_type()
        if self.value_field is None:
            for key, val in value_dict.items():
                result[key] = val
        else:
            for key, val in value_dict.items():
                try:
                    deser_val = self.value_field.deserialize(val, **kwargs)
                except ValidationError as error:
                    errors[key]["value"] = error.messages
                    if error.valid_data is not None:
                        result[key] = error.valid_data
                else:
                    result[key] = deser_val

        if errors:
            raise ValidationError(errors, valid_data=result)

        return result


class NumberField(fields.Number):

    def _format_num(self, value):
        """Return the number value for value, given this field's `num_type`."""
        if not isinstance(value, str):
            return super()._format_num(value)
        elif value.lower() in ['nan', 'inf', '-inf']:
            return float(value)
        elif '/' in value:
            return Fraction(value)
        else:
            return super()._format_num(value)

    def _serialize(self, value, attr, obj, **kwargs):
        """Return a string if `self.as_string=True`, otherwise return this field's `num_type`."""
        if value is None:
            return None
        elif isinstance(value, Fraction):
            return self._to_string(value)
        elif math.isnan(value) or math.isinf(value):
            return self._to_string(value)
        else:
            ret = super()._format_num(value)
        return self._to_string(ret) if self.as_string else ret
