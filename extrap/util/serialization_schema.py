# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import collections
import math
import typing
from abc import abstractmethod, ABC
from collections.abc import Mapping
from typing import Union, Sequence, Tuple, Type

from marshmallow import post_load, fields, ValidationError, EXCLUDE
from marshmallow.base import SchemaABC
from marshmallow.fields import Field
from marshmallow.schema import Schema as _Schema

from extrap.entities.fraction import Fraction
from extrap.util.exceptions import SerializationError


class SchemaMeta(type(_Schema), type(ABC)):
    pass


class Schema(_Schema, ABC, metaclass=SchemaMeta):
    class Meta:
        ordered = True
        unknown = EXCLUDE

    @abstractmethod
    def create_object(self) -> Union[object, Tuple[type(NotImplemented), Type]]:
        raise NotImplementedError()

    def postprocess_object(self, obj: object) -> object:
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

    def on_missing_sub_schema(self, type_, data, **kwargs):
        """Handles missing subschema. May return a parsed object to fail gracefully."""
        raise ValidationError(f'No subschema found for {type_} in {type(self).__name__}')

    def load(self, data, **kwargs):
        if self.__is_direct_subclass(type(self), BaseSchema) and self.type_field in data:
            type_ = data[self.type_field]
            del data[self.type_field]
            try:
                schema = self._subclasses[type_]()
            except KeyError:
                return self.on_missing_sub_schema(type, data, **kwargs)
            return schema.load(data, **kwargs)
        else:
            return super(BaseSchema, self).load(data, **kwargs)

    def dump(self, obj, **kwargs):
        obj_type = type(obj)
        if self.__is_direct_subclass(type(self), BaseSchema) and obj_type.__name__ in self._subclasses:
            return self._subclasses[obj_type.__name__]().dump(obj, **kwargs)
        else:
            try:
                serialization_type = type(self.create_object())
            except NotImplementedError as e:
                raise SerializationError() from e
            if serialization_type != obj_type:
                raise SerializationError(f"The serialization schema ({type(self)}, {serialization_type}) does not "
                                         f"match the type of the serialized object ({obj_type}).")
            try:
                result = super(BaseSchema, self).dump(obj, **kwargs)
                if obj_type.__name__ in self._subclasses:
                    result[self.type_field] = obj_type.__name__
                return result
            except Exception as e:
                raise SerializationError(f"Serialization in {type(self).__name__} failed. " + str(e)) from e


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
        elif isinstance(value, complex):
            raise SerializationError("The number field does not support serialization of complex values.")
        elif math.isnan(value) or math.isinf(value):
            return self._to_string(value)
        else:
            ret = super()._format_num(value)
        return self._to_string(ret) if self.as_string else ret


class ListToMappingField(fields.List):
    def __init__(self, nested: typing.Union[SchemaABC, type, str, typing.Callable[[], SchemaABC]], key_field: str, *,
                 list_type=list, dump_condition=None, **kwargs):
        super().__init__(fields.Nested(nested), **kwargs)
        self.dump_condition = dump_condition
        self.list_type = list_type
        only_field = self.inner.schema.fields[key_field]
        self.key_field_name = only_field.data_key or key_field
        self.inner: fields.Nested

    def _serialize(self, value: typing.Any, attr: str, obj: typing.Any, **kwargs):
        if value is None:
            return None
        if self.dump_condition:
            value = [v for v in value if self.dump_condition(v)]
        value = super()._serialize(value, attr, obj, **kwargs)
        result = {}
        for each in value:
            key = each[self.key_field_name]
            del each[self.key_field_name]
            result[key] = each
        return result

    def _deserialize(self, value: typing.Any, attr: str, data: typing.Optional[typing.Mapping[str, typing.Any]],
                     **kwargs) -> typing.List[typing.Any]:
        if not isinstance(value, Mapping):
            raise self.make_error("invalid")

        value_list = []
        for k, v in value.items():
            if not isinstance(v, typing.MutableMapping):
                raise self.make_error("invalid")
            v[self.key_field_name] = k
            value_list.append(v)

        result = super()._deserialize(value_list, attr, data, **kwargs)

        if self.list_type != list:
            return self.list_type(result)
        else:
            return result
