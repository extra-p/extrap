# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: NvtxCategories.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='NvtxCategories.proto',
  package='NV.Nvtx',
  serialized_pb=_b('\n\x14NvtxCategories.proto\x12\x07NV.Nvtx*>\n\x0cNvtxCategory\x12\x17\n\x13NvtxCategoryInvalid\x10\x00\x12\x15\n\x11NvtxCategoryState\x10\x01')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

_NVTXCATEGORY = _descriptor.EnumDescriptor(
  name='NvtxCategory',
  full_name='NV.Nvtx.NvtxCategory',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NvtxCategoryInvalid', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NvtxCategoryState', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=33,
  serialized_end=95,
)
_sym_db.RegisterEnumDescriptor(_NVTXCATEGORY)

NvtxCategory = enum_type_wrapper.EnumTypeWrapper(_NVTXCATEGORY)
NvtxCategoryInvalid = 0
NvtxCategoryState = 1


DESCRIPTOR.enum_types_by_name['NvtxCategory'] = _NVTXCATEGORY


# @@protoc_insertion_point(module_scope)