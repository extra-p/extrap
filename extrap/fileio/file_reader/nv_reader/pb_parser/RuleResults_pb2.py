# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: RuleResults.proto

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


from . import ProfilerSection_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='RuleResults.proto',
  package='NV.RuleSystem',
  serialized_pb=_b('\n\x11RuleResults.proto\x12\rNV.RuleSystem\x1a\x15ProfilerSection.proto\"X\n\x11RuleResultMessage\x12\x0f\n\x07Message\x18\x01 \x02(\t\x12\x32\n\x04Type\x18\x02 \x02(\x0e\x32$.NV.RuleSystem.RuleResultMessageType\"(\n\x12RuleResultProposal\x12\x12\n\nIdentifier\x18\x01 \x02(\t\"\xe4\x02\n\x12RuleResultBodyItem\x12\x31\n\x07Message\x18\x01 \x01(\x0b\x32 .NV.RuleSystem.RuleResultMessage\x12\x30\n\x05Table\x18\x02 \x01(\x0b\x32!.NV.Profiler.ProfilerSectionTable\x12\x36\n\x08\x42\x61rChart\x18\x03 \x01(\x0b\x32$.NV.Profiler.ProfilerSectionBarChart\x12\x42\n\x0eHistogramChart\x18\x04 \x01(\x0b\x32*.NV.Profiler.ProfilerSectionHistogramChart\x12\x38\n\tLineChart\x18\x05 \x01(\x0b\x32%.NV.Profiler.ProfilerSectionLineChart\x12\x33\n\x08Proposal\x18\x06 \x01(\x0b\x32!.NV.RuleSystem.RuleResultProposal\"B\n\x0eRuleResultBody\x12\x30\n\x05Items\x18\x01 \x03(\x0b\x32!.NV.RuleSystem.RuleResultBodyItem\"}\n\nRuleResult\x12\x12\n\nIdentifier\x18\x01 \x02(\t\x12\x13\n\x0b\x44isplayName\x18\x02 \x02(\t\x12+\n\x04\x42ody\x18\x03 \x01(\x0b\x32\x1d.NV.RuleSystem.RuleResultBody\x12\x19\n\x11SectionIdentifier\x18\x04 \x01(\t\"=\n\x0bRuleResults\x12.\n\x0bRuleResults\x18\x01 \x03(\x0b\x32\x19.NV.RuleSystem.RuleResult*A\n\x15RuleResultMessageType\x12\x08\n\x04None\x10\x00\x12\x06\n\x02Ok\x10\x01\x12\x0b\n\x07Warning\x10\x02\x12\t\n\x05\x45rror\x10\x03')
  ,
  dependencies=[ProfilerSection_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

_RULERESULTMESSAGETYPE = _descriptor.EnumDescriptor(
  name='RuleResultMessageType',
  full_name='NV.RuleSystem.RuleResultMessageType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='None', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Ok', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Warning', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Error', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=808,
  serialized_end=873,
)
_sym_db.RegisterEnumDescriptor(_RULERESULTMESSAGETYPE)

RuleResultMessageType = enum_type_wrapper.EnumTypeWrapper(_RULERESULTMESSAGETYPE)
None_ = 0
Ok = 1
Warning = 2
Error = 3



_RULERESULTMESSAGE = _descriptor.Descriptor(
  name='RuleResultMessage',
  full_name='NV.RuleSystem.RuleResultMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Message', full_name='NV.RuleSystem.RuleResultMessage.Message', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='Type', full_name='NV.RuleSystem.RuleResultMessage.Type', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=59,
  serialized_end=147,
)


_RULERESULTPROPOSAL = _descriptor.Descriptor(
  name='RuleResultProposal',
  full_name='NV.RuleSystem.RuleResultProposal',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Identifier', full_name='NV.RuleSystem.RuleResultProposal.Identifier', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=149,
  serialized_end=189,
)


_RULERESULTBODYITEM = _descriptor.Descriptor(
  name='RuleResultBodyItem',
  full_name='NV.RuleSystem.RuleResultBodyItem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Message', full_name='NV.RuleSystem.RuleResultBodyItem.Message', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='Table', full_name='NV.RuleSystem.RuleResultBodyItem.Table', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='BarChart', full_name='NV.RuleSystem.RuleResultBodyItem.BarChart', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='HistogramChart', full_name='NV.RuleSystem.RuleResultBodyItem.HistogramChart', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='LineChart', full_name='NV.RuleSystem.RuleResultBodyItem.LineChart', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='Proposal', full_name='NV.RuleSystem.RuleResultBodyItem.Proposal', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=192,
  serialized_end=548,
)


_RULERESULTBODY = _descriptor.Descriptor(
  name='RuleResultBody',
  full_name='NV.RuleSystem.RuleResultBody',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Items', full_name='NV.RuleSystem.RuleResultBody.Items', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=550,
  serialized_end=616,
)


_RULERESULT = _descriptor.Descriptor(
  name='RuleResult',
  full_name='NV.RuleSystem.RuleResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Identifier', full_name='NV.RuleSystem.RuleResult.Identifier', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='DisplayName', full_name='NV.RuleSystem.RuleResult.DisplayName', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='Body', full_name='NV.RuleSystem.RuleResult.Body', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='SectionIdentifier', full_name='NV.RuleSystem.RuleResult.SectionIdentifier', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=618,
  serialized_end=743,
)


_RULERESULTS = _descriptor.Descriptor(
  name='RuleResults',
  full_name='NV.RuleSystem.RuleResults',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='RuleResults', full_name='NV.RuleSystem.RuleResults.RuleResults', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=745,
  serialized_end=806,
)

_RULERESULTMESSAGE.fields_by_name['Type'].enum_type = _RULERESULTMESSAGETYPE
_RULERESULTBODYITEM.fields_by_name['Message'].message_type = _RULERESULTMESSAGE
_RULERESULTBODYITEM.fields_by_name['Table'].message_type = ProfilerSection_pb2._PROFILERSECTIONTABLE
_RULERESULTBODYITEM.fields_by_name['BarChart'].message_type = ProfilerSection_pb2._PROFILERSECTIONBARCHART
_RULERESULTBODYITEM.fields_by_name['HistogramChart'].message_type = ProfilerSection_pb2._PROFILERSECTIONHISTOGRAMCHART
_RULERESULTBODYITEM.fields_by_name['LineChart'].message_type = ProfilerSection_pb2._PROFILERSECTIONLINECHART
_RULERESULTBODYITEM.fields_by_name['Proposal'].message_type = _RULERESULTPROPOSAL
_RULERESULTBODY.fields_by_name['Items'].message_type = _RULERESULTBODYITEM
_RULERESULT.fields_by_name['Body'].message_type = _RULERESULTBODY
_RULERESULTS.fields_by_name['RuleResults'].message_type = _RULERESULT
DESCRIPTOR.message_types_by_name['RuleResultMessage'] = _RULERESULTMESSAGE
DESCRIPTOR.message_types_by_name['RuleResultProposal'] = _RULERESULTPROPOSAL
DESCRIPTOR.message_types_by_name['RuleResultBodyItem'] = _RULERESULTBODYITEM
DESCRIPTOR.message_types_by_name['RuleResultBody'] = _RULERESULTBODY
DESCRIPTOR.message_types_by_name['RuleResult'] = _RULERESULT
DESCRIPTOR.message_types_by_name['RuleResults'] = _RULERESULTS
DESCRIPTOR.enum_types_by_name['RuleResultMessageType'] = _RULERESULTMESSAGETYPE

RuleResultMessage = _reflection.GeneratedProtocolMessageType('RuleResultMessage', (_message.Message,), dict(
  DESCRIPTOR = _RULERESULTMESSAGE,
  __module__ = 'RuleResults_pb2'
  # @@protoc_insertion_point(class_scope:NV.RuleSystem.RuleResultMessage)
  ))
_sym_db.RegisterMessage(RuleResultMessage)

RuleResultProposal = _reflection.GeneratedProtocolMessageType('RuleResultProposal', (_message.Message,), dict(
  DESCRIPTOR = _RULERESULTPROPOSAL,
  __module__ = 'RuleResults_pb2'
  # @@protoc_insertion_point(class_scope:NV.RuleSystem.RuleResultProposal)
  ))
_sym_db.RegisterMessage(RuleResultProposal)

RuleResultBodyItem = _reflection.GeneratedProtocolMessageType('RuleResultBodyItem', (_message.Message,), dict(
  DESCRIPTOR = _RULERESULTBODYITEM,
  __module__ = 'RuleResults_pb2'
  # @@protoc_insertion_point(class_scope:NV.RuleSystem.RuleResultBodyItem)
  ))
_sym_db.RegisterMessage(RuleResultBodyItem)

RuleResultBody = _reflection.GeneratedProtocolMessageType('RuleResultBody', (_message.Message,), dict(
  DESCRIPTOR = _RULERESULTBODY,
  __module__ = 'RuleResults_pb2'
  # @@protoc_insertion_point(class_scope:NV.RuleSystem.RuleResultBody)
  ))
_sym_db.RegisterMessage(RuleResultBody)

RuleResult = _reflection.GeneratedProtocolMessageType('RuleResult', (_message.Message,), dict(
  DESCRIPTOR = _RULERESULT,
  __module__ = 'RuleResults_pb2'
  # @@protoc_insertion_point(class_scope:NV.RuleSystem.RuleResult)
  ))
_sym_db.RegisterMessage(RuleResult)

RuleResults = _reflection.GeneratedProtocolMessageType('RuleResults', (_message.Message,), dict(
  DESCRIPTOR = _RULERESULTS,
  __module__ = 'RuleResults_pb2'
  # @@protoc_insertion_point(class_scope:NV.RuleSystem.RuleResults)
  ))
_sym_db.RegisterMessage(RuleResults)


# @@protoc_insertion_point(module_scope)