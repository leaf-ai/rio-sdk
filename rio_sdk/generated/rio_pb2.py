# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rio.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='rio.proto',
  package='rio',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\trio.proto\x12\x03rio\"\x87\x02\n\x0cTrainRequest\x12\x30\n\x11\x66ramework_variant\x18\x01 \x01(\x0e\x32\x15.rio.FrameworkVariant\x12$\n\x0bkernel_type\x18\x02 \x01(\x0e\x32\x0f.rio.KernelType\x12\x19\n\x11normed_train_data\x18\x03 \x01(\t\x12\x14\n\x0ctrain_labels\x18\x04 \x01(\t\x12\x19\n\x11train_predictions\x18\x05 \x01(\t\x12 \n\x18num_svgp_inducing_points\x18\x06 \x01(\x05\x12 \n\x18max_iterations_optimizer\x18\x07 \x01(\x05\x12\x0f\n\x07use_ard\x18\x08 \x01(\x08\"\xab\x01\n\x0ePredictRequest\x12\x18\n\x10normed_test_data\x18\x01 \x01(\t\x12\x18\n\x10test_predictions\x18\x02 \x01(\t\x12\r\n\x05model\x18\x03 \x01(\x0c\x12\x30\n\x11\x66ramework_variant\x18\x04 \x01(\x0e\x32\x15.rio.FrameworkVariant\x12$\n\x0bkernel_type\x18\x05 \x01(\x0e\x32\x0f.rio.KernelType\">\n\x0bTrainResult\x12 \n\x18\x63omputation_time_seconds\x18\x01 \x01(\x02\x12\r\n\x05model\x18\x02 \x01(\x0c\"L\n\rPredictResult\x12\x0c\n\x04mean\x18\x01 \x01(\t\x12\x0b\n\x03var\x18\x02 \x01(\t\x12 \n\x18\x63omputation_time_seconds\x18\x03 \x01(\x02*\x8e\x01\n\x10\x46rameworkVariant\x12\x10\n\x0cGP_CORRECTED\x10\x00\x12\x1b\n\x17GP_CORRECTED_INPUT_ONLY\x10\x01\x12\x1c\n\x18GP_CORRECTED_OUTPUT_ONLY\x10\x02\x12\x06\n\x02GP\x10\x03\x12\x11\n\rGP_INPUT_ONLY\x10\x04\x12\x12\n\x0eGP_OUTPUT_ONLY\x10\x05*1\n\nKernelType\x12\x07\n\x03RBF\x10\x00\x12\x08\n\x04RBFY\x10\x01\x12\x10\n\x0cRBF_PLUS_RBF\x10\x02\x32r\n\nRioService\x12.\n\x05Train\x12\x11.rio.TrainRequest\x1a\x10.rio.TrainResult\"\x00\x12\x34\n\x07Predict\x12\x13.rio.PredictRequest\x1a\x12.rio.PredictResult\"\x00\x62\x06proto3')
)

_FRAMEWORKVARIANT = _descriptor.EnumDescriptor(
  name='FrameworkVariant',
  full_name='rio.FrameworkVariant',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='GP_CORRECTED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GP_CORRECTED_INPUT_ONLY', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GP_CORRECTED_OUTPUT_ONLY', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GP', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GP_INPUT_ONLY', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GP_OUTPUT_ONLY', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=601,
  serialized_end=743,
)
_sym_db.RegisterEnumDescriptor(_FRAMEWORKVARIANT)

FrameworkVariant = enum_type_wrapper.EnumTypeWrapper(_FRAMEWORKVARIANT)
_KERNELTYPE = _descriptor.EnumDescriptor(
  name='KernelType',
  full_name='rio.KernelType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RBF', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RBFY', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RBF_PLUS_RBF', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=745,
  serialized_end=794,
)
_sym_db.RegisterEnumDescriptor(_KERNELTYPE)

KernelType = enum_type_wrapper.EnumTypeWrapper(_KERNELTYPE)
GP_CORRECTED = 0
GP_CORRECTED_INPUT_ONLY = 1
GP_CORRECTED_OUTPUT_ONLY = 2
GP = 3
GP_INPUT_ONLY = 4
GP_OUTPUT_ONLY = 5
RBF = 0
RBFY = 1
RBF_PLUS_RBF = 2



_TRAINREQUEST = _descriptor.Descriptor(
  name='TrainRequest',
  full_name='rio.TrainRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='framework_variant', full_name='rio.TrainRequest.framework_variant', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_type', full_name='rio.TrainRequest.kernel_type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='normed_train_data', full_name='rio.TrainRequest.normed_train_data', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_labels', full_name='rio.TrainRequest.train_labels', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_predictions', full_name='rio.TrainRequest.train_predictions', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_svgp_inducing_points', full_name='rio.TrainRequest.num_svgp_inducing_points', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_iterations_optimizer', full_name='rio.TrainRequest.max_iterations_optimizer', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_ard', full_name='rio.TrainRequest.use_ard', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19,
  serialized_end=282,
)


_PREDICTREQUEST = _descriptor.Descriptor(
  name='PredictRequest',
  full_name='rio.PredictRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='normed_test_data', full_name='rio.PredictRequest.normed_test_data', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_predictions', full_name='rio.PredictRequest.test_predictions', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model', full_name='rio.PredictRequest.model', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='framework_variant', full_name='rio.PredictRequest.framework_variant', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kernel_type', full_name='rio.PredictRequest.kernel_type', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=285,
  serialized_end=456,
)


_TRAINRESULT = _descriptor.Descriptor(
  name='TrainResult',
  full_name='rio.TrainResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='computation_time_seconds', full_name='rio.TrainResult.computation_time_seconds', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model', full_name='rio.TrainResult.model', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=458,
  serialized_end=520,
)


_PREDICTRESULT = _descriptor.Descriptor(
  name='PredictResult',
  full_name='rio.PredictResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='mean', full_name='rio.PredictResult.mean', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='var', full_name='rio.PredictResult.var', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='computation_time_seconds', full_name='rio.PredictResult.computation_time_seconds', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=522,
  serialized_end=598,
)

_TRAINREQUEST.fields_by_name['framework_variant'].enum_type = _FRAMEWORKVARIANT
_TRAINREQUEST.fields_by_name['kernel_type'].enum_type = _KERNELTYPE
_PREDICTREQUEST.fields_by_name['framework_variant'].enum_type = _FRAMEWORKVARIANT
_PREDICTREQUEST.fields_by_name['kernel_type'].enum_type = _KERNELTYPE
DESCRIPTOR.message_types_by_name['TrainRequest'] = _TRAINREQUEST
DESCRIPTOR.message_types_by_name['PredictRequest'] = _PREDICTREQUEST
DESCRIPTOR.message_types_by_name['TrainResult'] = _TRAINRESULT
DESCRIPTOR.message_types_by_name['PredictResult'] = _PREDICTRESULT
DESCRIPTOR.enum_types_by_name['FrameworkVariant'] = _FRAMEWORKVARIANT
DESCRIPTOR.enum_types_by_name['KernelType'] = _KERNELTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrainRequest = _reflection.GeneratedProtocolMessageType('TrainRequest', (_message.Message,), dict(
  DESCRIPTOR = _TRAINREQUEST,
  __module__ = 'rio_pb2'
  # @@protoc_insertion_point(class_scope:rio.TrainRequest)
  ))
_sym_db.RegisterMessage(TrainRequest)

PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTREQUEST,
  __module__ = 'rio_pb2'
  # @@protoc_insertion_point(class_scope:rio.PredictRequest)
  ))
_sym_db.RegisterMessage(PredictRequest)

TrainResult = _reflection.GeneratedProtocolMessageType('TrainResult', (_message.Message,), dict(
  DESCRIPTOR = _TRAINRESULT,
  __module__ = 'rio_pb2'
  # @@protoc_insertion_point(class_scope:rio.TrainResult)
  ))
_sym_db.RegisterMessage(TrainResult)

PredictResult = _reflection.GeneratedProtocolMessageType('PredictResult', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTRESULT,
  __module__ = 'rio_pb2'
  # @@protoc_insertion_point(class_scope:rio.PredictResult)
  ))
_sym_db.RegisterMessage(PredictResult)



_RIOSERVICE = _descriptor.ServiceDescriptor(
  name='RioService',
  full_name='rio.RioService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=796,
  serialized_end=910,
  methods=[
  _descriptor.MethodDescriptor(
    name='Train',
    full_name='rio.RioService.Train',
    index=0,
    containing_service=None,
    input_type=_TRAINREQUEST,
    output_type=_TRAINRESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='rio.RioService.Predict',
    index=1,
    containing_service=None,
    input_type=_PREDICTREQUEST,
    output_type=_PREDICTRESULT,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_RIOSERVICE)

DESCRIPTOR.services_by_name['RioService'] = _RIOSERVICE

# @@protoc_insertion_point(module_scope)
