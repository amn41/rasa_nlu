# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: collector.proto
"""Generated protocol buffer code."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import model_pb2 as model__pb2
from gogoproto import gogo_pb2 as gogoproto_dot_gogo__pb2
from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
    name="collector.proto",
    package="jaeger.api_v2",
    syntax="proto3",
    serialized_options=b"\n\027io.jaegertracing.api_v2Z\006api_v2\310\342\036\001\320\342\036\001\340\342\036\001",
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x0f\x63ollector.proto\x12\rjaeger.api_v2\x1a\x0bmodel.proto\x1a\x14gogoproto/gogo.proto\x1a\x1cgoogle/api/annotations.proto"=\n\x10PostSpansRequest\x12)\n\x05\x62\x61tch\x18\x01 \x01(\x0b\x32\x14.jaeger.api_v2.BatchB\x04\xc8\xde\x1f\x00"\x13\n\x11PostSpansResponse2|\n\x10\x43ollectorService\x12h\n\tPostSpans\x12\x1f.jaeger.api_v2.PostSpansRequest\x1a .jaeger.api_v2.PostSpansResponse"\x18\x82\xd3\xe4\x93\x02\x12"\r/api/v2/spans:\x01*B-\n\x17io.jaegertracing.api_v2Z\x06\x61pi_v2\xc8\xe2\x1e\x01\xd0\xe2\x1e\x01\xe0\xe2\x1e\x01\x62\x06proto3',
    dependencies=[
        model__pb2.DESCRIPTOR,
        gogoproto_dot_gogo__pb2.DESCRIPTOR,
        google_dot_api_dot_annotations__pb2.DESCRIPTOR,
    ],
)


_POSTSPANSREQUEST = _descriptor.Descriptor(
    name="PostSpansRequest",
    full_name="jaeger.api_v2.PostSpansRequest",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="batch",
            full_name="jaeger.api_v2.PostSpansRequest.batch",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=b"\310\336\037\000",
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=99,
    serialized_end=160,
)


_POSTSPANSRESPONSE = _descriptor.Descriptor(
    name="PostSpansResponse",
    full_name="jaeger.api_v2.PostSpansResponse",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=162,
    serialized_end=181,
)

_POSTSPANSREQUEST.fields_by_name["batch"].message_type = model__pb2._BATCH
DESCRIPTOR.message_types_by_name["PostSpansRequest"] = _POSTSPANSREQUEST
DESCRIPTOR.message_types_by_name["PostSpansResponse"] = _POSTSPANSRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PostSpansRequest = _reflection.GeneratedProtocolMessageType(
    "PostSpansRequest",
    (_message.Message,),
    {
        "DESCRIPTOR": _POSTSPANSREQUEST,
        "__module__": "collector_pb2",
        # @@protoc_insertion_point(class_scope:jaeger.api_v2.PostSpansRequest)
    },
)
_sym_db.RegisterMessage(PostSpansRequest)

PostSpansResponse = _reflection.GeneratedProtocolMessageType(
    "PostSpansResponse",
    (_message.Message,),
    {
        "DESCRIPTOR": _POSTSPANSRESPONSE,
        "__module__": "collector_pb2",
        # @@protoc_insertion_point(class_scope:jaeger.api_v2.PostSpansResponse)
    },
)
_sym_db.RegisterMessage(PostSpansResponse)


DESCRIPTOR._options = None
_POSTSPANSREQUEST.fields_by_name["batch"]._options = None

_COLLECTORSERVICE = _descriptor.ServiceDescriptor(
    name="CollectorService",
    full_name="jaeger.api_v2.CollectorService",
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=183,
    serialized_end=307,
    methods=[
        _descriptor.MethodDescriptor(
            name="PostSpans",
            full_name="jaeger.api_v2.CollectorService.PostSpans",
            index=0,
            containing_service=None,
            input_type=_POSTSPANSREQUEST,
            output_type=_POSTSPANSRESPONSE,
            serialized_options=b'\202\323\344\223\002\022"\r/api/v2/spans:\001*',
            create_key=_descriptor._internal_create_key,
        ),
    ],
)
_sym_db.RegisterServiceDescriptor(_COLLECTORSERVICE)

DESCRIPTOR.services_by_name["CollectorService"] = _COLLECTORSERVICE

# @@protoc_insertion_point(module_scope)
