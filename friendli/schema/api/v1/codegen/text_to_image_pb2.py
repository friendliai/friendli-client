# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: text_to_image.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from __future__ import annotations

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC, 5, 27, 2, "", "text_to_image.proto"
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x13text_to_image.proto"\xd8\x02\n\x14V1TextToImageRequest\x12\x0e\n\x06prompt\x18\x01 \x01(\t\x12\x1c\n\x0fnegative_prompt\x18\x02 \x01(\tH\x00\x88\x01\x01\x12\x18\n\x0bnum_outputs\x18\x03 \x01(\x05H\x01\x88\x01\x01\x12 \n\x13num_inference_steps\x18\x04 \x01(\x05H\x02\x88\x01\x01\x12\x1b\n\x0eguidance_scale\x18\x05 \x01(\x02H\x03\x88\x01\x01\x12\x11\n\x04seed\x18\x06 \x01(\x05H\x04\x88\x01\x01\x12\x1c\n\x0fresponse_format\x18\x07 \x01(\tH\x05\x88\x01\x01\x12\x12\n\x05model\x18\x08 \x01(\tH\x06\x88\x01\x01\x42\x12\n\x10_negative_promptB\x0e\n\x0c_num_outputsB\x16\n\x14_num_inference_stepsB\x11\n\x0f_guidance_scaleB\x07\n\x05_seedB\x12\n\x10_response_formatB\x08\n\x06_modelb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "text_to_image_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_V1TEXTTOIMAGEREQUEST"]._serialized_start = 24
    _globals["_V1TEXTTOIMAGEREQUEST"]._serialized_end = 368
# @@protoc_insertion_point(module_scope)
