# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: friendli/schema/api/v1/codegen/chat_completions.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from __future__ import annotations

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n5friendli/schema/api/v1/codegen/chat_completions.proto"\xf1\x03\n\x18V1ChatCompletionsRequest\x12\x33\n\x08messages\x18\x01 \x03(\x0b\x32!.V1ChatCompletionsRequest.Message\x12\x12\n\x05model\x18\x02 \x01(\tH\x00\x88\x01\x01\x12\x1e\n\x11\x66requency_penalty\x18\x03 \x01(\x02H\x01\x88\x01\x01\x12\x17\n\nmax_tokens\x18\x05 \x01(\x05H\x02\x88\x01\x01\x12\x0e\n\x01n\x18\x06 \x01(\x05H\x03\x88\x01\x01\x12\x1d\n\x10presence_penalty\x18\x07 \x01(\x02H\x04\x88\x01\x01\x12\x0c\n\x04stop\x18\x08 \x03(\t\x12\x13\n\x06stream\x18\t \x01(\x08H\x05\x88\x01\x01\x12\x18\n\x0btemperature\x18\n \x01(\x02H\x06\x88\x01\x01\x12\x12\n\x05top_p\x18\x0b \x01(\x02H\x07\x88\x01\x01\x12!\n\x14timeout_microseconds\x18\x1e \x01(\x05H\x08\x88\x01\x01\x1a(\n\x07Message\x12\x0f\n\x07\x63ontent\x18\x01 \x01(\t\x12\x0c\n\x04role\x18\x02 \x01(\tB\x08\n\x06_modelB\x14\n\x12_frequency_penaltyB\r\n\x0b_max_tokensB\x04\n\x02_nB\x13\n\x11_presence_penaltyB\t\n\x07_streamB\x0e\n\x0c_temperatureB\x08\n\x06_top_pB\x17\n\x15_timeout_microsecondsb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "friendli.schema.api.v1.codegen.chat_completions_pb2", _globals
)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_V1CHATCOMPLETIONSREQUEST"]._serialized_start = 58
    _globals["_V1CHATCOMPLETIONSREQUEST"]._serialized_end = 555
    _globals["_V1CHATCOMPLETIONSREQUEST_MESSAGE"]._serialized_start = 379
    _globals["_V1CHATCOMPLETIONSREQUEST_MESSAGE"]._serialized_end = 419
# @@protoc_insertion_point(module_scope)
