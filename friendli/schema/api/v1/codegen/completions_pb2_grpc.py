# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
from __future__ import annotations

import warnings

import grpc

from friendli.schema.api.v1.codegen import completions_pb2 as completions__pb2

GRPC_GENERATED_VERSION = "1.66.2"
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower

    _version_not_supported = first_version_is_lower(
        GRPC_VERSION, GRPC_GENERATED_VERSION
    )
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f"The grpc package installed is at version {GRPC_VERSION},"
        + f" but the generated code in completions_pb2_grpc.py depends on"
        + f" grpcio>={GRPC_GENERATED_VERSION}."
        + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
        + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
    )


class TextGenerationServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Generate = channel.unary_stream(
            "/orca.TextGenerationService/Generate",
            request_serializer=completions__pb2.V1CompletionsRequest.SerializeToString,
            response_deserializer=completions__pb2.V1CompletionsResponse.FromString,
            _registered_method=True,
        )


class TextGenerationServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Generate(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_TextGenerationServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Generate": grpc.unary_stream_rpc_method_handler(
            servicer.Generate,
            request_deserializer=completions__pb2.V1CompletionsRequest.FromString,
            response_serializer=completions__pb2.V1CompletionsResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "orca.TextGenerationService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers(
        "orca.TextGenerationService", rpc_method_handlers
    )


# This class is part of an EXPERIMENTAL API.
class TextGenerationService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Generate(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/orca.TextGenerationService/Generate",
            completions__pb2.V1CompletionsRequest.SerializeToString,
            completions__pb2.V1CompletionsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )
