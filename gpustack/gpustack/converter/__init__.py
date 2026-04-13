"""Format conversion engine for Anthropic <-> OpenAI API compatibility.

This module provides format conversion between Anthropic Messages API and
OpenAI Chat Completions API, enabling GPUStack to route requests to providers
regardless of their native protocol format.
"""

from gpustack.converter.converters import (
    FINISH_REASON_MAP,
    REVERSE_FINISH_REASON_MAP,
    convert_content_block,
    convert_error,
    convert_messages,
    convert_request,
    convert_response,
    convert_tools,
    generate_msg_id,
    reverse_convert_content_block,
    reverse_convert_message,
    reverse_convert_request,
    reverse_convert_response,
    reverse_convert_tools,
    sse_event,
    build_message_start_event,
    build_content_block_start_event,
    build_content_block_delta_event,
    build_content_block_stop_event,
    build_message_delta_event,
    build_message_stop_event,
)
from gpustack.converter.auth import AuthAdapter
from gpustack.converter.urls import build_openai_url, dedupe_base_url_path
from gpustack.converter.router import FormatRouter

__all__ = [
    # Constants
    "FINISH_REASON_MAP",
    "REVERSE_FINISH_REASON_MAP",
    # Request conversion
    "convert_request",
    "reverse_convert_request",
    "convert_messages",
    "reverse_convert_message",
    "convert_content_block",
    "reverse_convert_content_block",
    "convert_tools",
    "reverse_convert_tools",
    # Response conversion
    "convert_response",
    "reverse_convert_response",
    "convert_error",
    # SSE event builders
    "sse_event",
    "generate_msg_id",
    "build_message_start_event",
    "build_content_block_start_event",
    "build_content_block_delta_event",
    "build_content_block_stop_event",
    "build_message_delta_event",
    "build_message_stop_event",
    # Auth & URLs
    "AuthAdapter",
    "build_openai_url",
    "dedupe_base_url_path",
    # Router
    "FormatRouter",
]
