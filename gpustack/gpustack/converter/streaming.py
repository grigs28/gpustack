"""Streaming SSE conversion between OpenAI and Anthropic formats.

Ported from cc_proxy/client.py streaming logic and adapted for GPUStack.
"""

import json
import logging
import uuid
from typing import AsyncGenerator

from gpustack.converter.converters import (
    FINISH_REASON_MAP,
    build_content_block_delta_event,
    build_content_block_start_event,
    build_content_block_stop_event,
    build_message_delta_event,
    build_message_start_event,
    build_message_stop_event,
    generate_msg_id,
    sse_event,
)

logger = logging.getLogger(__name__)


async def convert_openai_stream_to_anthropic(
    response,
    model: str,
) -> AsyncGenerator[bytes, None]:
    """Convert an OpenAI SSE stream into Anthropic SSE events.

    Reads OpenAI streaming chunks and yields Anthropic-format SSE events.

    Args:
        response: An aiohttp/httpx streaming response context.
        model: The model name to use in Anthropic events.

    Yields:
        bytes: Anthropic-format SSE event bytes.
    """
    msg_id = generate_msg_id()
    yield build_message_start_event(model=model, msg_id=msg_id).encode()

    block_index = 0
    current_type = None
    tc_states: dict[int, dict] = {}
    finish = "end_turn"
    out_tokens = 0

    async for line in _iter_sse_lines(response):
        if not line.startswith("data: "):
            continue

        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            # Usage-only chunk
            u = chunk.get("usage")
            if u:
                out_tokens = u.get("completion_tokens", 0)
            continue

        ch = choices[0]
        delta = ch.get("delta", {})

        if ch.get("finish_reason"):
            finish = FINISH_REASON_MAP.get(ch["finish_reason"], "end_turn")

        u = chunk.get("usage")
        if u:
            out_tokens = u.get("completion_tokens", 0)

        # thinking / reasoning
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if current_type != "thinking":
                if current_type is not None:
                    yield build_content_block_stop_event(block_index).encode()
                    block_index += 1
                yield build_content_block_start_event(
                    block_index, "thinking"
                ).encode()
                current_type = "thinking"
            yield build_content_block_delta_event(
                block_index, "thinking_delta", text=reasoning
            ).encode()
            continue

        # text content
        text = delta.get("content")
        if text:
            if current_type != "text":
                if current_type is not None:
                    yield build_content_block_stop_event(block_index).encode()
                    block_index += 1
                yield build_content_block_start_event(
                    block_index, "text"
                ).encode()
                current_type = "text"
            yield build_content_block_delta_event(
                block_index, "text_delta", text=text
            ).encode()
            continue

        # tool calls
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                idx = tc.get("index", 0)
                if idx not in tc_states:
                    if current_type is not None:
                        yield build_content_block_stop_event(
                            block_index
                        ).encode()
                        block_index += 1
                    tid = tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
                    tn = tc.get("function", {}).get("name", "")
                    tc_states[idx] = {
                        "id": tid,
                        "name": tn,
                        "bi": block_index,
                    }
                    yield build_content_block_start_event(
                        block_index, "tool_use", tool_id=tid, tool_name=tn
                    ).encode()
                    current_type = "tool_use"
                ad = tc.get("function", {}).get("arguments", "")
                if ad:
                    yield build_content_block_delta_event(
                        tc_states[idx]["bi"],
                        "input_json_delta",
                        partial_json=ad,
                    ).encode()

    # Close any open blocks
    if current_type is not None:
        yield build_content_block_stop_event(block_index).encode()

    yield build_message_delta_event(finish, out_tokens).encode()
    yield build_message_stop_event().encode()


async def convert_anthropic_stream_to_openai(
    response,
    model: str,
) -> AsyncGenerator[bytes, None]:
    """Convert an Anthropic SSE stream into OpenAI SSE events.

    Reads Anthropic streaming events and yields OpenAI-format SSE chunks.

    Args:
        response: An aiohttp/httpx streaming response context.
        model: The model name to use in OpenAI chunks.

    Yields:
        bytes: OpenAI-format SSE event bytes.
    """
    msg_id = generate_msg_id()
    output_tokens = 0

    async for line in _iter_sse_lines(response):
        if not line.startswith("event: "):
            continue

        event_type = line[7:].strip()

        # Read the data line that follows
        data_line = None
        try:
            data_line = await _next_data_line(response)
        except StopAsyncIteration:
            break

        if data_line is None or not data_line.startswith("data: "):
            continue

        data_str = data_line[6:].strip()
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if event_type == "message_start":
            # Extract initial message data
            pass  # OpenAI doesn't have an equivalent initial event

        elif event_type == "content_block_start":
            block = data.get("content_block", {})
            btype = block.get("type")

            if btype == "text":
                chunk = _openai_chunk(msg_id, model, {"role": "assistant"})
                yield f"data: {json.dumps(chunk)}\n\n".encode()

            elif btype == "tool_use":
                chunk = _openai_chunk(
                    msg_id,
                    model,
                    None,
                    tool_calls=[{
                        "index": data.get("index", 0),
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": "",
                        },
                    }],
                )
                yield f"data: {json.dumps(chunk)}\n\n".encode()

        elif event_type == "content_block_delta":
            delta_data = data.get("delta", {})
            delta_type = delta_data.get("type")
            index = data.get("index", 0)

            if delta_type == "text_delta":
                text = delta_data.get("text", "")
                if text:
                    chunk = _openai_chunk(msg_id, model, text)
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

            elif delta_type == "thinking_delta":
                thinking = delta_data.get("thinking", "")
                if thinking:
                    chunk = _openai_chunk(
                        msg_id, model, None, reasoning=thinking
                    )
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

            elif delta_type == "input_json_delta":
                partial = delta_data.get("partial_json", "")
                if partial:
                    chunk = _openai_chunk(
                        msg_id,
                        model,
                        None,
                        tool_calls=[{
                            "index": index,
                            "function": {"arguments": partial},
                        }],
                    )
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            stop_reason = delta.get("stop_reason", "end_turn")
            finish_reason = "stop"

            # Map Anthropic stop reasons to OpenAI finish reasons
            from gpustack.converter.converters import REVERSE_FINISH_REASON_MAP
            finish_reason = REVERSE_FINISH_REASON_MAP.get(
                stop_reason, "stop"
            )

            usage = data.get("usage", {})
            output_tokens = usage.get("output_tokens", 0)

            chunk = _openai_chunk(
                msg_id, model, None, finish_reason=finish_reason
            )
            yield f"data: {json.dumps(chunk)}\n\n".encode()

        elif event_type == "message_stop":
            yield b"data: [DONE]\n\n"

    # Ensure we always send [DONE]
    yield b"data: [DONE]\n\n"


# --- Helper Functions ---

def _openai_chunk(
    msg_id: str,
    model: str,
    content: str | None = None,
    finish_reason: str | None = None,
    tool_calls: list | None = None,
    reasoning: str | None = None,
) -> dict:
    """Build an OpenAI streaming chunk."""
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if tool_calls:
        delta["tool_calls"] = tool_calls
    if not delta and finish_reason is None:
        delta["role"] = "assistant"

    chunk: dict = {
        "id": msg_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return chunk


async def _iter_sse_lines(response) -> AsyncGenerator[str, None]:
    """Iterate over SSE lines from a streaming response.

    Works with both aiohttp and httpx streaming responses.
    """
    # Try httpx first (has aiter_lines)
    if hasattr(response, "aiter_lines"):
        async for line in response.aiter_lines():
            yield line
    elif hasattr(response, "aiter_text"):
        async for text in response.aiter_text():
            for line in text.split("\n"):
                if line.strip():
                    yield line
    elif hasattr(response, "content"):
        # aiohttp: iterate by chunks
        buffer = b""
        async for data in response.content.iter_chunked(4096):
            buffer += data
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if line:
                    yield line
        if buffer.strip():
            yield buffer.decode("utf-8", errors="replace").strip()


async def _next_data_line(response) -> str | None:
    """Get the next data line from an SSE stream."""
    async for line in _iter_sse_lines(response):
        return line
    return None
