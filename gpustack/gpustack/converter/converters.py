"""Format conversion functions for Anthropic <-> OpenAI API compatibility.

Ported from cc_proxy/converter.py and adapted for GPUStack architecture.
"""

import json
import uuid
from typing import Any


# --- Constants ---

FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}

REVERSE_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


# --- Utility Functions ---

def generate_msg_id() -> str:
    """Generate a unique message ID in Anthropic format."""
    return f"msg_{uuid.uuid4().hex[:24]}"


# --- Request Conversion (Anthropic -> OpenAI) ---

def convert_content_block(block: dict) -> dict:
    """Convert a single Anthropic content block to OpenAI format."""
    block_type = block.get("type")

    if block_type == "text":
        return {"type": "text", "text": block["text"]}

    if block_type == "image":
        source = block["source"]
        media_type = source["media_type"]
        data = source["data"]
        data_url = f"data:{media_type};base64,{data}"
        return {
            "type": "image_url",
            "image_url": {"url": data_url},
        }

    # Unknown block type: pass through
    return block


def convert_messages(messages: list) -> list:
    """Convert Anthropic messages array to OpenAI format."""
    result = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        # Simple string content
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        # Content is a list of blocks
        if isinstance(content, list):
            # --- User message: check for tool_result blocks ---
            if role == "user":
                has_tool_result = any(
                    b.get("type") == "tool_result" for b in content
                )
                if has_tool_result:
                    for block in content:
                        if block.get("type") == "tool_result":
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, list):
                                parts = []
                                for sub in tool_content:
                                    if sub.get("type") == "text":
                                        parts.append(sub["text"])
                                tool_content = "\n".join(parts)
                            result.append({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": tool_content,
                            })
                        else:
                            converted = convert_content_block(block)
                            result.append({
                                "role": "user",
                                "content": [converted],
                            })
                else:
                    converted_blocks = [
                        convert_content_block(b) for b in content
                    ]
                    result.append({
                        "role": "user",
                        "content": converted_blocks,
                    })
                continue

            # --- Assistant message: handle tool_use and thinking blocks ---
            if role == "assistant":
                text_parts = []
                tool_calls = []

                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block["text"])
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"]),
                            },
                        })
                    elif btype == "thinking":
                        pass  # Skip thinking blocks

                out_msg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    out_msg["content"] = (
                        "\n".join(text_parts)
                        if len(text_parts) > 1
                        else text_parts[0]
                    )
                else:
                    out_msg["content"] = None
                if tool_calls:
                    out_msg["tool_calls"] = tool_calls

                result.append(out_msg)
                continue

            # Other roles with list content
            converted_blocks = [convert_content_block(b) for b in content]
            result.append({"role": role, "content": converted_blocks})

    return result


def convert_tools(tools: list) -> list:
    """Convert Anthropic tool definitions to OpenAI format."""
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return result


def convert_request(anthropic_req: dict, model_map: dict | None = None) -> dict:
    """Convert an Anthropic Messages API request to OpenAI Chat Completions format."""
    if model_map is None:
        model_map = {}

    openai_req: dict[str, Any] = {}

    # Model mapping
    raw_model = anthropic_req["model"]
    openai_req["model"] = model_map.get(raw_model, raw_model)

    # max_tokens
    if "max_tokens" in anthropic_req:
        openai_req["max_tokens"] = anthropic_req["max_tokens"]

    # Build messages list
    converted_messages = []

    # System prompt -> first system message
    system = anthropic_req.get("system")
    if system is not None:
        if isinstance(system, str):
            converted_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
            converted_messages.append({
                "role": "system",
                "content": "\n".join(parts),
            })

    # Convert and append user/assistant messages
    converted_messages.extend(
        convert_messages(anthropic_req.get("messages", []))
    )
    openai_req["messages"] = converted_messages

    # Pass-through fields
    if "temperature" in anthropic_req:
        openai_req["temperature"] = anthropic_req["temperature"]
    if "top_p" in anthropic_req:
        openai_req["top_p"] = anthropic_req["top_p"]

    # stop_sequences -> stop
    if "stop_sequences" in anthropic_req:
        openai_req["stop"] = anthropic_req["stop_sequences"]

    # stream: add stream_options when True
    if "stream" in anthropic_req:
        openai_req["stream"] = anthropic_req["stream"]
        if anthropic_req["stream"]:
            openai_req["stream_options"] = {"include_usage": True}

    # tools
    if "tools" in anthropic_req:
        openai_req["tools"] = convert_tools(anthropic_req["tools"])

    # tool_choice
    if "tool_choice" in anthropic_req:
        tc = anthropic_req["tool_choice"]
        if isinstance(tc, dict):
            if tc.get("type") == "auto":
                openai_req["tool_choice"] = "auto"
            elif tc.get("type") == "any":
                openai_req["tool_choice"] = "required"
            elif tc.get("type") == "tool":
                openai_req["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc.get("name", "")},
                }
        elif isinstance(tc, str):
            openai_req["tool_choice"] = tc

    return openai_req


# --- Response Conversion (OpenAI -> Anthropic) ---

def convert_response(openai_resp: dict, model: str) -> dict:
    """Convert OpenAI Chat Completions response to Anthropic Messages format."""
    choice = openai_resp["choices"][0]
    message = choice["message"]
    finish_reason = choice.get("finish_reason", "stop")
    usage = openai_resp.get("usage", {})

    content = []

    # Add reasoning/thinking block if present
    reasoning = message.get("reasoning_content")
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})

    # Add text content
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    # Add tool_use blocks
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        func = tc["function"]
        try:
            input_data = json.loads(func["arguments"])
        except (json.JSONDecodeError, TypeError):
            input_data = {}
        content.append({
            "type": "tool_use",
            "id": tc["id"],
            "name": func["name"],
            "input": input_data,
        })

    if not content:
        content.append({"type": "text", "text": ""})

    return {
        "id": generate_msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": FINISH_REASON_MAP.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# --- SSE Event Builders ---

def sse_event(event_type: str, data: dict) -> str:
    """Build a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def build_message_start_event(model: str, msg_id: str | None = None) -> str:
    """Build a message_start SSE event."""
    return sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id or generate_msg_id(),
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })


def build_content_block_start_event(
    index: int,
    block_type: str,
    tool_id: str | None = None,
    tool_name: str | None = None,
) -> str:
    """Build a content_block_start SSE event."""
    if block_type == "text":
        block = {"type": "text", "text": ""}
    elif block_type == "thinking":
        block = {"type": "thinking", "thinking": ""}
    elif block_type == "tool_use":
        block = {
            "type": "tool_use",
            "id": tool_id or "",
            "name": tool_name or "",
            "input": {},
        }
    else:
        block = {"type": block_type}
    return sse_event("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": block,
    })


def build_content_block_delta_event(
    index: int,
    delta_type: str,
    text: str = "",
    partial_json: str = "",
) -> str:
    """Build a content_block_delta SSE event."""
    if delta_type == "text_delta":
        delta = {"type": "text_delta", "text": text}
    elif delta_type == "thinking_delta":
        delta = {"type": "thinking_delta", "thinking": text}
    elif delta_type == "input_json_delta":
        delta = {"type": "input_json_delta", "partial_json": partial_json}
    else:
        delta = {"type": delta_type}
    return sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": delta,
    })


def build_content_block_stop_event(index: int) -> str:
    """Build a content_block_stop SSE event."""
    return sse_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": index},
    )


def build_message_delta_event(
    stop_reason: str, output_tokens: int = 0
) -> str:
    """Build a message_delta SSE event."""
    return sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })


def build_message_stop_event() -> str:
    """Build a message_stop SSE event."""
    return sse_event("message_stop", {"type": "message_stop"})


# --- Reverse Request Conversion (OpenAI -> Anthropic) ---

def reverse_convert_content_block(block: dict) -> dict:
    """Convert a single OpenAI content block to Anthropic format."""
    block_type = block.get("type")

    if block_type == "text":
        return {"type": "text", "text": block["text"]}

    if block_type == "image_url":
        url = block.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            parts = url[5:].split(";")
            media_type = parts[0] if parts else "image/png"
            data = parts[1].split(",")[1] if len(parts) > 1 else ""
        else:
            media_type = "image/png"
            data = url
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    return block


def reverse_convert_message(msg: dict) -> dict:
    """Convert an OpenAI message to Anthropic format."""
    role = msg.get("role")
    content = msg.get("content", "")

    if role == "tool":
        tool_content = content or ""
        if isinstance(tool_content, str):
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": tool_content,
                }],
            }
        else:
            parts = []
            for sub in tool_content:
                if sub.get("type") == "text":
                    parts.append(sub["text"])
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": "\n".join(parts),
                }],
            }

    if role == "assistant":
        text_parts = []
        image_blocks = []
        tool_uses = []

        if isinstance(content, str) and content:
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    img = reverse_convert_content_block(block)
                    if img:
                        image_blocks.append(img)

        # Handle tool_calls
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            arguments = func.get("arguments", "{}")
            try:
                input_data = (
                    json.loads(arguments)
                    if isinstance(arguments, str)
                    else arguments
                )
            except (json.JSONDecodeError, TypeError):
                input_data = {}
            tool_uses.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "input": input_data,
            })

        blocks = []
        if text_parts:
            blocks.append({
                "type": "text",
                "text": "\n".join(str(p) for p in text_parts),
            })
        blocks.extend(image_blocks)
        blocks.extend(tool_uses)

        if not blocks:
            blocks.append({"type": "text", "text": ""})

        return {
            "role": "assistant",
            "content": blocks,
        }

    # system, user, and other roles
    if isinstance(content, str):
        return {"role": role, "content": content}

    # Content is a list of blocks
    converted_blocks = [reverse_convert_content_block(b) for b in content]
    return {"role": role, "content": converted_blocks}


def reverse_convert_tools(tools: list) -> list:
    """Convert OpenAI tool definitions to Anthropic format."""
    result = []
    for tool in tools:
        func = tool.get("function", {})
        result.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        })
    return result


def reverse_convert_request(
    openai_req: dict, model_map: dict | None = None
) -> dict:
    """Convert an OpenAI Chat Completions request to Anthropic Messages format."""
    if model_map is None:
        model_map = {}

    anthropic_req: dict[str, Any] = {}

    # Model mapping
    raw_model = openai_req["model"]
    anthropic_req["model"] = model_map.get(raw_model, raw_model)

    # Build messages list
    converted_messages = []
    system_content = None
    other_messages = []

    for msg in openai_req.get("messages", []):
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
        else:
            other_messages.append(msg)

    # System prompt
    if system_content is not None:
        anthropic_req["system"] = system_content

    # Convert messages
    for msg in other_messages:
        converted = reverse_convert_message(msg)
        if isinstance(converted.get("content"), list):
            has_tool_result = any(
                b.get("type") == "tool_result"
                for b in converted["content"]
            )
            if has_tool_result and converted["role"] == "user":
                for block in converted["content"]:
                    converted_messages.append({
                        "role": "user",
                        "content": [block],
                    })
            else:
                converted_messages.append(converted)
        else:
            converted_messages.append(converted)

    anthropic_req["messages"] = converted_messages

    # max_tokens: default to 4096 if not specified (Anthropic requires it)
    if "max_tokens" in openai_req:
        anthropic_req["max_tokens"] = openai_req["max_tokens"]
    else:
        anthropic_req["max_tokens"] = 4096

    # Pass-through fields
    if "temperature" in openai_req:
        anthropic_req["temperature"] = openai_req["temperature"]
    if "top_p" in openai_req:
        anthropic_req["top_p"] = openai_req["top_p"]

    # stop -> stop_sequences
    if "stop" in openai_req:
        stop = openai_req["stop"]
        if isinstance(stop, str):
            anthropic_req["stop_sequences"] = [stop]
        elif isinstance(stop, list):
            anthropic_req["stop_sequences"] = stop

    # stream
    if "stream" in openai_req:
        anthropic_req["stream"] = openai_req["stream"]

    # tools
    if "tools" in openai_req:
        anthropic_req["tools"] = reverse_convert_tools(openai_req["tools"])

    # tool_choice
    if "tool_choice" in openai_req:
        tc = openai_req["tool_choice"]
        if tc == "auto":
            anthropic_req["tool_choice"] = {"type": "auto"}
        elif tc == "required":
            anthropic_req["tool_choice"] = {"type": "any"}
        elif isinstance(tc, dict) and tc.get("type") == "function":
            anthropic_req["tool_choice"] = {
                "type": "tool",
                "name": tc.get("function", {}).get("name", ""),
            }

    return anthropic_req


# --- Reverse Response Conversion (Anthropic -> OpenAI) ---

def reverse_convert_response(anthropic_resp: dict) -> dict:
    """Convert Anthropic Messages response to OpenAI Chat Completions format."""
    content = anthropic_resp.get("content", [])

    text_parts = []
    tool_calls = []
    reasoning = None

    for block in content:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            reasoning = block.get("thinking", "")
        elif btype == "tool_use":
            func = block.get("input", {})
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": (
                        json.dumps(func) if isinstance(func, dict) else func
                    ),
                },
            })

    text_content = "\n".join(text_parts) if text_parts else None
    stop_reason = anthropic_resp.get("stop_reason", "stop")
    finish_reason = REVERSE_FINISH_REASON_MAP.get(stop_reason, "stop")

    result: dict[str, Any] = {
        "id": anthropic_resp.get("id", generate_msg_id()),
        "object": "chat.completion",
        "created": 0,
        "model": anthropic_resp.get("model", ""),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text_content,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": anthropic_resp.get("usage", {}).get(
                "input_tokens", 0
            ),
            "completion_tokens": anthropic_resp.get("usage", {}).get(
                "output_tokens", 0
            ),
            "total_tokens": sum([
                anthropic_resp.get("usage", {}).get("input_tokens", 0),
                anthropic_resp.get("usage", {}).get("output_tokens", 0),
            ]),
        },
    }

    if reasoning:
        result["choices"][0]["message"]["reasoning_content"] = reasoning

    if tool_calls:
        result["choices"][0]["message"]["tool_calls"] = tool_calls

    return result


# --- Error Conversion ---

def convert_error(status_code: int, openai_error) -> tuple[int, dict]:
    """Convert OpenAI error response to Anthropic format."""
    if isinstance(openai_error, str):
        return status_code, {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": openai_error,
            },
        }
    error_body = openai_error.get("error", {})
    if isinstance(error_body, str):
        error_body = {"message": error_body}
    error_type = error_body.get("type", "api_error")
    type_map = {
        "invalid_request_error": "invalid_request_error",
        "authentication_error": "authentication_error",
        "rate_limit_error": "rate_limit_error",
        "not_found_error": "not_found_error",
    }
    return status_code, {
        "type": "error",
        "error": {
            "type": type_map.get(error_type, "api_error"),
            "message": error_body.get("message", "Unknown error"),
        },
    }
