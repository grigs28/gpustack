"""Unit tests for the Anthropic <-> OpenAI format converter module."""

import json
import pytest

from gpustack.converter.converters import (
    FINISH_REASON_MAP,
    REVERSE_FINISH_REASON_MAP,
    convert_content_block,
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
)
from gpustack.converter.router import FormatRouter
from gpustack.converter.auth import AuthAdapter


# --- generate_msg_id ---

def test_generate_msg_id():
    msg_id = generate_msg_id()
    assert msg_id.startswith("msg_")
    assert len(msg_id) == 28  # "msg_" + 24 hex chars


# --- Request Conversion: Anthropic -> OpenAI ---

class TestConvertRequest:
    def test_basic_request(self):
        anthropic_req = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": False,
        }
        result = convert_request(anthropic_req)
        assert result["model"] == "claude-3-opus"
        assert result["max_tokens"] == 1024
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["stream"] is False

    def test_system_prompt_string(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "system": "You are helpful",
        }
        result = convert_request(req)
        assert result["messages"][0] == {"role": "system", "content": "You are helpful"}
        assert result["messages"][1] == {"role": "user", "content": "hi"}

    def test_system_prompt_blocks(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "system": [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
        }
        result = convert_request(req)
        assert result["messages"][0]["role"] == "system"
        assert "Part 1" in result["messages"][0]["content"]
        assert "Part 2" in result["messages"][0]["content"]

    def test_stop_sequences(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["stop1", "stop2"],
        }
        result = convert_request(req)
        assert result["stop"] == ["stop1", "stop2"]

    def test_stream_adds_options(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        result = convert_request(req)
        assert result["stream"] is True
        assert result["stream_options"] == {"include_usage": True}

    def test_temperature_top_p(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        result = convert_request(req)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

    def test_model_map(self):
        req = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = convert_request(req, model_map={"claude-3-opus": "gpt-4"})
        assert result["model"] == "gpt-4"

    def test_tool_choice_auto(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "auto"},
        }
        result = convert_request(req)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "any"},
        }
        result = convert_request(req)
        assert result["tool_choice"] == "required"

    def test_tool_choice_named(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "tool", "name": "my_func"},
        }
        result = convert_request(req)
        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "my_func"


class TestConvertMessages:
    def test_string_content(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = convert_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_tool_result(self):
        msgs = [{
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "res"}],
        }]
        result = convert_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "t1"
        assert result[0]["content"] == "res"

    def test_assistant_with_tool_use(self):
        msgs = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "t1", "name": "func", "input": {"a": 1}},
            ],
        }]
        result = convert_messages(msgs)
        assert result[0]["content"] == "Let me check"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "func"

    def test_assistant_thinking_skipped(self):
        msgs = [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "answer"},
            ],
        }]
        result = convert_messages(msgs)
        assert result[0]["content"] == "answer"

    def test_image_block(self):
        msgs = [{
            "role": "user",
            "content": [{
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
            }],
        }]
        result = convert_messages(msgs)
        assert result[0]["content"][0]["type"] == "image_url"
        assert result[0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


class TestConvertTools:
    def test_basic(self):
        tools = [{"name": "func", "description": "desc", "input_schema": {"type": "object"}}]
        result = convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "func"
        assert result[0]["function"]["parameters"] == {"type": "object"}


# --- Response Conversion: OpenAI -> Anthropic ---

class TestConvertResponse:
    def test_basic_text(self):
        openai_resp = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = convert_response(openai_resp, "claude-3")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10

    def test_with_reasoning(self):
        openai_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "reasoning_content": "Thinking...",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = convert_response(openai_resp, "m")
        types = [b["type"] for b in result["content"]]
        assert "thinking" in types
        assert "text" in types

    def test_with_tool_calls(self):
        openai_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "func", "arguments": '{"a": 1}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = convert_response(openai_resp, "m")
        assert result["stop_reason"] == "tool_use"
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "func"

    def test_finish_reason_mapping(self):
        for openai_reason, anthropic_reason in FINISH_REASON_MAP.items():
            resp = {
                "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": openai_reason}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            result = convert_response(resp, "m")
            assert result["stop_reason"] == anthropic_reason


# --- Reverse Request Conversion: OpenAI -> Anthropic ---

class TestReverseConvertRequest:
    def test_basic(self):
        openai_req = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 1024,
        }
        result = reverse_convert_request(openai_req)
        assert result["model"] == "gpt-4"
        assert result["system"] == "You are helpful"
        assert result["max_tokens"] == 1024
        assert len(result["messages"]) == 1

    def test_default_max_tokens(self):
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        result = reverse_convert_request(req)
        assert result["max_tokens"] == 4096

    def test_stop_string_to_array(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "END",
        }
        result = reverse_convert_request(req)
        assert result["stop_sequences"] == ["END"]

    def test_stop_list(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["END1", "END2"],
        }
        result = reverse_convert_request(req)
        assert result["stop_sequences"] == ["END1", "END2"]

    def test_tools(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "func", "description": "desc", "parameters": {"type": "object"}},
            }],
        }
        result = reverse_convert_request(req)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "func"
        assert "input_schema" in result["tools"][0]

    def test_tool_choice_auto(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "auto",
        }
        result = reverse_convert_request(req)
        assert result["tool_choice"] == {"type": "auto"}

    def test_tool_choice_required(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
        }
        result = reverse_convert_request(req)
        assert result["tool_choice"] == {"type": "any"}


class TestReverseConvertMessage:
    def test_tool_message(self):
        msg = {"role": "tool", "tool_call_id": "c1", "content": "result"}
        result = reverse_convert_message(msg)
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "c1"

    def test_assistant_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "text",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": '{"a":1}'},
            }],
        }
        result = reverse_convert_message(msg)
        blocks = result["content"]
        types = [b["type"] for b in blocks]
        assert "text" in types
        assert "tool_use" in types

    def test_image_url(self):
        msg = {
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
            }],
        }
        result = reverse_convert_message(msg)
        block = result["content"][0]
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"


class TestReverseConvertResponse:
    def test_basic(self):
        resp = {
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-3",
        }
        result = reverse_convert_response(resp)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 15

    def test_with_thinking(self):
        resp = {
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = reverse_convert_response(resp)
        assert result["choices"][0]["message"]["reasoning_content"] == "hmm"

    def test_with_tool_use(self):
        resp = {
            "content": [{
                "type": "tool_use",
                "id": "t1",
                "name": "func",
                "input": {"a": 1},
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = reverse_convert_response(resp)
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1


# --- FormatRouter ---

class TestFormatRouter:
    def test_detect_anthropic_path(self):
        assert FormatRouter.detect_request_format("/v1/messages") == "anthropic"

    def test_detect_openai_path(self):
        assert FormatRouter.detect_request_format("/v1/chat/completions") == "openai"

    def test_detect_from_body_system(self):
        body = {"system": "You are helpful", "messages": [{"role": "user", "content": "hi"}]}
        assert FormatRouter.detect_request_format("/other", body) == "anthropic"

    def test_detect_from_body_tool_result(self):
        body = {
            "messages": [{
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}],
            }]
        }
        assert FormatRouter.detect_request_format("/other", body) == "anthropic"

    def test_detect_default_openai(self):
        assert FormatRouter.detect_request_format("/other", {}) == "openai"

    def test_passthrough(self):
        assert FormatRouter.should_passthrough("openai", ["openai"]) is True
        assert FormatRouter.should_passthrough("anthropic", ["openai"]) is False

    def test_get_target_format_direct(self):
        assert FormatRouter.get_target_format("openai", ["openai"]) == "openai"

    def test_get_target_format_convert(self):
        assert FormatRouter.get_target_format("anthropic", ["openai"]) == "openai"

    def test_route_anthropic_to_openai(self):
        decision = FormatRouter.route(
            path="/v1/messages",
            body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            base_url="https://api.openai.com",
            api_key="sk-test",
            provider_supported_formats=["openai"],
        )
        assert decision.action == "convert"
        assert decision.source_format == "anthropic"
        assert decision.target_format == "openai"
        assert "/v1/chat/completions" in decision.target_url
        assert decision.converted_body is not None

    def test_route_openai_passthrough(self):
        decision = FormatRouter.route(
            path="/v1/chat/completions",
            body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            base_url="https://api.openai.com",
            api_key="sk-test",
            provider_supported_formats=["openai"],
        )
        assert decision.action == "passthrough"
        assert decision.source_format == "openai"

    def test_build_target_url_with_anthropic_base(self):
        """When base_url_anthropic is set, Anthropic requests use it."""
        url = FormatRouter.build_target_url(
            base_url="https://api.openai.com",
            target_format="anthropic",
            base_url_anthropic="https://api.anthropic.com",
        )
        assert url == "https://api.anthropic.com/v1/messages"

    def test_build_target_url_without_anthropic_base(self):
        """When base_url_anthropic is not set, fallback to base_url."""
        url = FormatRouter.build_target_url(
            base_url="https://api.example.com",
            target_format="anthropic",
        )
        assert url == "https://api.example.com/v1/messages"

    def test_route_dual_url_anthropic_passthrough(self):
        """Dual-format provider with separate Anthropic URL."""
        decision = FormatRouter.route(
            path="/v1/messages",
            body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            base_url="https://api.openai.com",
            api_key="sk-test",
            provider_supported_formats=["openai", "anthropic"],
            base_url_anthropic="https://api.anthropic.com",
        )
        assert decision.action == "passthrough"
        assert decision.target_format == "anthropic"
        assert decision.target_url.startswith("https://api.anthropic.com")

    def test_route_dual_url_openai_passthrough(self):
        """Dual-format provider: OpenAI request uses primary base_url."""
        decision = FormatRouter.route(
            path="/v1/chat/completions",
            body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            base_url="https://api.openai.com",
            api_key="sk-test",
            provider_supported_formats=["openai", "anthropic"],
            base_url_anthropic="https://api.anthropic.com",
        )
        assert decision.action == "passthrough"
        assert decision.target_format == "openai"
        assert decision.target_url.startswith("https://api.openai.com")


# --- AuthAdapter ---

class TestAuthAdapter:
    def test_openai_headers(self):
        headers = AuthAdapter.build_headers("sk-test", "openai")
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["content-type"] == "application/json"

    def test_anthropic_auto_headers(self):
        headers = AuthAdapter.build_headers("sk-test", "anthropic", "auto")
        assert headers["x-api-key"] == "sk-test"
        assert headers["Authorization"] == "Bearer sk-test"
        assert "anthropic-version" in headers

    def test_anthropic_bearer_headers(self):
        headers = AuthAdapter.build_headers("sk-test", "anthropic", "bearer")
        assert headers["Authorization"] == "Bearer sk-test"
        assert "x-api-key" not in headers

    def test_anthropic_x_api_key_headers(self):
        headers = AuthAdapter.build_headers("sk-test", "anthropic", "x-api-key")
        assert headers["x-api-key"] == "sk-test"
        assert "Authorization" not in headers

    def test_determine_auth_style_default(self):
        assert AuthAdapter.determine_auth_style() == "bearer"

    def test_determine_auth_style_claude(self):
        assert AuthAdapter.determine_auth_style(provider_type="claude") == "x-api-key"

    def test_determine_auth_style_explicit(self):
        assert AuthAdapter.determine_auth_style(auth_style="x-api-key") == "x-api-key"
