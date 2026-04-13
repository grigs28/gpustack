"""Unified provider proxy with format conversion support.

Handles proxying requests to external providers with automatic format
conversion between Anthropic and OpenAI protocols.
"""

import json
import logging
from typing import AsyncGenerator, Optional

import aiohttp
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from gpustack.converter.auth import AuthAdapter
from gpustack.converter.converters import (
    convert_error,
    convert_response,
    generate_msg_id,
    reverse_convert_response,
)
from gpustack.converter.router import FormatRouter
from gpustack.converter.streaming import (
    convert_anthropic_stream_to_openai,
    convert_openai_stream_to_anthropic,
)
from gpustack.converter.urls import build_openai_url
from gpustack.schemas.model_provider import ModelProvider
from gpustack import envs

logger = logging.getLogger(__name__)

# Anthropic core fields for stripping when strip_fields is True
_ANTHROPIC_CORE_KEYS = {
    "model", "messages", "max_tokens", "stream", "stop_sequences",
    "temperature", "top_p", "top_k", "system", "tools", "tool_choice",
}

# Status codes eligible for retry
_RETRY_STATUSES = {404, 429, 500, 502, 503, 529}
_MAX_RETRIES = 3


def _clean_anthropic_body(body: dict) -> dict:
    """Strip non-standard fields from Anthropic request body."""
    return {k: v for k, v in body.items() if k in _ANTHROPIC_CORE_KEYS}


def _select_api_token(provider: ModelProvider, index: int = 0) -> str:
    """Select an API token from provider, with round-robin support."""
    tokens = provider.api_tokens or []
    if not tokens:
        raise ValueError(f"Provider {provider.name} has no API tokens configured")
    return tokens[index % len(tokens)]


async def proxy_to_provider(
    request: Request,
    provider: ModelProvider,
    model_name: str,
    request_format: str,
    body: dict,
    stream: bool,
    model_map: dict | None = None,
) -> Response:
    """Proxy a request to an external provider with format conversion as needed.

    Args:
        request: The incoming FastAPI request.
        provider: The target provider.
        model_name: The model name to use.
        request_format: Format of the incoming request ("anthropic" or "openai").
        body: The request body.
        stream: Whether this is a streaming request.
        model_map: Optional model name mapping.

    Returns:
        FastAPI Response (JSON or Streaming).
    """
    api_token = _select_api_token(provider)
    base_url = provider.config.get_base_url()
    if not base_url:
        return _format_error_response(
            request_format,
            500,
            f"Provider {provider.name} has no base URL configured",
        )

    # Get Anthropic-specific base URL if configured
    base_url_anthropic = None
    if hasattr(provider.config, "get_base_url"):
        base_url_anthropic = provider.config.get_base_url(fmt="anthropic")
        # If same as primary, treat as not set to avoid redundancy
        if base_url_anthropic == base_url:
            base_url_anthropic = None

    # Get provider format preferences
    supported_formats = getattr(provider, "supported_formats", None)
    auth_style = getattr(provider, "auth_style", None)
    auth_style_anthropic = getattr(provider.config, "auth_style_anthropic", None)
    strip_fields = getattr(provider, "strip_fields", False)

    # Make routing decision
    decision = FormatRouter.route(
        path=str(request.url.path),
        body=body,
        base_url=base_url,
        api_key=api_token,
        provider_config=provider.config,
        provider_supported_formats=supported_formats,
        auth_style=auth_style,
        auth_style_anthropic=auth_style_anthropic,
        model_map=model_map,
        base_url_anthropic=base_url_anthropic,
    )

    target_body = decision.converted_body or body

    # Strip non-standard fields if requested (for Anthropic passthrough)
    if strip_fields and decision.target_format == "anthropic":
        target_body = _clean_anthropic_body(target_body)

    logger.info(
        f"[ProviderProxy] model={model_name} provider={provider.name} "
        f"action={decision.action} {decision.source_format}->{decision.target_format} "
        f"stream={stream}"
    )

    try:
        if stream:
            return await _handle_streaming(
                request, provider, model_name, decision,
                target_body, request_format,
            )
        else:
            return await _handle_non_streaming(
                provider, model_name, decision,
                target_body, request_format,
            )
    except aiohttp.ClientError as e:
        return _format_error_response(
            request_format,
            529,
            f"Failed to connect to provider '{provider.name}': {e}",
        )
    except asyncio.TimeoutError:
        return _format_error_response(
            request_format,
            529,
            f"Upstream request to provider '{provider.name}' timed out",
        )
    except Exception as e:
        logger.error(f"[ProviderProxy] Unexpected error: {e}", exc_info=True)
        return _format_error_response(
            request_format,
            500,
            f"Internal error: {e}",
        )


async def _handle_non_streaming(
    provider: ModelProvider,
    model_name: str,
    decision,
    body: dict,
    request_format: str,
) -> Response:
    """Handle non-streaming request with format conversion."""
    timeout = aiohttp.ClientTimeout(total=provider.timeout or 120)

    for attempt in range(_MAX_RETRIES):
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                decision.target_url,
                json=body,
                headers=decision.headers,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(
                        f"[ProviderProxy] {resp.status} (attempt {attempt+1}): "
                        f"{error_text[:300]}"
                    )
                    if (
                        resp.status in _RETRY_STATUSES
                        and attempt < _MAX_RETRIES - 1
                    ):
                        import asyncio
                        await asyncio.sleep(attempt + 1)
                        continue

                    return _upstream_error_response(
                        resp.status, error_text, request_format
                    )

                # Successful response
                resp_body = await resp.json()

                # Convert response if needed
                if decision.action == "convert":
                    if decision.target_format == "openai":
                        # OpenAI response -> Anthropic format
                        converted = convert_response(resp_body, model=model_name)
                        return JSONResponse(content=converted)
                    elif decision.target_format == "anthropic":
                        # Anthropic response -> OpenAI format
                        converted = reverse_convert_response(resp_body)
                        converted["model"] = model_name
                        return JSONResponse(content=converted)
                else:
                    # Passthrough - return as-is
                    return JSONResponse(content=resp_body)


async def _handle_streaming(
    request: Request,
    provider: ModelProvider,
    model_name: str,
    decision,
    body: dict,
    request_format: str,
) -> Response:
    """Handle streaming request with format conversion."""
    timeout = aiohttp.ClientTimeout(total=provider.timeout or 120)

    if decision.action == "convert":
        if decision.target_format == "openai":
            # OpenAI stream -> Anthropic SSE
            return await _openai_to_anthropic_stream(
                provider, model_name, decision, body, timeout
            )
        elif decision.target_format == "anthropic":
            # Anthropic stream -> OpenAI SSE
            return await _anthropic_to_openai_stream(
                provider, model_name, decision, body, timeout
            )
    else:
        # Passthrough streaming
        return await _passthrough_stream(decision, body, timeout)


async def _openai_to_anthropic_stream(
    provider, model_name, decision, body, timeout
) -> StreamingResponse:
    """Stream OpenAI response and convert to Anthropic SSE events."""

    async def generate():
        for attempt in range(_MAX_RETRIES):
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    decision.target_url,
                    json=body,
                    headers=decision.headers,
                ) as resp:
                    if resp.status != 200:
                        chunks = []
                        async for chunk in resp.content.iter_chunked(4096):
                            chunks.append(chunk)
                        err = b"".join(chunks).decode("utf-8", errors="replace")
                        if (
                            resp.status in _RETRY_STATUSES
                            and attempt < _MAX_RETRIES - 1
                        ):
                            import asyncio
                            await asyncio.sleep(attempt + 1)
                            continue
                        try:
                            eb = json.loads(err)
                        except Exception:
                            eb = {"error": {"message": err, "type": "api_error"}}
                        _, e = convert_error(resp.status, eb)
                        from gpustack.converter.converters import sse_event
                        yield sse_event("error", e).encode()
                        return

                    async for chunk in convert_openai_stream_to_anthropic(
                        resp, model_name
                    ):
                        yield chunk
                    break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _anthropic_to_openai_stream(
    provider, model_name, decision, body, timeout
) -> StreamingResponse:
    """Stream Anthropic response and convert to OpenAI SSE events."""

    async def generate():
        for attempt in range(_MAX_RETRIES):
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    decision.target_url,
                    json=body,
                    headers=decision.headers,
                ) as resp:
                    if resp.status != 200:
                        chunks = []
                        async for chunk in resp.content.iter_chunked(4096):
                            chunks.append(chunk)
                        err = b"".join(chunks).decode("utf-8", errors="replace")
                        if (
                            resp.status in _RETRY_STATUSES
                            and attempt < _MAX_RETRIES - 1
                        ):
                            import asyncio
                            await asyncio.sleep(attempt + 1)
                            continue
                        yield (
                            f"data: {json.dumps({'error': {'message': err, 'type': 'api_error'}})}\n\n"
                        ).encode()
                        return

                    async for chunk in convert_anthropic_stream_to_openai(
                        resp, model_name
                    ):
                        yield chunk
                    break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _passthrough_stream(decision, body, timeout) -> StreamingResponse:
    """Passthrough streaming without conversion."""

    async def generate():
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                decision.target_url,
                json=body,
                headers=decision.headers,
            ) as resp:
                async for chunk in resp.content.iter_chunked(4096):
                    yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _format_error_response(
    request_format: str, status_code: int, message: str
) -> Response:
    """Build error response in the appropriate format."""
    if request_format == "anthropic":
        return JSONResponse(
            status_code=status_code,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": message,
                },
            },
        )
    else:
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": message,
                    "type": "api_error",
                    "code": "provider_error",
                }
            },
        )


def _upstream_error_response(
    status_code: int, error_text: str, request_format: str
) -> Response:
    """Build error response from upstream error."""
    try:
        error_body = json.loads(error_text)
    except (json.JSONDecodeError, TypeError):
        error_body = {"error": {"message": error_text}}

    if request_format == "anthropic":
        sc, body = convert_error(status_code, error_body)
        return JSONResponse(status_code=sc, content=body)
    else:
        return JSONResponse(status_code=status_code, content=error_body)


# Need asyncio import for retry sleep
import asyncio
