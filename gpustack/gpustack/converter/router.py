"""Format routing logic for Anthropic <-> OpenAI request routing.

Determines whether to passthrough or convert based on provider capabilities.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from gpustack.converter.auth import AuthAdapter
from gpustack.converter.urls import build_openai_url, strip_trailing_v1

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """Result of a format routing decision."""
    action: str  # "passthrough" or "convert"
    source_format: str  # "anthropic" or "openai"
    target_format: str  # "anthropic" or "openai"
    target_url: str
    headers: dict[str, str]
    converted_body: Optional[dict] = None


class FormatRouter:
    """Routes requests between Anthropic and OpenAI formats based on provider capabilities."""

    @staticmethod
    def detect_request_format(path: str, body: dict | None = None) -> str:
        """Detect if request is 'anthropic' or 'openai' format.

        Args:
            path: The request URL path.
            body: Optional request body for additional detection.

        Returns:
            "anthropic" or "openai"
        """
        if "/v1/messages" in path:
            return "anthropic"
        if "/v1/chat/completions" in path:
            return "openai"

        # Fallback: check body structure
        if body is not None:
            if "messages" in body and isinstance(body["messages"], list):
                if body.get("system") is not None:
                    return "anthropic"
                # Check if messages use Anthropic-style content blocks
                for msg in body["messages"]:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") in (
                                    "tool_result",
                                    "tool_use",
                                    "thinking",
                                ):
                                    return "anthropic"
            return "openai"

        return "openai"

    @staticmethod
    def get_provider_supported_formats(
        provider_config,
        provider_supported_formats: Optional[List[str]] = None,
    ) -> List[str]:
        """Determine the formats a provider supports.

        Args:
            provider_config: The provider's config object.
            provider_supported_formats: Explicitly set supported_formats.

        Returns:
            List of supported format strings.
        """
        if provider_supported_formats:
            return provider_supported_formats

        # Infer from provider type
        from gpustack.schemas.model_provider import ModelProviderTypeEnum

        config_type = getattr(provider_config, "type", None)
        if config_type == ModelProviderTypeEnum.CLAUDE:
            return ["anthropic"]
        elif config_type == ModelProviderTypeEnum.BEDROCK:
            return ["anthropic"]
        else:
            # Most providers use OpenAI-compatible format
            return ["openai"]

    @staticmethod
    def should_passthrough(
        request_format: str, provider_formats: List[str]
    ) -> bool:
        """Check if provider natively supports the request format.

        Args:
            request_format: The format of the incoming request.
            provider_formats: Formats the provider natively supports.

        Returns:
            True if the provider can handle the request directly.
        """
        return request_format in provider_formats

    @staticmethod
    def get_target_format(
        request_format: str, provider_formats: List[str]
    ) -> str:
        """Determine what format to use for the upstream provider.

        Args:
            request_format: The format of the incoming request.
            provider_formats: Formats the provider supports.

        Returns:
            The target format to use.

        Raises:
            ValueError: If no compatible format can be determined.
        """
        # Direct match
        if request_format in provider_formats:
            return request_format

        # Provider supports the other format - convert
        if provider_formats:
            return provider_formats[0]

        raise ValueError(
            f"Cannot determine target format for request_format={request_format}, "
            f"provider_formats={provider_formats}"
        )

    @staticmethod
    def build_target_url(
        base_url: str,
        target_format: str,
        base_url_anthropic: str = None,
    ) -> str:
        """Build the target URL based on format.

        Args:
            base_url: Primary provider base URL (used for OpenAI format).
            target_format: "anthropic" or "openai".
            base_url_anthropic: Optional Anthropic-specific base URL.
                When set, used for Anthropic format requests instead of base_url.

        Returns:
            Full URL to send the request to.
        """
        if target_format == "openai":
            return build_openai_url(base_url, "/v1/chat/completions")
        elif target_format == "anthropic":
            actual_base = base_url_anthropic or base_url
            clean_base = strip_trailing_v1(actual_base)
            return f"{clean_base}/v1/messages"
        else:
            return base_url

    @staticmethod
    def build_auth_headers(
        api_key: str,
        target_format: str,
        provider_type: Optional[str] = None,
        auth_style: Optional[str] = None,
    ) -> dict[str, str]:
        """Build authentication headers for the target format.

        Args:
            api_key: The API key.
            target_format: The format being sent to.
            provider_type: The provider type for default auth style.
            auth_style: Explicit auth style override.

        Returns:
            Headers dictionary.
        """
        style = AuthAdapter.determine_auth_style(provider_type, auth_style)
        return AuthAdapter.build_headers(api_key, target_format, style)

    @staticmethod
    def route(
        path: str,
        body: dict,
        base_url: str,
        api_key: str,
        provider_config=None,
        provider_supported_formats: Optional[List[str]] = None,
        auth_style: Optional[str] = None,
        auth_style_anthropic: Optional[str] = None,
        model_map: dict | None = None,
        base_url_anthropic: Optional[str] = None,
    ) -> RouteDecision:
        """Make a complete routing decision.

        Args:
            path: The request URL path.
            body: The request body.
            base_url: Provider base URL (used for OpenAI format).
            api_key: API key for the provider.
            provider_config: The provider's config object.
            provider_supported_formats: Explicit supported formats.
            auth_style: Auth style for OpenAI format.
            auth_style_anthropic: Auth style for Anthropic format. Falls back to auth_style.
            model_map: Optional model name mapping.
            base_url_anthropic: Optional Anthropic-specific base URL.

        Returns:
            RouteDecision with all information needed to execute the request.
        """
        request_format = FormatRouter.detect_request_format(path, body)
        provider_formats = FormatRouter.get_provider_supported_formats(
            provider_config, provider_supported_formats
        )

        passthrough = FormatRouter.should_passthrough(
            request_format, provider_formats
        )
        target_format = FormatRouter.get_target_format(
            request_format, provider_formats
        )
        target_url = FormatRouter.build_target_url(
            base_url, target_format, base_url_anthropic=base_url_anthropic
        )

        # Select auth style based on target format
        effective_auth_style = auth_style
        if target_format == "anthropic" and auth_style_anthropic:
            effective_auth_style = auth_style_anthropic

        provider_type = (
            getattr(provider_config, "type", None)
            if provider_config
            else None
        )
        headers = FormatRouter.build_auth_headers(
            api_key, target_format, provider_type, effective_auth_style
        )

        converted_body = None
        if not passthrough:
            converted_body = FormatRouter._convert_body(
                body, request_format, target_format, model_map
            )
        else:
            converted_body = body

        action = "passthrough" if passthrough else "convert"
        if action == "convert":
            logger.info(
                f"[FormatRouter] {action}: {request_format} -> {target_format} "
                f"url={target_url}"
            )

        return RouteDecision(
            action=action,
            source_format=request_format,
            target_format=target_format,
            target_url=target_url,
            headers=headers,
            converted_body=converted_body,
        )

    @staticmethod
    def _convert_body(
        body: dict, source: str, target: str, model_map: dict | None = None
    ) -> dict:
        """Convert request body between formats.

        Args:
            body: The request body.
            source: Source format.
            target: Target format.
            model_map: Optional model name mapping.

        Returns:
            Converted request body.
        """
        if source == target:
            return body

        if source == "anthropic" and target == "openai":
            from gpustack.converter.converters import convert_request

            return convert_request(body, model_map=model_map)
        elif source == "openai" and target == "anthropic":
            from gpustack.converter.converters import reverse_convert_request

            return reverse_convert_request(body, model_map=model_map)
        else:
            raise ValueError(
                f"Unsupported conversion: {source} -> {target}"
            )
