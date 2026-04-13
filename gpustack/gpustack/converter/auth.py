"""Authentication header adaptation for different provider types."""

from typing import Optional


ANTHROPIC_VERSION = "2023-06-01"


class AuthAdapter:
    """Adapts authentication headers based on target provider format and auth style."""

    @staticmethod
    def build_headers(
        api_key: str,
        target_format: str,
        auth_style: str = "auto",
        user_agent: str = "",
    ) -> dict[str, str]:
        """Build authentication headers for the target provider.

        Args:
            api_key: The API key to use.
            target_format: The format of the target API ("openai" or "anthropic").
            auth_style: How to send auth credentials:
                - "bearer": Authorization: Bearer <key>
                - "x-api-key": x-api-key: <key>
                - "auto": Send both headers (Anthropic default)
            user_agent: Optional User-Agent header value.

        Returns:
            Dictionary of HTTP headers.
        """
        if target_format == "openai":
            return AuthAdapter._openai_headers(api_key, user_agent)
        elif target_format == "anthropic":
            return AuthAdapter._anthropic_headers(api_key, auth_style, user_agent)
        else:
            # Default to bearer
            hdrs = {
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            }
            if user_agent:
                hdrs["User-Agent"] = user_agent
            return hdrs

    @staticmethod
    def _openai_headers(api_key: str, user_agent: str = "") -> dict[str, str]:
        """Build OpenAI-style authentication headers."""
        hdrs: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        if user_agent:
            hdrs["User-Agent"] = user_agent
        return hdrs

    @staticmethod
    def _anthropic_headers(
        api_key: str, auth_style: str = "auto", user_agent: str = ""
    ) -> dict[str, str]:
        """Build Anthropic-style authentication headers."""
        hdrs: dict[str, str] = {
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if auth_style == "bearer":
            hdrs["Authorization"] = f"Bearer {api_key}"
        elif auth_style == "x-api-key":
            hdrs["x-api-key"] = api_key
        else:  # auto
            hdrs["x-api-key"] = api_key
            hdrs["Authorization"] = f"Bearer {api_key}"
        if user_agent:
            hdrs["User-Agent"] = user_agent
        return hdrs

    @staticmethod
    def determine_auth_style(
        provider_type: Optional[str] = None,
        auth_style: Optional[str] = None,
    ) -> str:
        """Determine the auth style based on provider type and explicit setting.

        Args:
            provider_type: The provider type (e.g., "claude", "openai").
            auth_style: Explicit auth style override.

        Returns:
            The auth style to use.
        """
        if auth_style:
            return auth_style
        if provider_type == "claude":
            return "x-api-key"
        return "bearer"
