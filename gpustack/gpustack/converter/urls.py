"""URL building utilities for format conversion routing.

Ported from cc_proxy/urls.py.
"""

import re


def dedupe_base_url_path(base_url: str, target_url: str) -> str:
    """Remove path segments duplicated between base_url and target_url.

    Example:
        base="http://host/v1", target="http://host/v1/v1/messages"
        -> "http://host/v1/messages"
    """
    if not base_url or not target_url:
        return target_url

    base_path = base_url.rstrip("/").split("//")[-1]
    if "/" in base_path:
        last_segment = "/" + base_path.rsplit("/", 1)[-1]
    else:
        return target_url

    doubled = last_segment + last_segment
    if doubled in target_url:
        return target_url.replace(doubled, last_segment, 1)

    return target_url


def build_openai_url(base_url: str, path: str) -> str:
    """Build an OpenAI upstream URL.

    If base_url already contains a version path (/v2, /v4, etc.), do not
    prepend /v1 from the path.

    Example:
        base="https://host/api/paas/v4", path="/v1/chat/completions"
        -> "https://host/api/paas/v4/chat/completions"
    """
    stripped = base_url.rstrip("/")
    if re.search(r"/v\d+$", stripped):
        actual_path = re.sub(r"^/v\d+/", "/", path)
        raw_url = stripped + actual_path
    else:
        raw_url = stripped + path
    return dedupe_base_url_path(stripped, raw_url)


def strip_trailing_v1(url: str) -> str:
    """Remove trailing /v1 from a URL to avoid duplication when building paths."""
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url
