"""Client helper for the AI Private Layer REST API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

from config import PRIVATE_LAYER_TIMEOUT


class PrivateLayerError(RuntimeError):
    """Raised when the private layer API returns an error."""


@dataclass
class SanitizedResult:
    """Structured response from /v1/detect-encrypt."""

    text_with_placeholders: str
    bundles: List[Dict[str, Any]]
    request_id: str


class PrivateLayerClient:
    """Lightweight HTTP client around detect/decrypt endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        tenant_id: str,
        detect_path: str = "/v1/detect-encrypt",
        decrypt_path: str = "/v1/decrypt",
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.detect_url = urljoin(self.base_url, detect_path.lstrip("/"))
        self.decrypt_url = urljoin(self.base_url, decrypt_path.lstrip("/"))
        self.timeout_seconds = timeout_seconds or PRIVATE_LAYER_TIMEOUT

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def sanitize(
        self,
        text: str,
        *,
        threshold: Optional[float] = None,
        schema: str = "v1",
        salt: Optional[str] = None,
    ) -> SanitizedResult:
        payload: Dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "text": text,
            "schema": schema,
        }
        if threshold is not None:
            payload["threshold"] = threshold
        if salt:
            payload["salt"] = salt
        try:
            response = requests.post(
                self.detect_url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.RequestException as exc:
            raise PrivateLayerError(
                f"Private layer detect request failed: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise PrivateLayerError(
                f"Private layer detect failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        return SanitizedResult(
            text_with_placeholders=data.get("text_with_placeholders", ""),
            bundles=data.get("bundles", []) or [],
            request_id=data.get("request_id", ""),
        )

    def decrypt(
        self,
        text_with_placeholders: str,
        bundles: List[Dict[str, Any]],
        *,
        salt: Optional[str] = None,
    ) -> Tuple[str, str]:
        payload: Dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "text_with_placeholders": text_with_placeholders,
            "bundles": bundles,
        }
        if salt:
            payload["salt"] = salt
        try:
            response = requests.post(
                self.decrypt_url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.RequestException as exc:
            raise PrivateLayerError(
                f"Private layer decrypt request failed: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise PrivateLayerError(
                f"Private layer decrypt failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        return data.get("text", ""), data.get("request_id", "")
