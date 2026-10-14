"""Thin client for the Flask inference microservice."""

import io

import requests
from django.conf import settings


class InferenceServiceError(Exception):
    """Raised when the upstream inference service cannot fulfil a request."""


def _predict_url():
    return settings.FORGERY_API_URL.rstrip("/") + "/predict"


def analyze_image(image_bytes, filename="upload.png"):
    """Send ``image_bytes`` to the model service and return its JSON result.

    Raises :class:`InferenceServiceError` for any transport or service-level
    failure so callers can translate it into a clean API error.
    """
    files = {"file": (filename or "upload.png", io.BytesIO(image_bytes), "application/octet-stream")}
    try:
        response = requests.post(_predict_url(), files=files, timeout=settings.FORGERY_API_TIMEOUT)
    except requests.RequestException as exc:
        raise InferenceServiceError(f"Inference service unreachable: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise InferenceServiceError("Inference service returned a non-JSON response.") from exc

    if response.status_code != 200 or "error" in payload:
        message = payload.get("error", f"Inference failed with status {response.status_code}.")
        raise InferenceServiceError(message)

    return payload


def check_health():
    """Return a small dict describing the inference service status."""
    try:
        response = requests.get(
            settings.FORGERY_API_URL.rstrip("/") + "/health",
            timeout=min(settings.FORGERY_API_TIMEOUT, 10),
        )
        data = response.json()
        return {"reachable": True, "status_code": response.status_code, "inference": data}
    except requests.RequestException as exc:
        return {"reachable": False, "error": str(exc)}
    except ValueError:
        return {"reachable": False, "error": "Invalid JSON from inference service."}
