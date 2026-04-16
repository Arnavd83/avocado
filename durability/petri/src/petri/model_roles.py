from __future__ import annotations

import os
from dataclasses import dataclass

from inspect_ai.model import GenerateConfig, Model, get_model


@dataclass(frozen=True)
class RoleModelOverrides:
    base_url: str | None = None
    api_key: str | None = None
    api_version: str | None = None

    @property
    def enabled(self) -> bool:
        return any((self.base_url, self.api_key, self.api_version))


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def role_model_overrides(role: str) -> RoleModelOverrides:
    """Read optional per-role model credentials from environment variables."""
    prefix = f"PETRI_{role.upper()}_"
    return RoleModelOverrides(
        base_url=_env(f"{prefix}BASE_URL"),
        api_key=_env(f"{prefix}API_KEY"),
        api_version=_env(f"{prefix}API_VERSION"),
    )


def _normalize_base_url_for_model(model_name: str, base_url: str) -> str:
    """
    Normalize provider-specific base URLs for inspect model strings.

    Azure Anthropic clients expect a base like:
      https://<resource>.services.ai.azure.com/anthropic
    and will append the messages path themselves.
    Users often paste:
      .../anthropic/v1/messages
    so trim that suffix when model is ``azure/...``.
    """
    normalized = base_url.rstrip("/")
    if model_name.startswith("azure/") or "/azure/" in model_name:
        for suffix in ("/v1/messages", "/messages"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
    return normalized


def _build_override_kwargs(
    overrides: RoleModelOverrides, *, model_name: str,
) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    if overrides.base_url:
        kwargs["base_url"] = _normalize_base_url_for_model(model_name, overrides.base_url)
    if overrides.api_key:
        kwargs["api_key"] = overrides.api_key
    if overrides.api_version:
        kwargs["api_version"] = overrides.api_version
    return kwargs


def get_model_for_role(
    role: str,
    *,
    default: str | Model | None = None,
    config: GenerateConfig = GenerateConfig(),
) -> Model:
    """
    Resolve a role-bound model and optionally apply per-role credential overrides.

    By default this mirrors ``get_model(role=...)`` behavior. If any of:
    ``PETRI_<ROLE>_BASE_URL``, ``PETRI_<ROLE>_API_KEY``, or
    ``PETRI_<ROLE>_API_VERSION`` are set, we rebuild the role model with those
    settings. This allows different credentials/endpoints for auditor, target,
    and judge within the same run.
    """
    role_model = get_model(role=role, default=default, config=config)
    overrides = role_model_overrides(role)
    if not overrides.enabled:
        return role_model

    return get_model(
        model=str(role_model),
        config=config,
        memoize=True,
        **_build_override_kwargs(overrides, model_name=str(role_model)),
    )


def get_model_with_role_overrides(
    model_name: str,
    *,
    role: str,
    config: GenerateConfig = GenerateConfig(),
) -> Model:
    """
    Resolve an explicit model name while applying overrides for a logical role.

    Useful for fixed fallback models (e.g., judge fallback) that should still use
    role-specific credentials.
    """
    overrides = role_model_overrides(role)
    return get_model(
        model=model_name,
        config=config,
        memoize=True,
        **_build_override_kwargs(overrides, model_name=model_name),
    )
