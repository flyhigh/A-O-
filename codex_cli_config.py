from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


def _parse_bool_env(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return None


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_read_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:
        return {}
    try:
        if not path.is_file():
            return {}
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _default_codex_dir() -> Path:
    home_override = os.getenv("CODEX_HOME")
    if home_override:
        return Path(home_override).expanduser()
    return Path.home() / ".codex"


@dataclass(frozen=True)
class CodexCliConfig:
    api_key: Optional[str]
    provider: Optional[str]
    base_url: Optional[str]
    responses_url: Optional[str]
    wire_api: Optional[str]
    requires_openai_auth: Optional[bool]
    model: Optional[str]
    model_reasoning_effort: Optional[str]
    disable_response_storage: Optional[bool]

    @staticmethod
    def from_env() -> "CodexCliConfig":
        codex_dir = _default_codex_dir()

        auth_path = Path(os.getenv("CODEX_AUTH_JSON_PATH") or (codex_dir / "auth.json")).expanduser()
        config_path = Path(os.getenv("CODEX_CONFIG_TOML_PATH") or (codex_dir / "config.toml")).expanduser()

        auth = _safe_read_json(auth_path)
        cfg = _safe_read_toml(config_path)

        provider = cfg.get("model_provider")
        model = cfg.get("model")
        model_reasoning_effort = cfg.get("model_reasoning_effort")
        disable_response_storage = cfg.get("disable_response_storage")

        provider_cfg = (cfg.get("model_providers") or {}).get(provider or "") if provider else None
        base_url = provider_cfg.get("base_url") if isinstance(provider_cfg, dict) else None
        wire_api = provider_cfg.get("wire_api") if isinstance(provider_cfg, dict) else None
        requires_openai_auth = provider_cfg.get("requires_openai_auth") if isinstance(provider_cfg, dict) else None

        responses_url = None
        if isinstance(base_url, str) and base_url.strip():
            responses_url = base_url.rstrip("/") + "/v1/responses"

        api_key = auth.get("OPENAI_API_KEY")
        if not api_key:
            # Back-compat with some setups
            api_key = auth.get("UPSTREAM_API_KEY") or auth.get("API_KEY")

        # Allow simple env overrides without requiring file edits.
        if os.getenv("CODEX_MODEL"):
            model = os.getenv("CODEX_MODEL")
        if os.getenv("CODEX_MODEL_REASONING_EFFORT"):
            model_reasoning_effort = os.getenv("CODEX_MODEL_REASONING_EFFORT")
        env_disable_storage = _parse_bool_env(os.getenv("CODEX_DISABLE_RESPONSE_STORAGE"))
        if env_disable_storage is not None:
            disable_response_storage = env_disable_storage

        return CodexCliConfig(
            api_key=api_key if isinstance(api_key, str) else None,
            provider=provider if isinstance(provider, str) else None,
            base_url=base_url if isinstance(base_url, str) else None,
            responses_url=responses_url,
            wire_api=wire_api if isinstance(wire_api, str) else None,
            requires_openai_auth=requires_openai_auth if isinstance(requires_openai_auth, bool) else None,
            model=model if isinstance(model, str) else None,
            model_reasoning_effort=model_reasoning_effort if isinstance(model_reasoning_effort, str) else None,
            disable_response_storage=disable_response_storage if isinstance(disable_response_storage, bool) else None,
        )

