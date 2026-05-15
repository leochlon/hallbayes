from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import config_path, ensure_berry_home


@dataclass(frozen=True)
class BerryConfig:
    # Safety / permissions - all disabled by default
    allow_write: bool = False
    allow_exec: bool = False
    allow_web: bool = False
    allow_web_private: bool = False
    allowed_roots: List[str] = field(default_factory=list)

    # No approval handshakes by default
    require_plan_approval: bool = False

    # Verification policy knobs (server-owned defaults)
    # Higher = stricter (harder to pass); lower = easier to pass.
    verification_write_default_target: float = 0.95
    verification_output_default_target: float = 0.95
    # Prevent "gaming" audits by setting very low targets.
    verification_min_target: float = 0.55

    # Command execution policy (shell is never used; commands are allowlisted by argv[0]).
    exec_allowed_commands: List[str] = field(
        default_factory=lambda: [
            "python",
            "python3",
            "pytest",
            "git",
            "npm",
            "node",
            "yarn",
            "pnpm",
            "make",
            "cargo",
            "go",
        ]
    )

    # Command execution: network policy.
    # - inherit: run command normally (network access depends on host)
    # - deny: require OS-level network isolation (fails closed if unavailable)
    # - deny_if_possible: attempt isolation; if unavailable, run unsandboxed and report so
    exec_network_mode: str = "inherit"

    # Verification / enforcement - disabled by default (no blocking)
    enforce_verification: bool = False

    # Privacy
    diagnostics_opt_in: bool = False

    # Web search provider configuration (for the web_search tool)
    # Provider choices: "duckduckgo" (default), "brave", "searxng", "stub" (testing only).
    web_search_provider: str = "duckduckgo"
    # For "brave" provider
    brave_search_api_key: Optional[str] = None
    # For "searxng" provider
    searxng_url: Optional[str] = None
    # Testing-only: when provider == "stub", these results are returned without network.
    # Each result should be {"url":..., "title":..., "snippet":...}
    web_search_stub_results: List[Dict[str, str]] = field(default_factory=list)

    # Audit logging
    audit_log_enabled: bool = True
    audit_log_retention_days: int = 30

    # Product flags (paid layer)
    paid_features_enabled: bool = False

    # Verifier backend (used by hallucination/verification calls)
    verifier_backend: str = "openai"  # choices: openai, gemini, vertex, local
    verifier_model: Optional[str] = None  # resolved at call time if None
    local_base_url: str = "http://127.0.0.1:1234/v1"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _coerce(cfg: Dict[str, Any]) -> BerryConfig:
    # Keep this explicit so unknown keys are ignored.
    return BerryConfig(
        allow_write=bool(cfg.get("allow_write", False)),
        allow_exec=bool(cfg.get("allow_exec", False)),
        allow_web=bool(cfg.get("allow_web", False)),
        allow_web_private=bool(cfg.get("allow_web_private", False)),
        allowed_roots=[str(x) for x in (cfg.get("allowed_roots") or [])],
        require_plan_approval=bool(cfg.get("require_plan_approval", False)),
        verification_write_default_target=float(cfg.get("verification_write_default_target", 0.95)),
        verification_output_default_target=float(
            cfg.get("verification_output_default_target", 0.95)
        ),
        verification_min_target=float(cfg.get("verification_min_target", 0.55)),
        exec_allowed_commands=[
            str(x)
            for x in (cfg.get("exec_allowed_commands") or BerryConfig().exec_allowed_commands)
        ],
        exec_network_mode=str(cfg.get("exec_network_mode", "inherit")),
        enforce_verification=bool(cfg.get("enforce_verification", False)),
        diagnostics_opt_in=bool(cfg.get("diagnostics_opt_in", False)),
        web_search_provider=str(cfg.get("web_search_provider", "duckduckgo")),
        brave_search_api_key=(
            None
            if cfg.get("brave_search_api_key") in {None, ""}
            else str(cfg.get("brave_search_api_key"))
        ),
        searxng_url=(None if cfg.get("searxng_url") in {None, ""} else str(cfg.get("searxng_url"))),
        web_search_stub_results=[
            {
                "url": str((x or {}).get("url") or ""),
                "title": str((x or {}).get("title") or ""),
                "snippet": str((x or {}).get("snippet") or ""),
            }
            for x in (cfg.get("web_search_stub_results") or [])
            if isinstance(x, dict)
        ],
        audit_log_enabled=bool(cfg.get("audit_log_enabled", True)),
        audit_log_retention_days=int(cfg.get("audit_log_retention_days", 30)),
        paid_features_enabled=bool(cfg.get("paid_features_enabled", False)),
        verifier_backend=str(cfg.get("verifier_backend", "openai")),
        verifier_model=(
            None if cfg.get("verifier_model") in {None, ""} else str(cfg.get("verifier_model"))
        ),
        local_base_url=str(cfg.get("local_base_url", "http://127.0.0.1:1234/v1")),
    )


def load_config(project_root: Optional[Path] = None) -> BerryConfig:
    """Load config (global + optional project override)."""
    global_cfg = _coerce(_load_json(config_path()))
    if not project_root:
        cfg = global_cfg
    else:
        project_cfg_path = Path(project_root) / ".berry" / "config.json"
        project_cfg_raw = _load_json(project_cfg_path)
        if not project_cfg_raw:
            cfg = global_cfg
        else:
            merged: Dict[str, Any] = {**asdict(global_cfg), **project_cfg_raw}
            cfg = _coerce(merged)

    # Environment overrides (highest precedence).
    env_enforce = os.environ.get("BERRY_ENFORCE_VERIFICATION")
    if env_enforce is not None and env_enforce.strip() != "":
        truthy = env_enforce.strip().lower() in {"1", "true", "yes", "y", "on"}
        cfg = replace(cfg, enforce_verification=bool(truthy))

    env_web_provider = os.environ.get("BERRY_WEB_SEARCH_PROVIDER")
    if env_web_provider is not None and env_web_provider.strip() != "":
        cfg = replace(cfg, web_search_provider=str(env_web_provider).strip())

    env_brave = os.environ.get("BRAVE_SEARCH_API_KEY")
    if env_brave is not None and env_brave.strip() != "":
        cfg = replace(cfg, brave_search_api_key=str(env_brave).strip())

    env_searx = os.environ.get("SEARXNG_URL")
    if env_searx is not None and env_searx.strip() != "":
        cfg = replace(cfg, searxng_url=str(env_searx).strip())

    env_exec_net = os.environ.get("BERRY_EXEC_NETWORK_MODE")
    if env_exec_net is not None and env_exec_net.strip() != "":
        cfg = replace(cfg, exec_network_mode=str(env_exec_net).strip())

    env_verifier_backend = os.environ.get("BERRY_VERIFIER_BACKEND")
    if env_verifier_backend is not None and env_verifier_backend.strip() != "":
        cfg = replace(cfg, verifier_backend=str(env_verifier_backend).strip())

    env_verifier_model = os.environ.get("BERRY_VERIFIER_MODEL")
    if env_verifier_model is not None and env_verifier_model.strip() != "":
        cfg = replace(cfg, verifier_model=str(env_verifier_model).strip())

    env_local_base_url = os.environ.get("BERRY_LOCAL_BASE_URL")
    if env_local_base_url is not None and env_local_base_url.strip() != "":
        cfg = replace(cfg, local_base_url=str(env_local_base_url).strip())

    return cfg


def save_global_config(cfg: BerryConfig) -> Path:
    ensure_berry_home()
    p = config_path()
    p.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p
