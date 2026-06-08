from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

from . import __version__
from .audit import export_events, prune_events
from .auto_auth import maybe_provision_anon_key
from .clients import (
    berry_server_specs,
    render_claude_mcp_json,
    render_codex_config_toml,
    render_cursor_deeplink,
    render_cursor_mcp_json,
    render_gemini_settings_json,
    write_claude_mcp_json,
    write_codex_config_toml,
    write_cursor_mcp_json,
    write_gemini_settings_json,
)
from .config import load_config, save_global_config
from .installer import actions_to_json, format_actions, install_many, platform_keys
from .integration import integrate, results_as_json
from .mcp_server import main as mcp_classic_main
from .paths import ensure_berry_home, license_path, mcp_env_path
from .recipes import (
    builtin_recipes,
    export_recipes,
    get_builtin_recipe,
    install_recipe_file_to_project,
    install_recipe_to_project,
)
from .support import create_support_bundle
from .verify import verify_blob_with_cosign

# --------------------------------------------------------------------
# Claude Code skill file content (repo-scoped).
# Written by `berry init` unless --no-claude-skill is passed.
# --------------------------------------------------------------------

_CLAUDE_BERRY_SKILL_MD = """# Berry: evidence-first workflow

You have access to Berry MCP tools that *verify* claims against gathered evidence.

## Which tool to use
- Use **berry_solve** to answer questions.
- Use **berry_change** to produce a verified *plan* for code changes.
- Use **berry_status** to see what the server can do (web/exec/write, baseline mode).
- Use **berry_approve** only after the user explicitly approves a pending grant.
- Use **berry_health** for a quick self-test.

## Read the tool state machine (do not guess)
Berry responses include **state**:

- **state=need_grant**
  - Action: Show the user what scopes are being requested (from **grant_scopes** and **grant_summary**).
  - Ask: "Approve? (yes/no)".
  - Only if the user says yes: call **berry_approve(run_id, grant_token)**.
  - Then retry the original call with the same **run_id**.

- **state=ask_user**
  - Action: Ask the user the returned **questions** verbatim.
  - Then retry with the same **run_id**, passing answers in **user_context** (or append to the question).

- **state=done**
  - Action: Use the returned verified **answer** / **plan**.

- **state=cannot**
  - Action: Switch to a different tool surface or ask the user for the missing artifact.

## Evidence rules (how to avoid hallucinations)
- Treat Berry's evidence spans as the only source of truth for factual claims.
- Prefer repo-baseline evidence (git) over working-tree evidence.
- If the repo is empty (greenfield), ask the user for requirements unless they explicitly say "use best judgement".

## Common pitfalls
- Do not keep re-calling Berry when it returns **state=ask_user**. Ask the user first.
- Do not answer Berry's clarifying questions yourself unless the user delegated ("use best judgement").
- If you need working-tree evidence, capture it explicitly as spans (e.g., `add_file_span`) rather than relying on unstated context.
"""


def _find_repo_root(start: Path) -> Path:
    p = Path(start).resolve()
    for _ in range(50):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(start).resolve()


def _write_claude_skill_file(project_root: Path, *, force: bool) -> Optional[Path]:
    """Write a repo-scoped Claude Code rules/skill file for Berry."""
    dst = project_root / ".claude" / "rules" / "berry.md"
    try:
        if dst.exists() and not force:
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(_CLAUDE_BERRY_SKILL_MD, encoding="utf-8")
        return dst
    except Exception:
        return None


def cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    argv: list[str] = []
    if args.transport:
        argv += ["--transport", str(args.transport)]
    if args.host:
        argv += ["--host", str(args.host)]
    if args.port is not None:
        argv += ["--port", str(int(args.port))]
    if args.project_root:
        argv += ["--project-root", str(args.project_root)]
    # Berry ships a single MCP surface (classic). Older configs may still pass
    # `--server science` or `--server forge`; treat them as aliases for classic.
    _server = str(getattr(args, "server", "classic") or "classic").strip().lower()
    if _server != "classic":
        _server = "classic"
    mcp_classic_main(argv=argv)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    # Resolve the intended project root.
    #
    # Safety: if we can't find a .git root and the user didn't explicitly set
    # --project-root, we fail closed by default to avoid accidentally treating a
    # broad directory (e.g. $HOME) as the "project".
    if getattr(args, "project_root", None):
        project_root = Path(args.project_root).expanduser().resolve()
    else:
        project_root = _find_repo_root(Path.cwd())
        allow_non_git = os.environ.get("BERRY_ALLOW_NON_GIT_ROOT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if not (project_root / ".git").exists() and not allow_non_git:
            raise SystemExit(
                "Could not find a .git directory from the current working directory. "
                "Run `berry init` from inside a git repo, pass --project-root, or set BERRY_ALLOW_NON_GIT_ROOT=1."
            )
    force = bool(args.force)
    strict = bool(getattr(args, "strict", False))

    # Preflight: avoid partial state by checking for any conflicts before writing.
    targets: list[Path] = [
        project_root / ".cursor" / "mcp.json",
        project_root / ".codex" / "config.toml",
        project_root / ".mcp.json",
        project_root / ".gemini" / "settings.json",
    ]
    if strict:
        targets.append(project_root / ".berry" / "config.json")
    if not force:
        conflicts = [t for t in targets if t.exists()]
        if conflicts:
            raise FileExistsError(
                "Refusing to overwrite existing files (use --force): "
                + ", ".join(str(c) for c in conflicts)
            )

    created: list[Path] = []
    # Berry now ships a single MCP surface (classic). Any profile value is treated as classic.
    profile = str(getattr(args, "profile", "classic") or "classic")
    specs = berry_server_specs(profile=profile, name="berry")

    created.append(write_cursor_mcp_json(project_root=project_root, spec=specs, force=force))
    created.append(write_codex_config_toml(project_root=project_root, spec=specs, force=force))
    created.append(write_claude_mcp_json(project_root=project_root, spec=specs, force=force))
    created.append(write_gemini_settings_json(project_root=project_root, spec=specs, force=force))

    # Optional: install a repo-scoped Claude Code skill file so agents understand Berry's state machine.
    if not bool(getattr(args, "no_claude_skill", False)):
        p = _write_claude_skill_file(project_root, force=force)
        if p is not None:
            created.append(p)

    # Project-local Berry config folder (optional but useful for recipes/workflows).
    berry_dir = project_root / ".berry"
    berry_dir.mkdir(parents=True, exist_ok=True)
    if strict:
        cfg_path = berry_dir / "config.json"
        if cfg_path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite: {cfg_path} (use --force)")
        cfg_path.write_text(
            json.dumps({"enforce_verification": True}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        created.append(cfg_path)

    for p in created:
        print(str(p))

    # Best-effort: provision a hosted key for first-time users (opt-out).
    try:
        res = maybe_provision_anon_key(client="berry", version=__version__)
        if res.changed and res.message:
            print(res.message, file=sys.stderr)
        elif (not res.ok) and res.message:
            print(
                f"warn: {res.message}. If you have your own key, run `berry setup` (advanced) instead.",
                file=sys.stderr,
            )
    except Exception as e:
        print(f"warn: auto-auth failed: {e}", file=sys.stderr)
    return 0


def _resolve_project_root_for_install(args: argparse.Namespace) -> Path:
    if getattr(args, "project_root", None):
        return Path(args.project_root).expanduser().resolve()
    project_root = _find_repo_root(Path.cwd())
    allow_non_git = os.environ.get("BERRY_ALLOW_NON_GIT_ROOT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if not (project_root / ".git").exists() and not allow_non_git:
        raise SystemExit(
            "Could not find a .git directory from the current working directory. "
            "Run from inside a git repo, pass --project-root, or set BERRY_ALLOW_NON_GIT_ROOT=1."
        )
    return project_root


def cmd_install(args: argparse.Namespace) -> int:
    if bool(getattr(args, "list_platforms", False)):
        for key in platform_keys():
            print(key)
        return 0

    raw_platforms = list(getattr(args, "platforms", None) or [])
    if not raw_platforms:
        raw_platform = getattr(args, "platform", None) or "auto"
        raw_platforms = [str(raw_platform)]
    elif getattr(args, "platform", None):
        raw_platforms.append(str(args.platform))

    project = bool(getattr(args, "project", False))
    scope = "project" if project else "user"
    project_root = _resolve_project_root_for_install(args) if project else Path.cwd().resolve()

    try:
        actions = install_many(
            raw_platforms,
            scope=scope,  # type: ignore[arg-type]
            project_dir=project_root,
            force=bool(getattr(args, "force", False)),
            dry_run=bool(getattr(args, "dry_run", False)),
            berry_command_raw=getattr(args, "berry_command", None),
            name=str(getattr(args, "name", "berry") or "berry"),
            install_hooks=not bool(getattr(args, "no_hooks", False)),
            install_mcp=not bool(getattr(args, "no_mcp", False)),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if bool(getattr(args, "json", False)):
        print(actions_to_json(actions), end="")
    else:
        print(format_actions(actions), end="")
        failed = [a for a in actions if a.status == "failed"]
        if not failed:
            print(
                "Done. Re-run `berry install` after moving/upgrading Berry to refresh embedded paths."
            )

    return 2 if any(a.status == "failed" for a in actions) else 0


def cmd_doctor(_: argparse.Namespace) -> int:
    project_root = _find_repo_root(Path.cwd())
    cfg = load_config(project_root=project_root)

    checks = {
        "python": sys.version.split()[0],
        "project_root": str(project_root),
        "allow_write": cfg.allow_write,
        "allow_exec": cfg.allow_exec,
        "allow_web": cfg.allow_web,
        "enforce_verification": cfg.enforce_verification,
        "require_plan_approval": cfg.require_plan_approval,
        "audit_log_enabled": cfg.audit_log_enabled,
        "diagnostics_opt_in": cfg.diagnostics_opt_in,
    }
    print(json.dumps(checks, indent=2))
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    project_root = _find_repo_root(Path.cwd())
    cfg = load_config(project_root=project_root)
    print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    return 0


def cmd_config_set(args: argparse.Namespace) -> int:
    cfg = load_config(project_root=None)
    key = str(args.key)
    raw = str(args.value)

    bool_keys = {
        "allow_write",
        "allow_exec",
        "allow_web",
        "allow_web_private",
        "enforce_verification",
        "require_plan_approval",
        "diagnostics_opt_in",
        "audit_log_enabled",
        "paid_features_enabled",
    }
    float_keys = {
        "verification_write_default_target",
        "verification_output_default_target",
        "verification_min_target",
    }
    int_keys = {
        "audit_log_retention_days",
    }

    str_keys = {
        # Execution
        "exec_network_mode",  # inherit|deny|deny_if_possible
        # Web search
        "web_search_provider",  # duckduckgo|brave|searxng|stub
        "searxng_url",
        "brave_search_api_key",
    }

    if key in bool_keys:
        truthy = raw.strip().lower() in {"1", "true", "yes", "y", "on"}
        new_cfg = replace(cfg, **{key: bool(truthy)})  # type: ignore[arg-type]
    elif key in float_keys:
        new_cfg = replace(cfg, **{key: float(raw)})  # type: ignore[arg-type]
    elif key in int_keys:
        new_cfg = replace(cfg, **{key: int(raw)})  # type: ignore[arg-type]
    elif key == "exec_allowed_commands":
        cmds = [c.strip() for c in raw.split(",") if c.strip()]
        if not cmds:
            raise SystemExit("exec_allowed_commands must be a non-empty comma-separated list")
        new_cfg = replace(cfg, exec_allowed_commands=cmds)
    elif key in str_keys:
        v = raw.strip()
        if key == "exec_network_mode":
            if v not in {"inherit", "deny", "deny_if_possible"}:
                raise SystemExit(
                    "exec_network_mode must be one of: inherit, deny, deny_if_possible"
                )
            new_cfg = replace(cfg, exec_network_mode=v)
        elif key == "web_search_provider":
            if v not in {"duckduckgo", "brave", "searxng", "stub"}:
                raise SystemExit(
                    "web_search_provider must be one of: duckduckgo, brave, searxng, stub"
                )
            new_cfg = replace(cfg, web_search_provider=v)
        elif key == "searxng_url":
            # Empty string unsets.
            new_cfg = replace(cfg, searxng_url=(v if v else None))
        elif key == "brave_search_api_key":
            # Empty string unsets.
            new_cfg = replace(cfg, brave_search_api_key=(v if v else None))
        else:
            new_cfg = replace(cfg, **{key: v})  # type: ignore[arg-type]
    else:
        raise SystemExit(f"Unsupported key: {key}")

    p = save_global_config(new_cfg)
    print(str(p))
    return 0


def cmd_config_add_root(args: argparse.Namespace) -> int:
    cfg = load_config(project_root=None)
    root = str(Path(args.path).expanduser().resolve())
    if root in cfg.allowed_roots:
        print("already-present")
        return 0
    new_cfg = replace(cfg, allowed_roots=[*cfg.allowed_roots, root])
    p = save_global_config(new_cfg)
    print(str(p))
    return 0


def cmd_config_remove_root(args: argparse.Namespace) -> int:
    cfg = load_config(project_root=None)
    root = str(Path(args.path).expanduser().resolve())
    new_roots = [r for r in cfg.allowed_roots if r != root]
    new_cfg = replace(cfg, allowed_roots=new_roots)
    p = save_global_config(new_cfg)
    print(str(p))
    return 0


_SETUP_ENV_KEYS = {
    # Verifier selection
    "BERRY_VERIFIER_BACKEND",
    "BERRY_VERIFIER_MODEL",
    "BERRY_MODEL",
    # OpenAI-compatible backends (OpenAI/OpenRouter/vLLM/etc.)
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    # Gemini Developer API
    "GEMINI_API_KEY",
    "GEMINI_BASE_URL",
    # Vertex AI (Gemini via Vertex generateContent)
    "VERTEX_ACCESS_TOKEN",
    "VERTEX_BASE_URL",
    "VERTEX_PROJECT",
    "VERTEX_LOCATION",
}


def _load_env_file(p: Path) -> dict[str, str]:
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items() if k and v is not None}
    except Exception:
        return {}


def _write_env_file(p: Path, env: dict[str, str]) -> None:
    if env:
        p.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            if os.name != "nt":
                p.chmod(0o600)
        except Exception:
            pass
        return
    if p.exists():
        p.unlink()


def _mask_secret(s: str) -> str:
    ss = (s or "").strip()
    if not ss:
        return ""
    if len(ss) <= 12:
        return "***"
    return ss[:8] + "..." + ss[-4:]


def cmd_setup_status(_: argparse.Namespace) -> int:
    """Show current verifier setup status (no secrets)."""
    p = mcp_env_path()
    env = _load_env_file(p)
    if not env:
        print("Not set up.")
        print("Run `berry setup` to configure a verifier.")
        return 1

    backend = (env.get("BERRY_VERIFIER_BACKEND") or "openai").strip() or "openai"
    model = (env.get("BERRY_VERIFIER_MODEL") or env.get("BERRY_MODEL") or "").strip()
    api_key = (env.get("OPENAI_API_KEY") or "").strip()
    base_url = (env.get("OPENAI_BASE_URL") or "").strip()
    gemini_key = (env.get("GEMINI_API_KEY") or "").strip()
    gemini_base_url = (env.get("GEMINI_BASE_URL") or "").strip()
    vertex_token = (env.get("VERTEX_ACCESS_TOKEN") or "").strip()
    vertex_base_url = (env.get("VERTEX_BASE_URL") or "").strip()
    vertex_project = (env.get("VERTEX_PROJECT") or "").strip()
    vertex_location = (env.get("VERTEX_LOCATION") or "").strip()

    print("Berry setup")
    print(f"  Config:  {p}")
    print(f"  Backend: {backend}")
    if model:
        print(f"  Model:   {model}")
    if backend == "openai":
        if base_url:
            print(f"  Base URL: {base_url}")
        if api_key:
            print(f"  API Key: {_mask_secret(api_key)}")
    elif backend == "gemini":
        if gemini_base_url:
            print(f"  Base URL: {gemini_base_url}")
        if gemini_key:
            print(f"  API Key: {_mask_secret(gemini_key)}")
    elif backend == "vertex":
        if vertex_project:
            print(f"  Project: {vertex_project}")
        if vertex_location:
            print(f"  Location: {vertex_location}")
        if vertex_base_url:
            print(f"  Base URL: {vertex_base_url}")
        if vertex_token:
            print(f"  Access Token: {_mask_secret(vertex_token)}")

    ok = (
        bool(api_key)
        if backend == "openai"
        else bool(gemini_key)
        if backend == "gemini"
        else bool(vertex_token)
        if backend == "vertex"
        else True
    )
    return 0 if ok else 1


def cmd_setup_clear(_: argparse.Namespace) -> int:
    """Remove saved verifier setup (keeps unrelated env keys)."""
    p = mcp_env_path()
    env = _load_env_file(p)
    if not env:
        print("No setup found.")
        return 0

    for k in sorted(_SETUP_ENV_KEYS):
        env.pop(k, None)

    _write_env_file(p, env)
    print("Cleared verifier setup.")
    print(str(p))
    return 0


def _prompt_provider() -> str:
    choices = {
        "1": "openai",
        "2": "openrouter",
        "3": "vllm",
        "4": "custom",
        "5": "vertex",
        "6": "gemini",
    }
    print("Choose a verifier provider:")
    print("  1) OpenAI (api.openai.com)")
    print("  2) OpenRouter (openrouter.ai)")
    print("  3) Local vLLM (OpenAI-compatible server)")
    print("  4) Custom OpenAI-compatible base URL")
    print("  5) Vertex AI (Gemini via Vertex generateContent)")
    print("  6) Gemini Developer API (ai.google.dev)")
    while True:
        raw = (input("Provider [1-6]: ") or "").strip().lower()
        if raw in choices:
            return choices[raw]
        if raw in {"openai", "openrouter", "vllm", "custom", "vertex", "gemini"}:
            return raw
        print("Invalid choice. Enter 1-6.")


def _normalize_base_url(raw: Optional[str]) -> Optional[str]:
    s = (raw or "").strip()
    if not s:
        return None
    # Avoid accidental double-slashes when SDK appends paths.
    return s.rstrip("/")


def _probe_openai_compat_logprobs(
    *, base_url: Optional[str], api_key: str, model: str
) -> tuple[bool, str]:
    """Probe for top_logprobs support."""
    try:
        from openai import OpenAI
    except Exception as e:
        return False, f"openai SDK not available: {e}"

    client = OpenAI(api_key=str(api_key), timeout=20, base_url=str(base_url) if base_url else None)

    try:
        resp = client.chat.completions.create(
            model=str(model),
            messages=[
                {"role": "system", "content": "Reply with a single token: YES"},
                {"role": "user", "content": "YES"},
            ],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )
    except Exception as e:
        return False, f"probe call failed: {e}"

    try:
        choice = resp.choices[0]
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp is not None else None
        if not content:
            return False, "probe succeeded but logprobs were missing/empty"
        first = content[0]
        top = getattr(first, "top_logprobs", None)
        if not top:
            return False, "probe succeeded but top_logprobs were missing/empty"
    except Exception as e:
        return False, f"unexpected probe response shape: {e}"

    return True, "ok"


def _probe_vertex_logprobs(
    *, base_url: Optional[str], access_token: str, model: str
) -> tuple[bool, str]:
    """Probe for Vertex logprobs support."""
    try:
        import httpx
    except Exception as e:
        return False, f"httpx not available: {e}"

    base = (base_url or "https://aiplatform.googleapis.com").rstrip("/")
    url = f"{base}/v1/{model}:generateContent"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "YES"}]}],
        "systemInstruction": {"parts": [{"text": "Reply with a single token: YES"}]},
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1,
            "responseLogprobs": True,
            "logprobs": 5,
        },
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=20)
    except Exception as e:
        return False, f"probe call failed: {e}"
    if resp.status_code >= 400:
        return False, f"{resp.status_code}: {resp.text}"

    try:
        data = resp.json()
        cand = (data.get("candidates") or [None])[0] or {}
        if not isinstance(cand, dict):
            return False, "probe succeeded but response candidates[0] was not an object"
        lr = cand.get("logprobsResult") or cand.get("logprobs_result") or {}
        if not isinstance(lr, dict) or not lr:
            return False, "probe succeeded but logprobsResult was missing/empty"
        top = lr.get("topCandidates") or lr.get("top_candidates") or []
        if not top:
            return False, "probe succeeded but topCandidates were missing/empty"
    except Exception as e:
        return False, f"unexpected probe response shape: {e}"

    return True, "ok"


def _probe_gemini_logprobs(
    *, base_url: Optional[str], api_key: str, model: str
) -> tuple[bool, str]:
    """Probe for Gemini logprobs support."""
    try:
        import httpx
    except Exception as e:
        return False, f"httpx not available: {e}"

    base = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    m = str(model or "").strip()
    if not (m.startswith("models/") or m.startswith("tunedModels/")):
        m = "models/" + m

    url = f"{base}/v1beta/{m}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "YES"}]}],
        "systemInstruction": {"parts": [{"text": "Reply with a single token: YES"}]},
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1,
            "responseLogprobs": True,
            "logprobs": 5,
        },
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=20)
    except Exception as e:
        return False, f"probe call failed: {e}"
    if resp.status_code >= 400:
        return False, f"{resp.status_code}: {resp.text}"

    try:
        data = resp.json()
        cand = (data.get("candidates") or [None])[0] or {}
        if not isinstance(cand, dict):
            return False, "probe succeeded but response candidates[0] was not an object"
        lr = cand.get("logprobsResult") or cand.get("logprobs_result") or {}
        if not isinstance(lr, dict) or not lr:
            return False, "probe succeeded but logprobsResult was missing/empty"
        top = lr.get("topCandidates") or lr.get("top_candidates") or []
        if not top:
            return False, "probe succeeded but topCandidates were missing/empty"
    except Exception as e:
        return False, f"unexpected probe response shape: {e}"

    return True, "ok"


def cmd_setup_set(args: argparse.Namespace) -> int:
    """Configure Berry verifier backend/model and write ~/.berry/mcp_env.json."""
    ensure_berry_home()
    p = mcp_env_path()
    env = _load_env_file(p)

    provider = (getattr(args, "provider", None) or "").strip().lower()
    if not provider:
        if not sys.stdin.isatty():
            raise SystemExit("--provider is required in non-interactive mode")
        provider = _prompt_provider()

    if provider not in {"openai", "openrouter", "vllm", "custom", "vertex", "gemini"}:
        raise SystemExit(f"Unknown provider: {provider!r}")

    base_url = _normalize_base_url(getattr(args, "base_url", None))
    if provider == "gemini":
        base_url = base_url or "https://generativelanguage.googleapis.com"
    elif provider == "vertex":
        base_url = base_url or "https://aiplatform.googleapis.com"
    elif provider == "openai":
        base_url = base_url  # optional override (e.g., proxy)
    elif provider == "openrouter":
        base_url = base_url or "https://openrouter.ai/api/v1"
    elif provider == "vllm":
        base_url = base_url or "http://localhost:8000/v1"
    else:
        # custom (OpenAI-compatible)
        if not base_url:
            if not sys.stdin.isatty():
                raise SystemExit("--base-url is required for provider=custom")
            base_url = _normalize_base_url(
                input("Base URL (OpenAI-compatible, include /v1): ").strip()
            )
        if not base_url:
            raise SystemExit("No base URL provided.")

    model = (getattr(args, "model", None) or "").strip()
    if not model and sys.stdin.isatty():
        default_model = (
            (env.get("BERRY_VERIFIER_MODEL") or "").strip()
            or (env.get("BERRY_MODEL") or "").strip()
            or "gpt-4o-mini"
        )
        model = (input(f"Verifier model [{default_model}]: ") or "").strip() or default_model
    if not model:
        raise SystemExit("--model is required (or run interactively).")

    if provider == "vertex":
        vertex_project = (
            getattr(args, "vertex_project", None) or env.get("VERTEX_PROJECT") or ""
        ).strip()
        vertex_location = (
            getattr(args, "vertex_location", None) or env.get("VERTEX_LOCATION") or "us-central1"
        ).strip()

        if not model.startswith("projects/"):
            if not vertex_project and sys.stdin.isatty():
                vertex_project = (input("Vertex project id: ") or "").strip()
            if not vertex_project:
                raise SystemExit(
                    "--vertex-project is required when --model is not a full projects/... resource name"
                )
            if not vertex_location and sys.stdin.isatty():
                vertex_location = (
                    input("Vertex location [us-central1]: ") or ""
                ).strip() or "us-central1"
            if not vertex_location:
                raise SystemExit(
                    "--vertex-location is required when --model is not a full projects/... resource name"
                )
            model = f"projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}"

    stdin = bool(getattr(args, "stdin", False))
    api_key = (getattr(args, "api_key", None) or "").strip()
    if stdin:
        api_key = (sys.stdin.read() or "").strip()
    if not api_key:
        if not sys.stdin.isatty():
            raise SystemExit("--api-key or --stdin is required in non-interactive mode")
        if provider == "vllm":
            api_key = (
                input("API key for vLLM (press enter for 'local'): ") or ""
            ).strip() or "local"
        elif provider == "vertex":
            api_key = getpass.getpass("Vertex access token (will be saved locally): ").strip()
        elif provider == "gemini":
            api_key = getpass.getpass("Gemini API key (will be saved locally): ").strip()
        else:
            api_key = getpass.getpass("API key (will be saved locally): ").strip()
    if not api_key:
        raise SystemExit("No API key provided.")

    no_verify = bool(getattr(args, "no_verify", False))
    if not no_verify:
        if provider == "vertex":
            ok, msg = _probe_vertex_logprobs(base_url=base_url, access_token=api_key, model=model)
        elif provider == "gemini":
            ok, msg = _probe_gemini_logprobs(base_url=base_url, api_key=api_key, model=model)
        else:
            ok, msg = _probe_openai_compat_logprobs(base_url=base_url, api_key=api_key, model=model)
        if not ok:
            raise SystemExit(
                "Verifier probe failed (model missing or logprobs unsupported).\n"
                f"Provider: {provider}\n"
                f"Base URL: {base_url or '(default)'}\n"
                f"Model:    {model}\n"
                f"Error:    {msg}\n"
                "\n"
                "Tip: Berry's current verifier requires token logprobs + top_logprobs."
            )

    # Apply updates (preserve unrelated keys).
    env["BERRY_VERIFIER_MODEL"] = model
    if provider == "vertex":
        env["BERRY_VERIFIER_BACKEND"] = "vertex"
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_BASE_URL", None)
        env.pop("GEMINI_API_KEY", None)
        env.pop("GEMINI_BASE_URL", None)
        env["VERTEX_ACCESS_TOKEN"] = api_key
        if base_url:
            env["VERTEX_BASE_URL"] = base_url
        if vertex_project:
            env["VERTEX_PROJECT"] = vertex_project
        if vertex_location:
            env["VERTEX_LOCATION"] = vertex_location
    elif provider == "gemini":
        env["BERRY_VERIFIER_BACKEND"] = "gemini"
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_BASE_URL", None)
        env.pop("VERTEX_ACCESS_TOKEN", None)
        env.pop("VERTEX_BASE_URL", None)
        env.pop("VERTEX_PROJECT", None)
        env.pop("VERTEX_LOCATION", None)
        env["GEMINI_API_KEY"] = api_key
        if base_url:
            env["GEMINI_BASE_URL"] = base_url
    else:
        env["BERRY_VERIFIER_BACKEND"] = "openai"
        env.pop("VERTEX_ACCESS_TOKEN", None)
        env.pop("VERTEX_BASE_URL", None)
        env.pop("VERTEX_PROJECT", None)
        env.pop("VERTEX_LOCATION", None)
        env.pop("GEMINI_API_KEY", None)
        env.pop("GEMINI_BASE_URL", None)
        env["OPENAI_API_KEY"] = api_key
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        else:
            env.pop("OPENAI_BASE_URL", None)

    _write_env_file(p, env)

    # Optional: propagate env into global MCP client config files.
    results = []
    if not bool(getattr(args, "no_integrate", False)):
        try:
            results = integrate(
                clients=["cursor", "claude", "codex", "gemini"],
                name="berry",
                timeout_s=20,
                dry_run=False,
                managed=True,
                managed_only=False,
            )
        except Exception as e:
            print(f"warn: saved setup but failed to update global MCP configs: {e}")

    keys = sorted(env.keys())
    print(str(p))
    print("saved_keys=" + ",".join(keys) if keys else "saved_keys=(none)")

    for r in results:
        if r.status != "ok":
            print(f"{r.client}: {r.status} - {r.message}")

    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    subcmd = (getattr(args, "setup_cmd", None) or "").strip().lower()
    if subcmd == "status":
        return cmd_setup_status(args)
    if subcmd == "clear":
        return cmd_setup_clear(args)
    return cmd_setup_set(args)


def cmd_support_bundle(args: argparse.Namespace) -> int:
    project_root = _find_repo_root(Path.cwd())
    out = Path(args.out).resolve() if args.out else None
    p = create_support_bundle(project_root=project_root, out_path=out)
    print(str(p))
    return 0


def cmd_support_issue(args: argparse.Namespace) -> int:
    project_root = _find_repo_root(Path.cwd())
    out = Path(args.out).resolve() if args.out else None
    bundle = create_support_bundle(project_root=project_root, out_path=out)

    print(f"Support bundle: {bundle}")
    print("")
    print("Issue template (copy/paste):")
    print("")
    print("## Summary")
    print("- What did you try to do?")
    print("- What happened instead?")
    print("")
    print("## Repro steps")
    print("1) ...")
    print("2) ...")
    print("")
    print("## Expected vs actual")
    print("- Expected: ...")
    print("- Actual: ...")
    print("")
    print("## Attachments")
    print(f"- Support bundle: {bundle}")
    return 0


def cmd_audit_export(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    export_events(out)
    print(str(out))
    return 0


def cmd_audit_prune(args: argparse.Namespace) -> int:
    project_root = _find_repo_root(Path.cwd())
    cfg = load_config(project_root=project_root)
    removed = prune_events(retention_days=int(cfg.audit_log_retention_days))
    print(str(removed))
    return 0


def cmd_recipes_list(_: argparse.Namespace) -> int:
    for r in builtin_recipes():
        print(f"{r.name}\t{r.title}\t{r.author}")
    return 0


def cmd_recipes_export(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    export_recipes(builtin_recipes(), out)
    print(str(out))
    return 0


def cmd_recipes_install(args: argparse.Namespace) -> int:
    project_root = _find_repo_root(Path.cwd())
    r = get_builtin_recipe(args.name)
    if r is None:
        raise SystemExit(f"Unknown recipe: {args.name}")
    p = install_recipe_to_project(r, project_root=project_root, force=bool(args.force))
    print(str(p))
    return 0


def cmd_recipes_import(args: argparse.Namespace) -> int:
    project_root = _find_repo_root(Path.cwd())
    src = Path(args.path).expanduser().resolve()
    p = install_recipe_file_to_project(src, project_root=project_root, force=bool(args.force))
    print(str(p))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    res = verify_blob_with_cosign(
        artifact=Path(args.artifact).resolve(),
        signature=Path(args.signature).resolve(),
        public_key=(Path(args.public_key).resolve() if args.public_key else None),
    )
    print(json.dumps({"ok": res.ok, "message": res.message}, indent=2))
    return 0 if res.ok else 2


def cmd_license_set(args: argparse.Namespace) -> int:
    ensure_berry_home()
    p = license_path()
    features = [f.strip() for f in (args.features or "").split(",") if f.strip()]
    payload = {"plan": str(args.plan), "features": features}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(p))
    return 0


def cmd_license_show(_: argparse.Namespace) -> int:
    p = license_path()
    try:
        print(p.read_text(encoding="utf-8"), end="")
        return 0
    except FileNotFoundError:
        print("{}")
        return 0


def cmd_print_config(args: argparse.Namespace) -> int:
    profile = str(getattr(args, "profile", "science") or "science")
    specs = berry_server_specs(profile=profile, name=str(args.name))
    if args.client == "cursor":
        print(render_cursor_mcp_json(specs), end="")
    elif args.client == "codex":
        print(render_codex_config_toml(specs), end="")
    elif args.client == "claude":
        print(render_claude_mcp_json(specs), end="")
    elif args.client == "gemini":
        print(render_gemini_settings_json(specs), end="")
    else:
        raise SystemExit(f"Unknown client: {args.client}")
    return 0


def cmd_deeplink(args: argparse.Namespace) -> int:
    profile = str(getattr(args, "profile", "science") or "science")
    specs = berry_server_specs(profile=profile, name=str(args.name))
    spec = specs[0]
    if args.client == "cursor":
        print(render_cursor_deeplink(spec), end="")
    else:
        raise SystemExit(f"Unknown client: {args.client}")
    return 0


def cmd_integrate(args: argparse.Namespace) -> int:
    # Default: attempt integration for all supported clients and skip those
    # not present.
    clients = list(getattr(args, "clients", None) or [])
    if not clients:
        # Default: integrate everywhere that supports global config files.
        clients = ["cursor", "claude", "codex", "gemini"]

    managed_only = bool(getattr(args, "managed_only", False))
    managed = bool(getattr(args, "managed", False)) or managed_only

    results = integrate(
        clients=clients,
        name=str(getattr(args, "name", "berry")),
        timeout_s=int(getattr(args, "timeout", 20)),
        dry_run=bool(getattr(args, "dry_run", False)),
        managed=managed,
        managed_only=managed_only,
    )

    if bool(getattr(args, "json", False)):
        print(results_as_json(results), end="")
    else:
        for r in results:
            print(f"{r.client}: {r.status} - {r.message}")

    failed = [r for r in results if r.status == "failed"]
    return 0 if not failed else 2


def cmd_quickstart(_: argparse.Namespace) -> int:
    print("1) Install Berry so `berry` is on PATH (e.g., via pipx).")
    print("2) Configure a verifier (recommended): `berry setup` (writes ~/.berry/mcp_env.json).")
    print(
        "3) Optional: run `berry integrate` to register Berry globally in supported clients (Cursor, Claude Code, Codex, Gemini CLI)."
    )
    print("4) In your repo root: run `berry init` to create repo-scoped MCP config files.")
    print(
        "5) In your MCP client (Cursor/Codex/Claude Code/Gemini CLI), reload MCP servers for the repo."
    )
    print(
        "6) Run a prompt/workflow (Search & Learn, Generate Boilerplate/Content, Inline completion guard, Greenfield prototyping, RCA Fix Agent)."
    )
    return 0


def cmd_instructions(args: argparse.Namespace) -> int:
    name = str(args.name)
    if args.client in (None, "cursor"):
        print(
            "Cursor (repo-scoped): commit `.cursor/mcp.json` (or copy/paste via "
            "`berry print-config cursor`, or install via `berry deeplink cursor`)."
        )
    if args.client in (None, "codex"):
        print(
            "Codex (repo-scoped): commit `.codex/config.toml` (or copy/paste via `berry print-config codex`)."
        )
    if args.client in (None, "claude"):
        print(
            "Claude Code (repo-scoped): commit `.mcp.json` (or copy/paste via `berry print-config claude`)."
        )
    if args.client in (None, "gemini"):
        print(
            "Gemini CLI (repo-scoped): commit `.gemini/settings.json` (or copy/paste via `berry print-config gemini`)."
        )
    if args.client is None:
        print(f"Server name in configs: {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="berry")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version").set_defaults(fn=cmd_version)

    mcp = sub.add_parser("mcp", help="Run Berry MCP server (stdio by default)")
    mcp.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    mcp.add_argument(
        "--server",
        # Berry now ships a single MCP surface (classic). The flag is kept for
        # backwards compatibility with older configs that may still reference
        # "science"/"forge"; those values are treated as aliases.
        default="classic",
        help="Which MCP surface to expose (classic). Legacy values may be accepted for older configs.",
    )
    mcp.add_argument("--host", type=str, default=None)
    mcp.add_argument("--port", type=int, default=None)
    mcp.add_argument("--project-root", type=str, default=None)
    mcp.set_defaults(fn=cmd_mcp)

    init = sub.add_parser(
        "init", help="Create repo-scoped MCP config files for Cursor/Codex/Claude/Gemini"
    )
    init.add_argument(
        "--profile",
        # Backwards compat: older docs/configs used profiles; any value now yields classic.
        default="classic",
        help="Which MCP server(s) to install into repo configs (classic)",
    )
    init.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Explicit project root (otherwise inferred by walking up to a .git directory)",
    )
    init.add_argument("--force", action="store_true", help="Overwrite existing config files")
    init.add_argument(
        "--strict",
        action="store_true",
        help="Also write .berry/config.json with enforce_verification=true",
    )
    init.add_argument(
        "--no-claude-skill",
        action="store_true",
        help="Do not write .claude/rules/berry.md (Berry skill file)",
    )
    init.set_defaults(fn=cmd_init)

    install = sub.add_parser(
        "install",
        help="Install Berry into an AI coding assistant (multi-platform installer)",
    )
    install.add_argument(
        "platforms",
        nargs="*",
        help="Optional platform names. Defaults to auto (Claude Code, or Windows on win32).",
    )
    install.add_argument(
        "--platform",
        default=None,
        help="Platform to install (aliases accepted). Use repeated positional names for multiple platforms.",
    )
    install.add_argument(
        "--project",
        action="store_true",
        help="Install into the current project instead of the user profile where supported.",
    )
    install.add_argument(
        "--project-root",
        default=None,
        help="Project root for --project installs (otherwise inferred from .git).",
    )
    install.add_argument("--name", default="berry", help="MCP server name to register")
    install.add_argument(
        "--berry-command",
        default=None,
        help="Command to embed in generated configs (default: resolved current berry executable).",
    )
    install.add_argument(
        "--force", action="store_true", help="Repair invalid JSON files where safe"
    )
    install.add_argument(
        "--dry-run", action="store_true", help="Show planned writes without writing"
    )
    install.add_argument("--json", action="store_true", help="Emit machine-readable action JSON")
    install.add_argument("--no-hooks", action="store_true", help="Do not install assistant hooks")
    install.add_argument("--no-mcp", action="store_true", help="Do not update MCP client configs")
    install.add_argument("--list-platforms", action="store_true", help="Print supported platforms")
    install.set_defaults(fn=cmd_install)

    # Convenience aliases: `berry codex install`, `berry cursor install`, ...
    for platform_name in platform_keys():
        alias = sub.add_parser(platform_name, help=f"{platform_name} platform helpers")
        alias_sub = alias.add_subparsers(dest=f"{platform_name}_cmd", required=True)
        alias_install = alias_sub.add_parser("install", help=f"Install Berry for {platform_name}")
        alias_install.add_argument("--project", action="store_true")
        alias_install.add_argument("--project-root", default=None)
        alias_install.add_argument("--name", default="berry")
        alias_install.add_argument("--berry-command", default=None)
        alias_install.add_argument("--force", action="store_true")
        alias_install.add_argument("--dry-run", action="store_true")
        alias_install.add_argument("--json", action="store_true")
        alias_install.add_argument("--no-hooks", action="store_true")
        alias_install.add_argument("--no-mcp", action="store_true")
        alias_install.set_defaults(fn=cmd_install, platforms=[platform_name], platform=None)

    doc = sub.add_parser("doctor", help="Run health checks / self-test")
    doc.set_defaults(fn=cmd_doctor)

    status = sub.add_parser("status", help="Show Berry config (effective)")
    status.set_defaults(fn=cmd_status)

    cfg = sub.add_parser("config", help="Edit global Berry config")
    cfg_sub = cfg.add_subparsers(dest="cfg_cmd", required=True)
    cfg_show = cfg_sub.add_parser("show")
    cfg_show.set_defaults(fn=cmd_status)
    cfg_set = cfg_sub.add_parser("set")
    cfg_set.add_argument(
        "key",
        choices=[
            "allow_write",
            "enforce_verification",
            "diagnostics_opt_in",
            "audit_log_enabled",
            "paid_features_enabled",
        ],
    )
    cfg_set.add_argument("value")
    cfg_set.set_defaults(fn=cmd_config_set)
    cfg_add = cfg_sub.add_parser("add-root")
    cfg_add.add_argument("path")
    cfg_add.set_defaults(fn=cmd_config_add_root)
    cfg_rm = cfg_sub.add_parser("remove-root")
    cfg_rm.add_argument("path")
    cfg_rm.set_defaults(fn=cmd_config_remove_root)

    setup = sub.add_parser("setup", help="Configure verifier backend/model and API keys")
    setup.add_argument(
        "--provider",
        choices=["openai", "openrouter", "vllm", "custom", "vertex", "gemini"],
        default=None,
        help="Verifier provider preset (prompts if omitted)",
    )
    setup.add_argument("--model", default=None, help="Verifier model name/id")
    setup.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key / access token (prompts if omitted)",
    )
    setup.add_argument("--stdin", action="store_true", help="Read API key from stdin")
    setup.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="Base URL (OpenAI-compatible include /v1; Vertex uses https://aiplatform.googleapis.com; Gemini uses https://generativelanguage.googleapis.com)",
    )
    setup.add_argument(
        "--vertex-project", dest="vertex_project", default=None, help="Vertex project id (optional)"
    )
    setup.add_argument(
        "--vertex-location",
        dest="vertex_location",
        default=None,
        help="Vertex location/region (optional)",
    )
    setup.add_argument(
        "--no-verify", action="store_true", help="Skip live logprobs probe (not recommended)"
    )
    setup.add_argument(
        "--no-integrate",
        action="store_true",
        help="Do not update global client config files after saving",
    )
    setup_sub = setup.add_subparsers(dest="setup_cmd")
    setup_sub.add_parser("status", help="Show current verifier setup status")
    setup_sub.add_parser("clear", help="Clear saved verifier setup")
    setup.set_defaults(fn=cmd_setup)

    sup = sub.add_parser("support", help="Support tooling")
    sup_sub = sup.add_subparsers(dest="support_cmd", required=True)
    bundle = sup_sub.add_parser("bundle", help="Create a redacted support bundle zip")
    bundle.add_argument("--out", type=str, default=None, help="Output path (optional)")
    bundle.set_defaults(fn=cmd_support_bundle)
    issue = sup_sub.add_parser("issue", help="Create a support bundle and print an issue template")
    issue.add_argument("--out", type=str, default=None, help="Output path (optional)")
    issue.set_defaults(fn=cmd_support_issue)

    audit = sub.add_parser("audit", help="Audit log tooling")
    audit_sub = audit.add_subparsers(dest="audit_cmd", required=True)
    audit_export = audit_sub.add_parser("export", help="Export audit log as JSON")
    audit_export.add_argument("--out", type=str, required=True)
    audit_export.set_defaults(fn=cmd_audit_export)
    audit_prune = audit_sub.add_parser("prune", help="Prune audit log based on retention window")
    audit_prune.set_defaults(fn=cmd_audit_prune)

    recipes = sub.add_parser("recipes", help="Recipes system (public workflow packs)")
    recipes_sub = recipes.add_subparsers(dest="recipes_cmd", required=True)
    recipes_list = recipes_sub.add_parser("list")
    recipes_list.set_defaults(fn=cmd_recipes_list)
    recipes_export = recipes_sub.add_parser("export")
    recipes_export.add_argument("--out", type=str, required=True)
    recipes_export.set_defaults(fn=cmd_recipes_export)
    recipes_import = recipes_sub.add_parser("import", help="Import a recipe JSON file")
    recipes_import.add_argument("path")
    recipes_import.add_argument("--force", action="store_true")
    recipes_import.set_defaults(fn=cmd_recipes_import)
    recipes_install = recipes_sub.add_parser("install")
    recipes_install.add_argument("name")
    recipes_install.add_argument("--force", action="store_true")
    recipes_install.set_defaults(fn=cmd_recipes_install)

    lic = sub.add_parser("license", help="License/entitlements (paid layer scaffolding)")
    lic_sub = lic.add_subparsers(dest="license_cmd", required=True)
    lic_set = lic_sub.add_parser("set")
    lic_set.add_argument("--plan", default="pro")
    lic_set.add_argument("--features", default="")
    lic_set.set_defaults(fn=cmd_license_set)
    lic_show = lic_sub.add_parser("show")
    lic_show.set_defaults(fn=cmd_license_show)

    quick = sub.add_parser("quickstart", help="Print the fastest path to first value")
    quick.set_defaults(fn=cmd_quickstart)

    inst = sub.add_parser("instructions", help="Per-client setup copy/paste guidance")
    inst.add_argument("--client", choices=["cursor", "codex", "claude", "gemini"], default=None)
    inst.add_argument("--name", default="berry")
    inst.set_defaults(fn=cmd_instructions)

    pc = sub.add_parser("print-config", help="Print per-client config for copy/paste")
    pc.add_argument("client", choices=["cursor", "codex", "claude", "gemini"])
    pc.add_argument("--name", default="berry")
    pc.add_argument(
        "--profile",
        default="classic",
        help="Which MCP servers to render (classic). Legacy values may be accepted for older configs.",
    )
    pc.set_defaults(fn=cmd_print_config)

    dl = sub.add_parser("deeplink", help="Print a client install deeplink")
    dl.add_argument("client", choices=["cursor"])
    dl.add_argument("--name", default="berry")
    dl.add_argument(
        "--profile",
        default="classic",
        help="Which MCP server to deeplink (classic)",
    )
    dl.set_defaults(fn=cmd_deeplink)

    verify = sub.add_parser("verify", help="Verify a signed artifact (integrity verification)")
    verify.add_argument("--artifact", required=True)
    verify.add_argument("--signature", required=True)
    verify.add_argument("--public-key", default=None)
    verify.set_defaults(fn=cmd_verify)

    integ = sub.add_parser(
        "integrate",
        help="Register Berry with supported clients globally (best-effort)",
    )
    integ.add_argument(
        "--client",
        action="append",
        dest="clients",
        choices=["cursor", "claude", "codex", "gemini"],
        default=None,
        help="Client to integrate (repeatable). Defaults to all supported clients.",
    )
    integ.add_argument("--name", default="berry", help="MCP server name to register")
    integ.add_argument(
        "--timeout", type=int, default=20, help="Per-client command timeout in seconds"
    )
    integ.add_argument(
        "--dry-run", action="store_true", help="Print what would be done without modifying anything"
    )
    integ.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    # Flags reserved for installer / future expansion.
    integ.add_argument("--global", action="store_true", help="Register globally (default behavior)")
    integ.add_argument("--noninteractive", action="store_true", help="Do not prompt (reserved)")
    integ.add_argument(
        "--managed",
        action="store_true",
        help="Also write system-managed config files where supported (requires admin rights).",
    )
    integ.add_argument(
        "--managed-only",
        dest="managed_only",
        action="store_true",
        help="Only write system-managed config files (implies --managed).",
    )
    integ.set_defaults(fn=cmd_integrate)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
