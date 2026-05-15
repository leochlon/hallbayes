"""Berry Classic MCP server (approved surface only).

This file replaces the previous broad-surface server.

Approved MCP tools only:
- detect_hallucination
- audit_trace_budget
- start_run
- load_run
- get_deliverable
- add_span
- add_file_span
- record_attempt
- list_attempts
- distill_span
- list_spans
- get_span
- search_spans

All other tools (web, exec, repo ops, grants, microplans, verified writes, health/status, etc.)
are intentionally not registered.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NotRequired, Optional, Tuple, TypedDict


class SpanInput(TypedDict):
    sid: str
    text: str
    snippet: NotRequired[str]
    url: NotRequired[str]
    title: NotRequired[str]


from .config import load_config
from .enforcement import AttemptRecord, EnforcementError, RunState, RunStore, SpanRecord
from .hallucination_detector.core import (
    run_audit_trace_budget,
    run_detect_hallucination,
)
from .mcp_env import load_mcp_env
from .paths import ensure_berry_home, resolve_user_path
from .permissions import can_read_path
from .prompts import list_prompts


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    # stdio transport: never write to stdout except JSON-RPC frames
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _berry_home() -> Path:
    return ensure_berry_home()


def _runs_dir() -> Path:
    d = _berry_home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_dir(run_id: str) -> Path:
    d = _runs_dir() / str(run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_json_path(run_id: str) -> Path:
    return _run_dir(run_id) / "run.json"


def _evidence_tsv_path(run_id: str) -> Path:
    return _run_dir(run_id) / "evidence.tsv"


def _attempts_tsv_path(run_id: str) -> Path:
    return _run_dir(run_id) / "attempts.tsv"


def _write_evidence_tsv(run: RunState) -> None:
    with _evidence_tsv_path(run.run_id).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["ts", "run_id", "sid", "source", "kind", "locator", "chars", "preview"])
        for sid in run.span_order:
            rec = run.spans[sid]
            meta = dict(rec.meta or {})
            locator = ""
            if meta.get("path"):
                locator = (
                    f"{meta.get('path')}:{meta.get('start_line', '')}-{meta.get('end_line', '')}"
                )
            writer.writerow(
                [
                    f"{float(rec.created_at):.6f}",
                    run.run_id,
                    rec.sid,
                    rec.source,
                    str(meta.get("kind") or ""),
                    locator,
                    len(rec.text),
                    rec.text.strip().replace("\n", " ")[:200],
                ]
            )


def _write_attempts_tsv(run: RunState) -> None:
    with _attempts_tsv_path(run.run_id).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "ts",
                "run_id",
                "attempt_id",
                "claim_id",
                "hypothesis",
                "action",
                "budget_minutes",
                "input_sids",
                "output_sids",
                "audit_status",
                "decision",
                "next_step",
            ]
        )
        for rec in run.attempts:
            writer.writerow(
                [
                    f"{float(rec.created_at):.6f}",
                    run.run_id,
                    rec.attempt_id,
                    rec.claim_id,
                    rec.hypothesis,
                    rec.action,
                    f"{float(rec.budget_minutes):.2f}",
                    ",".join(rec.input_sids),
                    ",".join(rec.output_sids),
                    rec.audit_status,
                    rec.decision,
                    rec.next_step,
                ]
            )


def _persist_run(run: RunState) -> None:
    try:
        payload = {
            "run_id": run.run_id,
            "created_at": float(run.created_at),
            "deliverable_sid": run.deliverable_sid,
            "next_span_idx": int(run.next_span_idx),
            "next_attempt_idx": int(run.next_attempt_idx),
            "span_order": list(run.span_order),
            "spans": {
                sid: {
                    "sid": rec.sid,
                    "text": rec.text,
                    "source": rec.source,
                    "created_at": float(rec.created_at),
                    "meta": dict(rec.meta or {}),
                }
                for sid, rec in run.spans.items()
            },
            "attempts": [
                {
                    "attempt_id": rec.attempt_id,
                    "created_at": float(rec.created_at),
                    "claim_id": rec.claim_id,
                    "hypothesis": rec.hypothesis,
                    "action": rec.action,
                    "budget_minutes": float(rec.budget_minutes),
                    "input_sids": list(rec.input_sids),
                    "output_sids": list(rec.output_sids),
                    "audit_status": rec.audit_status,
                    "decision": rec.decision,
                    "next_step": rec.next_step,
                }
                for rec in run.attempts
            ],
        }
        _run_json_path(run.run_id).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_evidence_tsv(run)
        _write_attempts_tsv(run)
    except Exception:
        pass


def _load_persisted_run(run_id: str) -> RunState:
    p = _run_json_path(run_id)
    raw = json.loads(p.read_text(encoding="utf-8"))
    rid = str(raw.get("run_id") or run_id).strip()
    if not rid:
        raise EnforcementError("Invalid persisted run: missing run_id")

    run = RunState(run_id=rid, created_at=float(raw.get("created_at") or time.time()))
    run.deliverable_sid = raw.get("deliverable_sid")
    run.next_span_idx = int(raw.get("next_span_idx") or 0)
    run.next_attempt_idx = int(raw.get("next_attempt_idx") or 0)

    spans_raw = raw.get("spans") or {}
    if isinstance(spans_raw, dict):
        for sid, rec in spans_raw.items():
            if not isinstance(rec, dict):
                continue
            s = SpanRecord(
                sid=str(rec.get("sid") or sid),
                text=str(rec.get("text") or ""),
                source=str(rec.get("source") or "manual"),
                created_at=float(rec.get("created_at") or time.time()),
                meta=dict(rec.get("meta") or {}),
            )
            if s.text.strip():
                run.spans[s.sid] = s

    order = raw.get("span_order")
    if isinstance(order, list):
        run.span_order = [str(x) for x in order if str(x) in run.spans]
    else:
        # fallback: stable order by sid
        run.span_order = sorted(run.spans.keys())

    # Ensure next_span_idx is at least one more than the largest S#
    try:
        max_idx = -1
        for sid in run.spans.keys():
            if sid.startswith("S") and sid[1:].isdigit():
                max_idx = max(max_idx, int(sid[1:]))
        run.next_span_idx = max(run.next_span_idx, max_idx + 1)
    except Exception:
        pass

    attempts_raw = raw.get("attempts") or []
    if isinstance(attempts_raw, list):
        for rec in attempts_raw:
            if not isinstance(rec, dict):
                continue
            run.attempts.append(
                AttemptRecord(
                    attempt_id=str(rec.get("attempt_id") or f"A{len(run.attempts)}"),
                    created_at=float(rec.get("created_at") or time.time()),
                    claim_id=str(rec.get("claim_id") or ""),
                    hypothesis=str(rec.get("hypothesis") or ""),
                    action=str(rec.get("action") or ""),
                    budget_minutes=float(rec.get("budget_minutes") or 0.0),
                    input_sids=[
                        str(v).strip() for v in (rec.get("input_sids") or []) if str(v).strip()
                    ],
                    output_sids=[
                        str(v).strip() for v in (rec.get("output_sids") or []) if str(v).strip()
                    ],
                    audit_status=str(rec.get("audit_status") or ""),
                    decision=str(rec.get("decision") or ""),
                    next_step=str(rec.get("next_step") or ""),
                )
            )

    return run


def _tokenize(q: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", (q or "").lower()) if t]


def _score_text(text: str, tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    t = (text or "").lower()
    score = 0.0
    for tok in tokens:
        if not tok:
            continue
        # simple frequency scoring; good enough for span search
        score += t.count(tok)
    return float(score)


def create_server(
    *, project_root: Optional[Path], host: str = "127.0.0.1", port: int = 8000
) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MCP SDK not installed. Run: pip install 'mcp[cli]'") from exc

    mcp = FastMCP("berry", json_response=True, host=host, port=port)

    resolved_project_root = Path(project_root).resolve() if project_root else None
    cfg = load_config(project_root=resolved_project_root)

    # Apply optional env defaults for MCP launches (e.g., OPENAI_BASE_URL / OPENAI_API_KEY).
    # Do not override explicitly set process env.
    try:
        for k, v in (load_mcp_env() or {}).items():
            if k and v and os.environ.get(k) in {None, ""}:
                os.environ[str(k)] = str(v)
    except Exception:
        pass

    store = RunStore()

    # -----------------------------
    # Prompts (workflow skills)
    # -----------------------------

    for _p in list_prompts():
        # Closure capture: bind prompt to avoid late-binding issues
        def _make_prompt_fn(prompt):
            @mcp.prompt(name=prompt.name, description=prompt.description)
            def _prompt_fn():
                return prompt.template

            return _prompt_fn

        _make_prompt_fn(_p)

    # -----------------------------
    # Run management
    # -----------------------------

    @mcp.tool()
    def start_run(
        problem_statement: str,
        deliverable: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new run directory with a problem statement + immutable deliverable anchor."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.start_run(run_id=run_id)

                # Anchor spans (not evidence, but captures intent).
                ps = store.add_span(
                    run=run,
                    text=str(problem_statement or "").strip(),
                    source="anchor",
                    meta={"kind": "problem"},
                )
                dv = store.add_span(
                    run=run,
                    text=str(deliverable or "").strip(),
                    source="anchor",
                    meta={"kind": "deliverable", "immutable": True},
                )
                run.deliverable_sid = dv.sid

                # Persist
                _persist_run(run)

                return {
                    "run_id": run.run_id,
                    "run_dir": str(_run_dir(run.run_id)),
                    "problem_sid": ps.sid,
                    "deliverable_sid": dv.sid,
                }
            except EnforcementError as exc:
                raise RuntimeError(str(exc))

    @mcp.tool()
    def load_run(run_id: str) -> Dict[str, Any]:
        """Resume an existing run (loads from disk if necessary) and set it active."""
        with _redirect_stdout_to_stderr():
            rid = str(run_id or "").strip()
            if not rid:
                raise RuntimeError("run_id is required")
            try:
                # If it's already in memory, just set active.
                try:
                    run = store.set_active_run(rid)
                    return {
                        "run_id": run.run_id,
                        "run_dir": str(_run_dir(run.run_id)),
                        "status": "active",
                    }
                except Exception:
                    pass

                run = _load_persisted_run(rid)
                # Install into store.
                store._runs[rid] = run
                store._active_run_id = rid
                return {
                    "run_id": run.run_id,
                    "run_dir": str(_run_dir(run.run_id)),
                    "status": "loaded",
                }
            except FileNotFoundError:
                raise RuntimeError(f"No persisted run found for run_id={rid}")
            except Exception as exc:
                raise RuntimeError(f"Failed to load run: {type(exc).__name__}: {exc}")

    @mcp.tool()
    def get_deliverable(run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the immutable deliverable anchor for the active run."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            sid = run.deliverable_sid
            if not sid or sid not in run.spans:
                raise RuntimeError("No deliverable anchor set for this run (call start_run).")
            rec = run.spans[sid]
            return {
                "run_id": run.run_id,
                "deliverable_sid": rec.sid,
                "text": rec.text,
                "meta": rec.meta,
            }

    # -----------------------------
    # Evidence spans
    # -----------------------------

    @mcp.tool()
    def add_span(
        text: str,
        source: str = "manual",
        run_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add evidence from text."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            rec = store.add_span(
                run=run, text=str(text or ""), source=str(source or "manual"), meta=meta
            )
            _persist_run(run)
            return {"run_id": run.run_id, "sid": rec.sid, "chars": len(rec.text)}

    @mcp.tool()
    def add_file_span(
        path: str,
        start_line: int,
        end_line: int,
        source: str = "file",
        run_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Capture lines from a local file (path + line range) as evidence."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            p = resolve_user_path(Path(path), project_root=resolved_project_root)

            decision = can_read_path(
                p,
                allowed_roots=getattr(cfg, "allowed_roots", []),
                project_root=resolved_project_root,
            )
            if not decision.allowed:
                raise RuntimeError(f"File read not allowed: {decision.reason}")

            s = max(1, int(start_line))
            e = max(s, int(end_line))
            # cap to prevent huge spans
            if (e - s) > 2000:
                e = s + 2000

            try:
                lines = p.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                lines = p.read_text(encoding="latin-1").splitlines()

            excerpt = "\n".join(lines[s - 1 : e])
            m = dict(meta or {})
            m.update({"path": str(p), "start_line": s, "end_line": e})

            rec = store.add_span(run=run, text=excerpt, source=str(source or "file"), meta=m)
            _persist_run(run)
            return {
                "run_id": run.run_id,
                "sid": rec.sid,
                "path": str(p),
                "start_line": s,
                "end_line": e,
            }

    @mcp.tool()
    def record_attempt(
        claim_id: str,
        action: str,
        hypothesis: str = "",
        budget_minutes: float = 5.0,
        input_sids: Optional[List[str]] = None,
        output_sids: Optional[List[str]] = None,
        audit_status: str = "",
        decision: str = "",
        next_step: str = "",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a structured attempt row for the current run."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                rec = store.record_attempt(
                    run=run,
                    claim_id=claim_id,
                    hypothesis=hypothesis,
                    action=action,
                    budget_minutes=budget_minutes,
                    input_sids=input_sids,
                    output_sids=output_sids,
                    audit_status=audit_status,
                    decision=decision,
                    next_step=next_step,
                )
                _persist_run(run)
                return {
                    "run_id": run.run_id,
                    "attempt_id": rec.attempt_id,
                    "claim_id": rec.claim_id,
                    "decision": rec.decision,
                }
            except EnforcementError as exc:
                raise RuntimeError(str(exc))

    @mcp.tool()
    def list_attempts(run_id: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
        """List recorded attempts for the active run."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            return {
                "run_id": run.run_id,
                "attempts": store.list_attempts(run=run, limit=int(limit or 200)),
            }

    @mcp.tool()
    def list_spans(run_id: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
        """List all spans (metadata only)."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            return {
                "run_id": run.run_id,
                "spans": store.list_spans(run=run, limit=int(limit or 200)),
            }

    @mcp.tool()
    def get_span(sid: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch full span text."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            k = str(sid or "").strip()
            if k not in run.spans:
                raise RuntimeError(f"Unknown span sid: {k}")
            rec = run.spans[k]
            return {
                "run_id": run.run_id,
                "sid": rec.sid,
                "text": rec.text,
                "source": rec.source,
                "created_at": rec.created_at,
                "meta": rec.meta,
            }

    @mcp.tool()
    def search_spans(query: str, run_id: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Search over span texts (lightweight token match scoring)."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            tokens = _tokenize(query)
            scored: List[Tuple[float, SpanRecord]] = []
            for sid in run.span_order:
                rec = run.spans[sid]
                s = _score_text(rec.text, tokens)
                if s > 0:
                    scored.append((s, rec))
            scored.sort(key=lambda x: (-x[0], x[1].sid))
            out = []
            for score, rec in scored[: max(1, int(limit or 10))]:
                preview = rec.text.strip().replace("\n", " ")[:200]
                out.append(
                    {
                        "sid": rec.sid,
                        "score": score,
                        "preview": preview,
                        "source": rec.source,
                        "meta": rec.meta,
                    }
                )
            return {"run_id": run.run_id, "query": query, "results": out}

    @mcp.tool()
    def distill_span(
        parent_sid: str,
        pattern: str,
        run_id: Optional[str] = None,
        source: str = "distill",
        flags: str = "i",
        max_lines: int = 200,
    ) -> Dict[str, Any]:
        """Extract key lines from a large span (regex-based), creating a new span."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            psid = str(parent_sid or "").strip()
            if psid not in run.spans:
                raise RuntimeError(f"Unknown parent span sid: {psid}")

            fl = 0
            if "i" in (flags or ""):
                fl |= re.IGNORECASE
            if "m" in (flags or ""):
                fl |= re.MULTILINE

            try:
                rx = re.compile(str(pattern or ""), fl)
            except Exception as exc:
                raise RuntimeError(f"Invalid regex pattern: {exc}")

            parent = run.spans[psid]
            matched: List[str] = []
            for line in parent.text.splitlines():
                if rx.search(line):
                    matched.append(line)
                    if len(matched) >= max(1, int(max_lines or 200)):
                        break

            distilled = "\n".join(matched).strip()
            if not distilled:
                distilled = "[no lines matched]"

            meta = {
                "parent_sid": psid,
                "pattern": pattern,
                "flags": flags,
                "max_lines": int(max_lines or 200),
            }
            rec = store.add_span(
                run=run, text=distilled, source=str(source or "distill"), meta=meta
            )
            _persist_run(run)
            return {"run_id": run.run_id, "sid": rec.sid, "parent_sid": psid, "lines": len(matched)}

    # -----------------------------
    # Verification tools (local core implementation)
    # -----------------------------
    # These tools run locally and rely on the configured verifier backend.

    def _default_verifier_model() -> str:
        return (
            (os.environ.get("BERRY_VERIFIER_MODEL") or "").strip()
            or (os.environ.get("BERRY_MODEL") or "").strip()
            or "gpt-4o-mini"
        )

    @mcp.tool()
    def detect_hallucination(
        answer: str,
        spans: List[SpanInput],
        verifier_model: Optional[str] = None,
        default_target: float = 0.95,
        max_claims: int = 25,
        require_citations: bool = False,
        context_mode: str = "cited",
        timeout_s: float = 60.0,
    ) -> Dict[str, Any]:
        """Information-budget diagnostic per claim.

        spans must each have ``sid`` (string id) and ``text`` (citation content).
        """
        with _redirect_stdout_to_stderr():
            try:
                return run_detect_hallucination(
                    answer=str(answer or ""),
                    spans=list(spans or []),
                    verifier_model=str(verifier_model or _default_verifier_model()),
                    default_target=float(default_target or 0.95),
                    max_claims=int(max_claims or 25),
                    require_citations=bool(require_citations),
                    context_mode=str(context_mode or "cited"),
                    timeout_s=float(timeout_s or 60.0),
                )
            except ValueError as e:
                return {"flagged": True, "under_budget": True, "error": str(e), "error_type": "malformed_input", "details": []}
            except Exception as e:
                return {"flagged": True, "under_budget": True, "error": str(e), "error_type": "internal", "details": []}

    @mcp.tool()
    def audit_trace_budget(
        steps: List[Dict[str, Any]],
        spans: List[SpanInput],
        verifier_model: Optional[str] = None,
        default_target: float = 0.95,
        require_citations: bool = False,
        context_mode: str = "cited",
        timeout_s: float = 60.0,
    ) -> Dict[str, Any]:
        """Score explicit (claim, cites) steps."""
        with _redirect_stdout_to_stderr():
            try:
                return run_audit_trace_budget(
                    steps=list(steps or []),
                    spans=list(spans or []),
                    verifier_model=str(verifier_model or _default_verifier_model()),
                    default_target=float(default_target or 0.95),
                    require_citations=bool(require_citations),
                    context_mode=str(context_mode or "cited"),
                    timeout_s=float(timeout_s or 60.0),
                )
            except ValueError as e:
                return {"flagged": True, "under_budget": True, "error": str(e), "error_type": "malformed_input", "details": []}
            except Exception as e:
                return {"flagged": True, "under_budget": True, "error": str(e), "error_type": "internal", "details": []}

    return mcp


def _find_repo_root(start: Path) -> Path:
    p = Path(start).resolve()
    for _ in range(50):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(start).resolve()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="berry mcp")
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio", "sse", "streamable-http"]
    )  # keep parity
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--server", default="classic")  # kept for compatibility; ignored
    args = parser.parse_args(argv)

    project_root_raw = str(args.project_root or "").strip()
    if not project_root_raw:
        project_root_raw = (os.environ.get("BERRY_PROJECT_ROOT") or "").strip()

    project_root: Optional[Path]
    if project_root_raw:
        project_root = Path(project_root_raw).expanduser().resolve()
    else:
        inferred = _find_repo_root(Path.cwd())
        allow_non_git = (os.environ.get("BERRY_ALLOW_NON_GIT_ROOT") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if (inferred / ".git").exists() or allow_non_git:
            project_root = inferred
        else:
            project_root = None

    mcp = create_server(project_root=project_root, host=str(args.host), port=int(args.port))
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
