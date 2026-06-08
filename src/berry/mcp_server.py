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
- extract_span
- mark_span
- list_spans
- get_span
- search_spans
- query_evidence
- get_evidence_pack
- create_claim
- link_claim_evidence
- list_claim_evidence
- list_claims
- get_claim
- mark_claim
- audit_claims
- list_audits
- detect_hallucination_run
- audit_trace_budget_run

All other tools (web, exec, repo ops, grants, microplans, verified writes, health/status, etc.)
are intentionally not registered.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import load_config
from .enforcement import EnforcementError, RunState, RunStore, SpanRecord
from .hallucination_detector.core import (
    run_audit_trace_budget,
    run_detect_hallucination,
)
from .mcp_env import load_mcp_env
from .paths import ensure_berry_home, resolve_user_path
from .permissions import can_read_path
from .prompts import list_prompts
from .run_ledger import (
    atomic_write_text as _ledger_atomic_write_text,
    attempts_tsv_path as _ledger_attempts_tsv_path,
    evidence_tsv_path as _ledger_evidence_tsv_path,
    load_persisted_run as _ledger_load_persisted_run,
    persist_run as _ledger_persist_run,
    run_dir as _ledger_run_dir,
    run_json_path as _ledger_run_json_path,
    run_sqlite_path as _ledger_run_sqlite_path,
    span_to_payload as _ledger_span_to_payload,
)


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
    return _ledger_run_dir(str(run_id))


def _run_json_path(run_id: str) -> Path:
    return _ledger_run_json_path(run_id)


def _run_sqlite_path(run_id: str) -> Path:
    return _ledger_run_sqlite_path(run_id)


def _evidence_tsv_path(run_id: str) -> Path:
    return _ledger_evidence_tsv_path(run_id)


def _attempts_tsv_path(run_id: str) -> Path:
    return _ledger_attempts_tsv_path(run_id)


def _atomic_write_text(path: Path, text: str) -> None:
    _ledger_atomic_write_text(path, text)


def _span_to_payload(rec: SpanRecord) -> Dict[str, Any]:
    return _ledger_span_to_payload(rec)


def _git_output(project_root: Optional[Path], args: List[str]) -> str:
    if not project_root:
        return ""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(project_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _git_bytes(project_root: Optional[Path], args: List[str]) -> Optional[bytes]:
    """Return exact git object bytes, preserving whitespace and empty files."""

    if not project_root:
        return None
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return bytes(proc.stdout or b"")


def _capture_git_baseline(project_root: Optional[Path]) -> Tuple[str, Optional[str]]:
    commit = _git_output(project_root, ["rev-parse", "--verify", "HEAD"])
    if commit:
        return "git", commit
    return "fs", None


def _rel_path(path: Path, project_root: Optional[Path]) -> str:
    if project_root:
        try:
            return str(path.resolve().relative_to(project_root.resolve()))
        except Exception:
            pass
    return str(path)


def _file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _decode_bytes(raw: bytes) -> Tuple[str, str]:
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def _line_byte_offsets(raw: bytes) -> List[int]:
    offsets = [0]
    for i, b in enumerate(raw):
        if b == 10:  # \n
            offsets.append(i + 1)
    offsets.append(len(raw))
    return offsets


def _extract_cited_sids(text: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for m in re.finditer(r"\[(S\d+)\]", str(text or "")):
        sid = m.group(1)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _collect_step_cites(steps: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for st in steps or []:
        if not isinstance(st, dict):
            continue
        for raw in st.get("cites") or []:
            sid = str(raw or "").strip()
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def _fail_closed_report(*, error: str, pack: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "flagged": True,
        "under_budget": True,
        "error": str(error),
        "summary": {"evidence_pack": pack or {}, "fail_closed": True},
        "details": [],
    }


def _write_evidence_tsv(run: RunState) -> None:
    path = _evidence_tsv_path(run.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(
                [
                    "ts",
                    "run_id",
                    "sid",
                    "eid",
                    "source",
                    "source_type",
                    "kind",
                    "trust",
                    "status",
                    "sensitivity",
                    "locator",
                    "parents",
                    "text_sha256",
                    "chars",
                    "preview",
                ]
            )
            for sid in run.span_order:
                rec = run.spans[sid]
                locator = ""
                if rec.locator:
                    path_val = rec.locator.get("path") or rec.locator.get("rel_path") or ""
                    start = rec.locator.get("start_line", "")
                    end = rec.locator.get("end_line", "")
                    locator = f"{path_val}:{start}-{end}" if path_val else json.dumps(rec.locator)
                writer.writerow(
                    [
                        f"{float(rec.created_at):.6f}",
                        run.run_id,
                        rec.sid,
                        rec.eid,
                        rec.source,
                        rec.source_type,
                        rec.kind,
                        rec.trust,
                        rec.status,
                        rec.sensitivity,
                        locator,
                        ",".join(rec.parents),
                        rec.text_sha256,
                        len(rec.text),
                        rec.preview(limit=200, redact_sensitive=True),
                    ]
                )
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _write_attempts_tsv(run: RunState) -> None:
    path = _attempts_tsv_path(run.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
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
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _persist_run(run: RunState) -> None:
    _ledger_persist_run(run)


def _load_persisted_run(run_id: str) -> RunState:
    return _ledger_load_persisted_run(run_id)


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
                run.baseline_kind, run.baseline_ref = _capture_git_baseline(resolved_project_root)

                # Anchor spans are immutable requirements/background, not verifier evidence.
                ps = store.add_span(
                    run=run,
                    text=str(problem_statement or "").strip(),
                    source="anchor",
                    source_type="user",
                    kind="anchor",
                    trust="manual",
                    sensitivity="unknown",
                    tags=["problem"],
                    meta={"kind": "problem", "citable": False},
                )
                dv = store.add_span(
                    run=run,
                    text=str(deliverable or "").strip(),
                    source="anchor",
                    source_type="user",
                    kind="anchor",
                    trust="manual",
                    sensitivity="unknown",
                    tags=["deliverable"],
                    meta={"kind": "deliverable", "immutable": True, "citable": False},
                )
                run.deliverable_sid = dv.sid

                # Persist
                _persist_run(run)

                return {
                    "run_id": run.run_id,
                    "run_dir": str(_run_dir(run.run_id)),
                    "ledger_path": str(_run_sqlite_path(run.run_id)),
                    "problem_sid": ps.sid,
                    "deliverable_sid": dv.sid,
                    "schema_version": 3,
                    "baseline_kind": run.baseline_kind,
                    "baseline_ref": run.baseline_ref,
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
                        "ledger_path": str(_run_sqlite_path(run.run_id)),
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
                    "ledger_path": str(_run_sqlite_path(run.run_id)),
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
        kind: str = "evidence",
        source_type: Optional[str] = None,
        trust: str = "primary",
        sensitivity: str = "unknown",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add immutable text evidence to the server-owned run ledger."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                rec = store.add_span(
                    run=run,
                    text=str(text or ""),
                    source=str(source or "manual"),
                    source_type=str(source_type or source or "manual"),
                    kind=str(kind or "evidence"),
                    trust=str(trust or "primary"),
                    sensitivity=str(sensitivity or "unknown"),
                    tags=list(tags or []),
                    meta=meta,
                )
                _persist_run(run)
                return {
                    "run_id": run.run_id,
                    "sid": rec.sid,
                    "eid": rec.eid,
                    "kind": rec.kind,
                    "trust": rec.trust,
                    "sensitivity": rec.sensitivity,
                    "chars": len(rec.text),
                    "text_sha256": rec.text_sha256,
                }
            except EnforcementError as exc:
                raise RuntimeError(str(exc))

    @mcp.tool()
    def add_file_span(
        path: str,
        start_line: int,
        end_line: int,
        source: str = "file",
        run_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        read_mode: str = "baseline",
        kind: str = "evidence",
        trust: str = "primary",
        sensitivity: str = "unknown",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Capture a file line range as immutable evidence with file/git provenance."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                p = resolve_user_path(Path(path), project_root=resolved_project_root)

                decision = can_read_path(
                    p,
                    allowed_roots=getattr(cfg, "allowed_roots", []),
                    project_root=resolved_project_root,
                )
                if not decision.allowed:
                    raise RuntimeError(f"File read not allowed: {decision.reason}")

                s_line = max(1, int(start_line))
                e_line = max(s_line, int(end_line))
                if (e_line - s_line) > 2000:
                    e_line = s_line + 2000

                rel = _rel_path(p, resolved_project_root)
                requested_mode = str(read_mode or "baseline").strip().lower()
                if requested_mode not in {"baseline", "worktree"}:
                    raise RuntimeError("read_mode must be either 'baseline' or 'worktree'")
                snapshot: Dict[str, Any] = {
                    "requested_read_mode": requested_mode,
                    "baseline_kind": run.baseline_kind,
                    "baseline_ref": run.baseline_ref,
                }

                raw: bytes
                actual_read_mode = "worktree"
                if requested_mode == "baseline" and run.baseline_kind == "git" and run.baseline_ref:
                    # Read exact object bytes from the immutable baseline commit.  This is
                    # the safest evidence mode for source files because it prevents the
                    # agent from citing code it just wrote as proof that it already existed.
                    base_raw = _git_bytes(
                        resolved_project_root, ["show", f"{run.baseline_ref}:{rel}"]
                    )
                    if base_raw is not None:
                        raw = base_raw
                        actual_read_mode = "baseline"
                        snapshot["git_object"] = f"{run.baseline_ref}:{rel}"
                    else:
                        raw = p.read_bytes()
                        snapshot["read_mode_fallback"] = "worktree_git_show_failed"
                else:
                    raw = p.read_bytes()
                    if requested_mode == "baseline":
                        snapshot["read_mode_fallback"] = "worktree_no_git_baseline"

                snapshot["actual_read_mode"] = actual_read_mode
                # Back-compat alias for older consumers that looked for read_mode.
                snapshot["read_mode"] = actual_read_mode

                text, encoding = _decode_bytes(raw)
                lines = text.splitlines()
                if s_line > len(lines):
                    raise RuntimeError(
                        f"start_line {s_line} is past end of file ({len(lines)} lines)"
                    )
                e_line = min(e_line, len(lines))
                excerpt = "\n".join(lines[s_line - 1 : e_line]).strip("\n")
                if not excerpt.strip():
                    raise RuntimeError("Selected file span is empty")

                offsets = _line_byte_offsets(raw)
                start_byte = offsets[min(s_line - 1, len(offsets) - 1)]
                end_byte = offsets[min(e_line, len(offsets) - 1)]

                file_hash = hashlib.sha256(raw).hexdigest()
                worktree_file_hash = _file_sha256(p) if p.exists() else ""
                git_commit = _git_output(resolved_project_root, ["rev-parse", "--verify", "HEAD"])
                git_status = _git_output(
                    resolved_project_root, ["status", "--porcelain", "--", rel]
                )
                snapshot.update(
                    {
                        "file_sha256": file_hash,
                        "worktree_file_sha256": worktree_file_hash,
                        "git_commit": git_commit,
                        "git_status": git_status,
                        "line_count": len(lines),
                    }
                )
                locator = {
                    "path": str(p),
                    "rel_path": rel,
                    "start_line": s_line,
                    "end_line": e_line,
                    "start_byte": start_byte,
                    "end_byte": end_byte,
                    "encoding": encoding,
                }
                m = dict(meta or {})
                m.update(locator)
                m.update(
                    {
                        "file_sha256": file_hash,
                        "worktree_file_sha256": worktree_file_hash,
                        "git_commit": git_commit,
                    }
                )

                effective_trust = str(trust or "primary")
                if actual_read_mode != "baseline" and effective_trust == "primary":
                    effective_trust = "worktree"
                if git_status and actual_read_mode != "baseline":
                    effective_trust = "worktree"
                if (
                    actual_read_mode == "baseline"
                    and worktree_file_hash
                    and worktree_file_hash != file_hash
                ):
                    snapshot["worktree_drift_from_baseline"] = True

                rec = store.add_span(
                    run=run,
                    text=excerpt,
                    source=str(source or "file"),
                    source_type="file",
                    kind=str(kind or "evidence"),
                    media_type="text/plain",
                    locator=locator,
                    snapshot=snapshot,
                    trust=effective_trust,
                    sensitivity=str(sensitivity or "unknown"),
                    tags=list(tags or []) + ["file"],
                    meta=m,
                )
                _persist_run(run)
                return {
                    "run_id": run.run_id,
                    "sid": rec.sid,
                    "eid": rec.eid,
                    "path": str(p),
                    "rel_path": rel,
                    "start_line": s_line,
                    "end_line": e_line,
                    "start_byte": start_byte,
                    "end_byte": end_byte,
                    "encoding": encoding,
                    "file_sha256": file_hash,
                    "worktree_file_sha256": worktree_file_hash,
                    "git_commit": git_commit,
                    "git_status": git_status,
                    "trust": rec.trust,
                    "sensitivity": rec.sensitivity,
                }
            except EnforcementError as exc:
                raise RuntimeError(str(exc))


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

    # -----------------------------
    # Claim / evidence graph
    # -----------------------------

    @mcp.tool()
    def create_claim(
        text: str,
        run_id: Optional[str] = None,
        kind: str = "fact",
        target: float = 0.95,
        status: str = "open",
        source: str = "manual",
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a structured claim node that can be linked to evidence and audited."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                claim = store.create_claim(
                    run=run,
                    text=text,
                    kind=kind,
                    target=float(target or 0.95),
                    status=status,
                    source=source,
                    tags=list(tags or []),
                    meta=dict(meta or {}),
                )
                _persist_run(run)
                return {"run_id": run.run_id, "claim": claim.to_public_dict()}
            except EnforcementError as exc:
                raise RuntimeError(str(exc))

    @mcp.tool()
    def link_claim_evidence(
        cid: str,
        sid: str,
        run_id: Optional[str] = None,
        relation: str = "supports",
        note: str = "",
        audit_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a typed edge between a claim and a run-owned evidence span."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                link = store.link_claim_evidence(
                    run=run,
                    cid=cid,
                    sid=sid,
                    relation=relation,
                    created_by="manual",
                    audit_id=audit_id,
                    note=note,
                    meta=dict(meta or {}),
                )
                _persist_run(run)
                return {"run_id": run.run_id, "link": link.to_public_dict()}
            except EnforcementError as exc:
                raise RuntimeError(str(exc))

    @mcp.tool()
    def list_claims(
        run_id: Optional[str] = None,
        limit: int = 200,
        status: Optional[List[str]] = None,
        kinds: Optional[List[str]] = None,
        include_evidence: bool = True,
    ) -> Dict[str, Any]:
        """List structured claims and, optionally, their evidence edges."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            return {
                "run_id": run.run_id,
                "claims": store.list_claims(
                    run=run,
                    limit=int(limit or 200),
                    status=status,
                    kinds=kinds,
                    include_evidence=bool(include_evidence),
                ),
            }

    @mcp.tool()
    def get_claim(
        cid: str,
        run_id: Optional[str] = None,
        include_evidence: bool = True,
        include_audits: bool = True,
    ) -> Dict[str, Any]:
        """Fetch a claim node with evidence and audit history."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                claim = store.get_claim(run=run, cid=cid)
                data = claim.to_public_dict()
                if include_evidence:
                    data["evidence"] = store.list_claim_evidence(run=run, cid=claim.cid)
                if include_audits:
                    data["audits"] = store.list_audits(run=run, claim_id=claim.cid)
                return {"run_id": run.run_id, "claim": data}
            except EnforcementError as exc:
                raise RuntimeError(str(exc))

    @mcp.tool()
    def mark_claim(
        cid: str,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        kind: Optional[str] = None,
        target: Optional[float] = None,
        latest_audit_id: Optional[str] = None,
        tags_add: Optional[List[str]] = None,
        tags_remove: Optional[List[str]] = None,
        meta_update: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Update mutable claim annotations without changing immutable claim text."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                meta = dict(meta_update or {})
                if reason:
                    meta["last_mark_reason"] = str(reason)
                claim = store.update_claim(
                    run=run,
                    cid=cid,
                    status=status,
                    kind=kind,
                    target=target,
                    latest_audit_id=latest_audit_id,
                    tags_add=list(tags_add or []),
                    tags_remove=list(tags_remove or []),
                    meta_update=meta,
                )
                _persist_run(run)
                return {"run_id": run.run_id, "claim": claim.to_public_dict()}
            except EnforcementError as exc:
                raise RuntimeError(str(exc))

    @mcp.tool()
    def list_claim_evidence(
        run_id: Optional[str] = None,
        cid: Optional[str] = None,
        sid: Optional[str] = None,
        relation: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """List typed claim/evidence edges with optional claim/span/relation filters."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            return {
                "run_id": run.run_id,
                "links": store.list_claim_evidence(run=run, cid=cid, sid=sid, relation=relation),
            }

    @mcp.tool()
    def list_audits(
        run_id: Optional[str] = None,
        claim_id: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List verifier audit records from the claim/evidence graph."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            return {
                "run_id": run.run_id,
                "audits": store.list_audits(run=run, claim_id=claim_id, limit=int(limit or 100)),
            }

    @mcp.tool()
    def list_spans(
        run_id: Optional[str] = None,
        limit: int = 200,
        kinds: Optional[List[str]] = None,
        source_types: Optional[List[str]] = None,
        trust: Optional[List[str]] = None,
        status: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """List span metadata with v2 provenance, lifecycle, and sensitivity fields."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            return {
                "run_id": run.run_id,
                "schema_version": 3,
                "spans": store.list_spans(
                    run=run,
                    limit=int(limit or 200),
                    kinds=kinds,
                    source_types=source_types,
                    trust=trust,
                    status=status,
                ),
            }

    @mcp.tool()
    def get_span(
        sid: str,
        run_id: Optional[str] = None,
        include_sensitive_text: bool = False,
    ) -> Dict[str, Any]:
        """Fetch a span. Sensitive text is redacted unless explicitly requested."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            try:
                rec = store.get_span(run=run, sid=sid)
            except EnforcementError as exc:
                raise RuntimeError(str(exc))
            data = rec.to_public_dict(
                include_text=True,
                preview_chars=240,
                redact_sensitive=not bool(include_sensitive_text),
            )
            data["run_id"] = run.run_id
            return data

    @mcp.tool()
    def mark_span(
        sid: str,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        sensitivity: Optional[str] = None,
        tags_add: Optional[List[str]] = None,
        tags_remove: Optional[List[str]] = None,
        trust: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update mutable span annotations without mutating immutable evidence text."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                rec = store.mark_span(
                    run=run,
                    sid=sid,
                    status=status,
                    sensitivity=sensitivity,
                    tags_add=tags_add,
                    tags_remove=tags_remove,
                    trust=trust,
                )
                _persist_run(run)
                return {"run_id": run.run_id, "span": rec.to_public_dict(include_text=False)}
            except EnforcementError as exc:
                raise RuntimeError(str(exc))

    @mcp.tool()
    def search_spans(query: str, run_id: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Backward-compatible lexical search over active non-derived spans."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            results = store.query_evidence(run=run, query=query, limit=int(limit or 10))
            return {"run_id": run.run_id, "query": query, "results": results}

    @mcp.tool()
    def query_evidence(
        query: str,
        run_id: Optional[str] = None,
        limit: int = 10,
        kinds: Optional[List[str]] = None,
        source_types: Optional[List[str]] = None,
        trust: Optional[List[str]] = None,
        status: Optional[List[str]] = None,
        include_derived: bool = False,
        include_stale: bool = False,
    ) -> Dict[str, Any]:
        """Search the evidence ledger with kind/source/trust/status filters."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            results = store.query_evidence(
                run=run,
                query=query,
                limit=int(limit or 10),
                kinds=kinds,
                source_types=source_types,
                trust=trust,
                status=status,
                include_derived=bool(include_derived),
                include_stale=bool(include_stale),
            )
            return {"run_id": run.run_id, "query": query, "results": results}

    @mcp.tool()
    def extract_span(
        parent_sid: str,
        selector: Dict[str, Any],
        run_id: Optional[str] = None,
        reason: str = "",
        source: str = "extract",
        max_lines: int = 200,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create an offset-preserving derived span from a regex or line-range selector."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                res = store.extract_span(
                    run=run,
                    parent_sid=str(parent_sid or ""),
                    selector=dict(selector or {}),
                    reason=str(reason or ""),
                    source=str(source or "extract"),
                    max_lines=int(max_lines or 200),
                    tags=list(tags or []),
                )
                if res.get("matched"):
                    _persist_run(run)
                return {"run_id": run.run_id, **res}
            except EnforcementError as exc:
                raise RuntimeError(str(exc))

    @mcp.tool()
    def distill_span(
        parent_sid: str,
        pattern: str,
        run_id: Optional[str] = None,
        source: str = "distill",
        flags: str = "i",
        max_lines: int = 200,
    ) -> Dict[str, Any]:
        """Compatibility wrapper for regex extraction. No-match creates no evidence span."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                res = store.extract_span(
                    run=run,
                    parent_sid=str(parent_sid or ""),
                    selector={"type": "regex", "pattern": str(pattern or ""), "flags": flags},
                    reason="distill_span compatibility extraction",
                    source=str(source or "distill"),
                    max_lines=int(max_lines or 200),
                    tags=["distilled"],
                )
                if res.get("matched"):
                    _persist_run(run)
                return {"run_id": run.run_id, "lines": len(res.get("matches") or []), **res}
            except EnforcementError as exc:
                raise RuntimeError(str(exc))

    @mcp.tool()
    def get_evidence_pack(
        sids: List[str],
        run_id: Optional[str] = None,
        max_chars: int = 12000,
        allow_sensitive: bool = False,
        include_stale: bool = False,
        allow_untrusted: bool = False,
    ) -> Dict[str, Any]:
        """Resolve run-owned SIDs into the exact verifier-safe evidence pack."""
        with _redirect_stdout_to_stderr():
            run = store.get_run(run_id)
            pack = store.resolve_evidence_pack(
                run=run,
                sids=list(sids or []),
                max_chars=int(max_chars or 12000),
                allow_sensitive=bool(allow_sensitive),
                include_stale=bool(include_stale),
                allow_untrusted=bool(allow_untrusted),
            )
            return pack

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

    def _pack_summary(pack: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pack_id": pack.get("pack_id"),
            "text_sha256": pack.get("text_sha256"),
            "input_sids": list(pack.get("input_sids") or []),
            "materialized_sids": list(pack.get("materialized_sids") or []),
            "excluded": list(pack.get("excluded") or []),
            "chars": int(pack.get("chars") or 0),
            "truncated": bool(pack.get("truncated")),
            "policy": dict(pack.get("policy") or {}),
        }

    def _detail_by_idx(report: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for detail in report.get("details") or []:
            if not isinstance(detail, dict):
                continue
            try:
                idx = int(detail.get("idx"))
            except Exception:
                continue
            out[idx] = detail
        return out

    def _claim_status_from_detail(detail: Dict[str, Any]) -> str:
        status = str(detail.get("status") or "").strip().lower()
        if status == "passed":
            return "supported"
        if status == "contradicted":
            return "contradicted"
        if status in {"missing_citations", "unknown_citations", "empty_context", "no_spans"}:
            return "open"
        return "insufficient"

    def _claim_relation_from_detail(detail: Dict[str, Any]) -> str:
        status = str(detail.get("status") or "").strip().lower()
        if status == "contradicted":
            return "contradicts"
        if status == "passed":
            return "supports"
        return "insufficient"

    def _claim_id_for_step(
        *, run: RunState, step: Dict[str, Any], kind: str, auto_create_claims: bool
    ) -> str:
        raw = step.get("claim_id", step.get("cid", ""))
        cid = str(raw or "").strip()
        if cid:
            store.get_claim(run=run, cid=cid)
            return cid
        if not auto_create_claims:
            return ""
        claim = store.create_claim(
            run=run,
            text=str(step.get("claim") or ""),
            kind="fact",
            target=float(step.get("confidence") or 0.95),
            status="open",
            source=kind,
            tags=["auto-created"],
            meta={"created_from_step_idx": step.get("idx")},
        )
        step["claim_id"] = claim.cid
        return claim.cid

    def _record_audit_graph(
        *,
        run: RunState,
        kind: str,
        steps: List[Dict[str, Any]],
        pack: Dict[str, Any],
        report: Dict[str, Any],
        verifier_model: str,
        audit_sid: str,
        auto_create_claims: bool = False,
    ) -> str:
        details = _detail_by_idx(report)
        claim_ids: List[str] = []
        for step in steps or []:
            if not isinstance(step, dict):
                continue
            cid = _claim_id_for_step(
                run=run,
                step=step,
                kind=kind,
                auto_create_claims=auto_create_claims,
            )
            if cid and cid not in claim_ids:
                claim_ids.append(cid)
        audit = store.record_audit(
            run=run,
            kind=kind,
            claim_ids=claim_ids,
            input_sids=list(pack.get("input_sids") or []),
            materialized_sids=list(pack.get("materialized_sids") or []),
            evidence_pack_id=str(pack.get("pack_id") or ""),
            evidence_pack_hash=str(pack.get("text_sha256") or ""),
            verifier_model=verifier_model,
            policy=dict(pack.get("policy") or {}),
            result={
                "flagged": bool(report.get("flagged", True)),
                "summary": dict(report.get("summary") or {}),
                "details_sha256": hashlib.sha256(
                    json.dumps(
                        report.get("details") or [], sort_keys=True, default=str
                    ).encode("utf-8")
                ).hexdigest(),
            },
            audit_sid=audit_sid,
        )
        materialized = {str(s) for s in (pack.get("materialized_sids") or [])}
        for i, step in enumerate(steps or []):
            if not isinstance(step, dict):
                continue
            cid = str(step.get("claim_id") or step.get("cid") or "").strip()
            if not cid:
                continue
            try:
                idx = int(step.get("idx", i))
            except Exception:
                idx = i
            detail = details.get(idx, {})
            store.update_claim(
                run=run,
                cid=cid,
                status=_claim_status_from_detail(detail),
                latest_audit_id=audit.audit_id,
                meta_update={"latest_detail_status": detail.get("status", "")},
            )
            relation = _claim_relation_from_detail(detail)
            for raw_sid in step.get("cites") or []:
                sid = str(raw_sid or "").strip()
                if not sid or sid not in materialized:
                    continue
                try:
                    store.link_claim_evidence(
                        run=run,
                        cid=cid,
                        sid=sid,
                        relation=relation,
                        created_by="audit",
                        audit_id=audit.audit_id,
                        note=str(detail.get("status") or ""),
                        meta={"detail_idx": idx},
                    )
                except EnforcementError:
                    # The pack resolver already checked materialized spans. Keep the
                    # audit record even if a concurrent annotation made one edge invalid.
                    continue
        return audit.audit_id

    def _record_audit_span(
        *,
        run: RunState,
        kind: str,
        pack: Dict[str, Any],
        report: Dict[str, Any],
        verifier_model: str,
    ) -> str:
        payload = {
            "type": kind,
            "evidence_pack": _pack_summary(pack),
            "flagged": bool(report.get("flagged", True)),
            "summary": dict(report.get("summary") or {}),
            "verifier_model": verifier_model,
            "created_at": time.time(),
        }
        # Prompts/details can be large or sensitive.  The audit span stores the
        # pack hash and summary by default, not full verifier context.
        rec = store.add_span(
            run=run,
            text=json.dumps(payload, indent=2, sort_keys=True),
            source="verifier",
            source_type="verifier",
            kind="audit",
            media_type="application/json",
            parents=list(pack.get("materialized_sids") or []),
            trust="derived",
            sensitivity="unknown",
            tags=["audit", kind],
            meta={"pack_id": pack.get("pack_id"), "verifier_model": verifier_model},
        )
        return rec.sid

    def _resolve_pack_or_error(
        *,
        run: RunState,
        sids: List[str],
        max_chars: int,
        allow_sensitive: bool,
        include_stale: bool,
        allow_untrusted: bool,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not sids:
            pack = store.resolve_evidence_pack(run=run, sids=[], max_chars=max_chars)
            return None, _fail_closed_report(
                error=(
                    "No cited SIDs were provided; server-resolved verification requires "
                    "explicit run-owned citations."
                ),
                pack=_pack_summary(pack),
            )
        pack = store.resolve_evidence_pack(
            run=run,
            sids=sids,
            max_chars=max_chars,
            allow_sensitive=allow_sensitive,
            include_stale=include_stale,
            allow_untrusted=allow_untrusted,
        )
        if pack.get("excluded"):
            return None, _fail_closed_report(
                error=(
                    "Evidence pack contains unknown, unsafe, stale, non-citable, "
                    "or over-budget spans."
                ),
                pack=_pack_summary(pack),
            )
        if not pack.get("spans"):
            return None, _fail_closed_report(
                error="Evidence pack is empty after policy filtering.", pack=_pack_summary(pack)
            )
        return pack, None

    @mcp.tool()
    def audit_trace_budget_run(
        steps: List[Dict[str, Any]],
        run_id: Optional[str] = None,
        sid_whitelist: Optional[List[str]] = None,
        verifier_model: Optional[str] = None,
        default_target: float = 0.95,
        require_citations: bool = True,
        context_mode: str = "cited",
        include_prompts: bool = False,
        max_prompt_chars: int = 3000,
        top_logprobs: int = 5,
        min_log_odds_gain: float = 0.0,
        use_cache: bool = True,
        group_claims: bool = True,
        max_group_size: int = 8,
        max_group_prompt_chars: int = 24000,
        timeout_s: float = 60.0,
        max_evidence_chars: int = 12000,
        allow_sensitive: bool = False,
        include_stale: bool = False,
        allow_untrusted: bool = False,
        record_audit: bool = True,
        record_claim_graph: bool = True,
        auto_create_claims: bool = False,
    ) -> Dict[str, Any]:
        """Audit explicit claims using server-resolved run spans, not caller-supplied text."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                step_list = list(steps or [])
                cited = _collect_step_cites(step_list)
                if bool(require_citations):
                    missing = [
                        int(st.get("idx", i))
                        for i, st in enumerate(step_list)
                        if not (st.get("cites") or [])
                    ]
                    if missing:
                        return _fail_closed_report(
                            error=f"Steps missing citations: {missing}",
                            pack={"input_sids": cited, "materialized_sids": []},
                        )
                if sid_whitelist:
                    allowed = {str(s).strip() for s in sid_whitelist if str(s).strip()}
                    outside = [sid for sid in cited if sid not in allowed]
                    if outside:
                        return _fail_closed_report(
                            error=f"Steps cite SIDs outside sid_whitelist: {outside}",
                            pack={"input_sids": cited, "materialized_sids": []},
                        )
                    pack_sids = cited or list(allowed)
                else:
                    pack_sids = cited

                pack, err = _resolve_pack_or_error(
                    run=run,
                    sids=pack_sids,
                    max_chars=int(max_evidence_chars or 12000),
                    allow_sensitive=bool(allow_sensitive),
                    include_stale=bool(include_stale),
                    allow_untrusted=bool(allow_untrusted),
                )
                if err:
                    return err
                assert pack is not None

                model = str(verifier_model or _default_verifier_model())
                try:
                    report = run_audit_trace_budget(
                        steps=step_list,
                        spans=list(pack.get("spans") or []),
                        verifier_model=model,
                        default_target=float(default_target or 0.95),
                        require_citations=bool(require_citations),
                        context_mode=str(context_mode or "cited"),
                        include_prompts=bool(include_prompts),
                        max_prompt_chars=int(max_prompt_chars or 3000),
                        top_logprobs=int(top_logprobs or 5),
                        min_log_odds_gain=float(min_log_odds_gain),
                        use_cache=bool(use_cache),
                        group_claims=bool(group_claims),
                        max_group_size=max_group_size,
                        max_group_prompt_chars=max_group_prompt_chars,
                        timeout_s=float(timeout_s or 60.0),
                    )
                    report.setdefault("summary", {})["evidence_pack"] = _pack_summary(pack)
                except Exception as exc:
                    report = _fail_closed_report(error=str(exc), pack=_pack_summary(pack))
                if bool(record_audit):
                    audit_sid = _record_audit_span(
                        run=run,
                        kind="audit_trace_budget_run",
                        pack=pack,
                        report=report,
                        verifier_model=model,
                    )
                    report["summary"]["audit_sid"] = audit_sid
                    if bool(record_claim_graph):
                        report["summary"]["audit_id"] = _record_audit_graph(
                            run=run,
                            kind="audit_trace_budget_run",
                            steps=step_list,
                            pack=pack,
                            report=report,
                            verifier_model=model,
                            audit_sid=audit_sid,
                            auto_create_claims=bool(auto_create_claims),
                        )
                    _persist_run(run)
                return report
            except Exception as e:
                return {"flagged": True, "under_budget": True, "error": str(e), "details": []}

    @mcp.tool()
    def audit_claims(
        claim_ids: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        evidence_relations: Optional[List[str]] = None,
        verifier_model: Optional[str] = None,
        context_mode: str = "cited",
        include_prompts: bool = False,
        max_prompt_chars: int = 3000,
        top_logprobs: int = 5,
        min_log_odds_gain: float = 0.0,
        use_cache: bool = True,
        group_claims: bool = True,
        max_group_size: int = 8,
        max_group_prompt_chars: int = 24000,
        timeout_s: float = 60.0,
        max_evidence_chars: int = 12000,
        allow_sensitive: bool = False,
        include_stale: bool = False,
        allow_untrusted: bool = False,
        max_claims: int = 25,
        record_audit: bool = True,
    ) -> Dict[str, Any]:
        """Audit claims from the structured claim/evidence graph."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                step_list = store.claim_steps(
                    run=run,
                    claim_ids=list(claim_ids or []),
                    max_claims=int(max_claims or 25),
                    evidence_relations=list(evidence_relations or ["supports", "background"]),
                )
                if not step_list:
                    return _fail_closed_report(
                        error="No auditable claims found. Create claims and link evidence first.",
                        pack={"input_sids": [], "materialized_sids": []},
                    )
                missing = [st["claim_id"] for st in step_list if not st.get("cites")]
                if missing:
                    return _fail_closed_report(
                        error=f"Claims have no linked evidence: {missing}",
                        pack={"input_sids": [], "materialized_sids": []},
                    )
                cited = _collect_step_cites(step_list)
                pack, err = _resolve_pack_or_error(
                    run=run,
                    sids=cited,
                    max_chars=int(max_evidence_chars or 12000),
                    allow_sensitive=bool(allow_sensitive),
                    include_stale=bool(include_stale),
                    allow_untrusted=bool(allow_untrusted),
                )
                if err:
                    return err
                assert pack is not None

                model = str(verifier_model or _default_verifier_model())
                try:
                    report = run_audit_trace_budget(
                        steps=step_list,
                        spans=list(pack.get("spans") or []),
                        verifier_model=model,
                        default_target=0.95,
                        require_citations=True,
                        context_mode=str(context_mode or "cited"),
                        include_prompts=bool(include_prompts),
                        max_prompt_chars=int(max_prompt_chars or 3000),
                        top_logprobs=int(top_logprobs or 5),
                        min_log_odds_gain=float(min_log_odds_gain),
                        use_cache=bool(use_cache),
                        group_claims=bool(group_claims),
                        max_group_size=max_group_size,
                        max_group_prompt_chars=max_group_prompt_chars,
                        timeout_s=float(timeout_s or 60.0),
                    )
                    report.setdefault("summary", {})["evidence_pack"] = _pack_summary(pack)
                    report["summary"]["claim_ids"] = [st["claim_id"] for st in step_list]
                except Exception as exc:
                    report = _fail_closed_report(error=str(exc), pack=_pack_summary(pack))
                if bool(record_audit):
                    audit_sid = _record_audit_span(
                        run=run,
                        kind="audit_claims",
                        pack=pack,
                        report=report,
                        verifier_model=model,
                    )
                    report.setdefault("summary", {})["audit_sid"] = audit_sid
                    report["summary"]["audit_id"] = _record_audit_graph(
                        run=run,
                        kind="audit_claims",
                        steps=step_list,
                        pack=pack,
                        report=report,
                        verifier_model=model,
                        audit_sid=audit_sid,
                    )
                    _persist_run(run)
                return report
            except Exception as e:
                return {"flagged": True, "under_budget": True, "error": str(e), "details": []}

    @mcp.tool()
    def detect_hallucination_run(
        answer: str,
        run_id: Optional[str] = None,
        sid_whitelist: Optional[List[str]] = None,
        verifier_model: Optional[str] = None,
        default_target: float = 0.95,
        max_claims: int = 25,
        claim_split: str = "sentences",
        require_citations: bool = True,
        context_mode: str = "cited",
        include_prompts: bool = False,
        max_prompt_chars: int = 3000,
        top_logprobs: int = 5,
        min_log_odds_gain: float = 0.0,
        use_cache: bool = True,
        group_claims: bool = True,
        max_group_size: int = 8,
        max_group_prompt_chars: int = 24000,
        timeout_s: float = 60.0,
        max_evidence_chars: int = 12000,
        allow_sensitive: bool = False,
        include_stale: bool = False,
        allow_untrusted: bool = False,
        record_audit: bool = True,
        record_claim_graph: bool = True,
    ) -> Dict[str, Any]:
        """Detect answer hallucinations using citations resolved from the server-owned run."""
        with _redirect_stdout_to_stderr():
            try:
                run = store.get_run(run_id)
                cited = _extract_cited_sids(str(answer or ""))
                if bool(require_citations) and not cited:
                    return _fail_closed_report(
                        error=(
                            "No [S#] citations found in answer; server-resolved "
                            "detection requires explicit citations."
                        ),
                        pack={"input_sids": [], "materialized_sids": []},
                    )
                if sid_whitelist:
                    allowed = {str(s).strip() for s in sid_whitelist if str(s).strip()}
                    outside = [sid for sid in cited if sid not in allowed]
                    if outside:
                        return _fail_closed_report(
                            error=f"Answer cites SIDs outside sid_whitelist: {outside}",
                            pack={"input_sids": cited, "materialized_sids": []},
                        )
                    pack_sids = cited or list(allowed)
                else:
                    pack_sids = cited

                pack, err = _resolve_pack_or_error(
                    run=run,
                    sids=pack_sids,
                    max_chars=int(max_evidence_chars or 12000),
                    allow_sensitive=bool(allow_sensitive),
                    include_stale=bool(include_stale),
                    allow_untrusted=bool(allow_untrusted),
                )
                if err:
                    return err
                assert pack is not None

                model = str(verifier_model or _default_verifier_model())
                try:
                    report = run_detect_hallucination(
                        answer=str(answer or ""),
                        spans=list(pack.get("spans") or []),
                        verifier_model=model,
                        default_target=float(default_target or 0.95),
                        max_claims=int(max_claims or 25),
                        claim_split=str(claim_split or "sentences"),
                        require_citations=bool(require_citations),
                        context_mode=str(context_mode or "cited"),
                        include_prompts=bool(include_prompts),
                        max_prompt_chars=int(max_prompt_chars or 3000),
                        top_logprobs=int(top_logprobs or 5),
                        min_log_odds_gain=float(min_log_odds_gain),
                        use_cache=bool(use_cache),
                        group_claims=bool(group_claims),
                        max_group_size=max_group_size,
                        max_group_prompt_chars=max_group_prompt_chars,
                        timeout_s=float(timeout_s or 60.0),
                    )
                    report.setdefault("summary", {})["evidence_pack"] = _pack_summary(pack)
                except Exception as exc:
                    report = _fail_closed_report(error=str(exc), pack=_pack_summary(pack))
                if bool(record_audit):
                    audit_sid = _record_audit_span(
                        run=run,
                        kind="detect_hallucination_run",
                        pack=pack,
                        report=report,
                        verifier_model=model,
                    )
                    report["summary"]["audit_sid"] = audit_sid
                    if bool(record_claim_graph):
                        report["summary"]["audit_id"] = _record_audit_graph(
                            run=run,
                            kind="detect_hallucination_run",
                            steps=[],
                            pack=pack,
                            report=report,
                            verifier_model=model,
                            audit_sid=audit_sid,
                        )
                    _persist_run(run)
                return report
            except Exception as e:
                return {"flagged": True, "under_budget": True, "error": str(e), "details": []}

    @mcp.tool()
    def detect_hallucination(
        answer: str,
        spans: List[Dict[str, str]],
        verifier_model: Optional[str] = None,
        default_target: float = 0.95,
        max_claims: int = 25,
        claim_split: str = "sentences",
        require_citations: bool = False,
        context_mode: str = "cited",
        include_prompts: bool = False,
        max_prompt_chars: int = 3000,
        top_logprobs: int = 5,
        min_log_odds_gain: float = 0.0,
        use_cache: bool = True,
        group_claims: bool = True,
        max_group_size: int = 8,
        max_group_prompt_chars: int = 24000,
        timeout_s: float = 60.0,
    ) -> Dict[str, Any]:
        """Information-budget diagnostic per claim."""
        with _redirect_stdout_to_stderr():
            try:
                return run_detect_hallucination(
                    answer=str(answer or ""),
                    spans=list(spans or []),
                    verifier_model=str(verifier_model or _default_verifier_model()),
                    default_target=float(default_target or 0.95),
                    max_claims=int(max_claims or 25),
                    claim_split=str(claim_split or "sentences"),
                    require_citations=bool(require_citations),
                    context_mode=str(context_mode or "cited"),
                    include_prompts=bool(include_prompts),
                    max_prompt_chars=int(max_prompt_chars or 3000),
                    top_logprobs=int(top_logprobs or 5),
                    min_log_odds_gain=float(min_log_odds_gain),
                    use_cache=bool(use_cache),
                    group_claims=bool(group_claims),
                    max_group_size=max_group_size,
                    max_group_prompt_chars=max_group_prompt_chars,
                    timeout_s=float(timeout_s or 60.0),
                )
            except Exception as e:
                return {"flagged": True, "under_budget": True, "error": str(e), "details": []}

    @mcp.tool()
    def audit_trace_budget(
        steps: List[Dict[str, Any]],
        spans: List[Dict[str, str]],
        verifier_model: Optional[str] = None,
        default_target: float = 0.95,
        require_citations: bool = False,
        context_mode: str = "cited",
        include_prompts: bool = False,
        max_prompt_chars: int = 3000,
        top_logprobs: int = 5,
        min_log_odds_gain: float = 0.0,
        use_cache: bool = True,
        group_claims: bool = True,
        max_group_size: int = 8,
        max_group_prompt_chars: int = 24000,
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
                    include_prompts=bool(include_prompts),
                    max_prompt_chars=int(max_prompt_chars or 3000),
                    top_logprobs=int(top_logprobs or 5),
                    min_log_odds_gain=float(min_log_odds_gain),
                    use_cache=bool(use_cache),
                    group_claims=bool(group_claims),
                    max_group_size=max_group_size,
                    max_group_prompt_chars=max_group_prompt_chars,
                    timeout_s=float(timeout_s or 60.0),
                )
            except Exception as e:
                return {"flagged": True, "under_budget": True, "error": str(e), "details": []}

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
