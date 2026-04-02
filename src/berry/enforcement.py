from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class EnforcementError(Exception):
    """User-facing policy/gating errors.

    Keep this as a normal Exception (not a frozen dataclass) to avoid surprising
    interactions with frameworks that mutate exception attributes.
    """

    def __init__(self, message: str):
        super().__init__(str(message))
        self.message = str(message)

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class SpanRecord:
    sid: str
    text: str
    source: str
    created_at: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MicroplanStep:
    idx: int
    claim: str
    cites: List[str]
    confidence: float


@dataclass(frozen=True)
class PlanAudit:
    spans_version: int
    ok: bool
    report: Dict[str, Any]
    audited_at: float


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    created_at: float
    claim_id: str
    hypothesis: str
    action: str
    budget_minutes: float
    input_sids: List[str] = field(default_factory=list)
    output_sids: List[str] = field(default_factory=list)
    audit_status: str = ""
    decision: str = ""
    git_state: str = ""
    objective_metric: str = ""
    objective_value: str = ""
    result_summary: str = ""
    next_step: str = ""


@dataclass(frozen=True)
class PendingWrite:
    token: str
    path: str
    contents: str
    # Verified write intent + justification (strictly audited).
    change_summary: str
    motivation: str
    # "verified" (default) or "exploratory" (explicit, less strict, requires confirm on apply).
    mode: str
    # Back-compat: keep the old rationale field for older clients, but do not rely on it.
    rationale: str
    pre_image_sha256: str
    diff_sid: str
    created_at: float
    expires_at: float
    verification: Dict[str, Any]
    step_idx: int


@dataclass(frozen=True)
class PendingGrant:
    """A pending approval request for one or more permission scopes."""

    token: str
    scopes: List[str]
    summary: str
    created_at: float
    expires_at: float


@dataclass
class RunState:
    run_id: str
    created_at: float
    spans: Dict[str, SpanRecord] = field(default_factory=dict)
    span_order: List[str] = field(default_factory=list)
    next_span_idx: int = 0
    spans_version: int = 0
    microplan: Optional[List[MicroplanStep]] = None
    microplan_audit: Optional[PlanAudit] = None
    pending_writes: Dict[str, PendingWrite] = field(default_factory=dict)
    attempts: List[AttemptRecord] = field(default_factory=list)
    next_attempt_idx: int = 0

    # Unified approval system (per-run grants).
    # - pending_grants maps token -> PendingGrant
    # - granted_scopes maps scope -> expires_at (unix seconds)
    pending_grants: Dict[str, PendingGrant] = field(default_factory=dict)
    granted_scopes: Dict[str, float] = field(default_factory=dict)

    # Optional planning / approvals (a lighter-weight alternative to strict microplans).
    plan_sid: Optional[str] = None
    plan_approved: bool = False
    plan_approval_token: Optional[str] = None

    # Optional per-run permission handshakes.
    web_access_granted: bool = False
    web_access_token: Optional[str] = None
    exec_access_granted: bool = False
    exec_access_token: Optional[str] = None

    # ------------------------------------------------------------------
    # Science-server metadata
    # ------------------------------------------------------------------
    # Baseline snapshot information used for evidence provenance.
    # If `baseline_kind == 'git'`, repo evidence should be read from `baseline_ref` (a git commit hash)
    # rather than the working tree, to prevent evidence-poisoning via self-authored files.
    baseline_kind: str = "fs"  # 'git'|'fs'
    baseline_ref: Optional[str] = None

    # Pending action details (used by the small-surface science server).
    # These are NOT evidence spans; they are server-side control state.
    pending_web: Optional[Dict[str, Any]] = None
    pending_exec: Optional[Dict[str, Any]] = None

    # ---------------------------------------
    # Classic-server metadata
    # ---------------------------------------
    # Immutable deliverable anchor span id (created by start_run).
    # This is *not* evidence by itself, but it captures the user's goal.
    deliverable_sid: Optional[str] = None


class RunStore:
    def __init__(self):
        self._runs: Dict[str, RunState] = {}
        self._active_run_id: Optional[str] = None

    def start_run(self, *, run_id: Optional[str] = None) -> RunState:
        rid = (run_id or secrets.token_hex(8)).strip()
        if not rid:
            raise EnforcementError("run_id is required")
        run = RunState(run_id=rid, created_at=time.time())
        self._runs[rid] = run
        self._active_run_id = rid
        return run

    def set_active_run(self, run_id: str) -> RunState:
        rid = str(run_id or "").strip()
        if rid not in self._runs:
            raise EnforcementError(f"Unknown run_id: {rid}")
        self._active_run_id = rid
        return self._runs[rid]

    def get_active_run_id(self) -> Optional[str]:
        return self._active_run_id

    def get_run(self, run_id: Optional[str]) -> RunState:
        rid = (str(run_id).strip() if run_id is not None else (self._active_run_id or "")).strip()
        if not rid:
            raise EnforcementError("No active run. Call start_run first (or pass run_id).")
        if rid not in self._runs:
            raise EnforcementError(f"Unknown run_id: {rid}")
        return self._runs[rid]

    def reset_run(self, run_id: Optional[str]) -> RunState:
        run = self.get_run(run_id)
        run.spans.clear()
        run.span_order.clear()
        run.next_span_idx = 0
        run.spans_version += 1
        run.attempts.clear()
        run.next_attempt_idx = 0
        run.microplan = None
        run.microplan_audit = None
        run.pending_writes.clear()

        # Clear grants.
        run.pending_grants.clear()
        run.granted_scopes.clear()

        run.plan_sid = None
        run.plan_approved = False
        run.plan_approval_token = None
        run.web_access_granted = False
        run.web_access_token = None
        run.exec_access_granted = False
        run.exec_access_token = None

        # Classic: clear deliverable anchor.
        run.deliverable_sid = None
        return run

    def record_attempt(
        self,
        *,
        run: RunState,
        claim_id: str,
        hypothesis: str,
        action: str,
        budget_minutes: float = 5.0,
        input_sids: Optional[List[str]] = None,
        output_sids: Optional[List[str]] = None,
        audit_status: str = "",
        decision: str = "",
        git_state: str = "",
        objective_metric: str = "",
        objective_value: str = "",
        result_summary: str = "",
        next_step: str = "",
    ) -> AttemptRecord:
        claim = str(claim_id or "").strip()
        act = str(action or "").strip()
        if not claim:
            raise EnforcementError("claim_id is required")
        if not act:
            raise EnforcementError("action is required")

        def _clean_sids(values: Optional[List[str]]) -> List[str]:
            out: List[str] = []
            for raw in values or []:
                sid = str(raw or "").strip()
                if not sid:
                    continue
                if sid not in run.spans:
                    raise EnforcementError(f"Unknown span id: {sid}")
                out.append(sid)
            return out

        rec = AttemptRecord(
            attempt_id=f"A{run.next_attempt_idx}",
            created_at=time.time(),
            claim_id=claim,
            hypothesis=str(hypothesis or "").strip(),
            action=act,
            budget_minutes=max(0.0, float(budget_minutes or 0.0)),
            input_sids=_clean_sids(input_sids),
            output_sids=_clean_sids(output_sids),
            audit_status=str(audit_status or "").strip(),
            decision=str(decision or "").strip(),
            git_state=str(git_state or "").strip(),
            objective_metric=str(objective_metric or "").strip(),
            objective_value=str(objective_value or "").strip(),
            result_summary=str(result_summary or "").strip(),
            next_step=str(next_step or "").strip(),
        )
        run.next_attempt_idx += 1
        run.attempts.append(rec)
        return rec

    def list_attempts(self, *, run: RunState, limit: int = 200) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        limit_n = max(1, int(limit or 200))
        for rec in run.attempts[:limit_n]:
            out.append(
                {
                    "attempt_id": rec.attempt_id,
                    "created_at": rec.created_at,
                    "claim_id": rec.claim_id,
                    "hypothesis": rec.hypothesis,
                    "action": rec.action,
                    "budget_minutes": rec.budget_minutes,
                    "input_sids": list(rec.input_sids),
                    "output_sids": list(rec.output_sids),
                    "audit_status": rec.audit_status,
                    "decision": rec.decision,
                    "git_state": rec.git_state,
                    "objective_metric": rec.objective_metric,
                    "objective_value": rec.objective_value,
                    "result_summary": rec.result_summary,
                    "next_step": rec.next_step,
                }
            )
        return out

    def add_span(
        self, *, run: RunState, text: str, source: str, meta: Optional[Dict[str, Any]] = None
    ) -> SpanRecord:
        t = str(text or "")
        if not t.strip():
            raise EnforcementError("Span text is empty")
        sid = f"S{run.next_span_idx}"
        run.next_span_idx += 1
        rec = SpanRecord(
            sid=sid,
            text=t,
            source=str(source or "manual"),
            created_at=time.time(),
            meta=dict(meta or {}),
        )
        run.spans[sid] = rec
        run.span_order.append(sid)
        run.spans_version += 1
        return rec

    def list_spans(self, *, run: RunState, limit: int = 200) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sid in run.span_order[: max(1, int(limit or 200))]:
            s = run.spans[sid]
            preview = s.text.strip().replace("\n", " ")
            out.append(
                {
                    "sid": s.sid,
                    "source": s.source,
                    "created_at": s.created_at,
                    "chars": len(s.text),
                    "preview": preview[:160],
                    "meta": s.meta,
                }
            )
        return out

    def get_span(self, *, run: RunState, sid: str) -> SpanRecord:
        key = str(sid or "").strip()
        if not key:
            raise EnforcementError("sid is required")
        if key not in run.spans:
            raise EnforcementError(f"Unknown span id: {key}")
        return run.spans[key]

    def set_microplan(
        self, *, run: RunState, steps: List[Dict[str, Any]], default_target: float = 0.8
    ) -> List[MicroplanStep]:
        out: List[MicroplanStep] = []
        for i, st in enumerate(steps or []):
            claim = str(st.get("claim") or "").strip()
            if not claim:
                continue
            idx = int(st.get("idx", i))
            cites = [str(c).strip() for c in (st.get("cites") or []) if str(c).strip()]
            confidence = float(st.get("confidence", default_target) or default_target)
            out.append(MicroplanStep(idx=idx, claim=claim, cites=cites, confidence=confidence))
        out.sort(key=lambda x: x.idx)
        run.microplan = out
        run.microplan_audit = None
        return out

    def get_microplan(self, *, run: RunState) -> List[Dict[str, Any]]:
        if not run.microplan:
            return []
        return [
            {"idx": s.idx, "claim": s.claim, "cites": list(s.cites), "confidence": s.confidence}
            for s in run.microplan
        ]

    def set_microplan_audit(self, *, run: RunState, report: Dict[str, Any]) -> PlanAudit:
        ok = not bool(report.get("flagged", True))
        audit = PlanAudit(
            spans_version=run.spans_version, ok=ok, report=report, audited_at=time.time()
        )
        run.microplan_audit = audit
        return audit

    def require_audited_step(self, *, run: RunState, step_idx: int) -> MicroplanStep:
        if not run.microplan:
            raise EnforcementError("No microplan set. Call set_microplan first.")
        if run.microplan_audit is None:
            raise EnforcementError("Microplan not audited. Call audit_microplan first.")
        if not run.microplan_audit.ok:
            raise EnforcementError("Microplan audit is flagged; fix plan/citations and re-audit.")
        if run.microplan_audit.spans_version != run.spans_version:
            raise EnforcementError("Evidence changed since last audit; re-audit microplan.")

        idx = int(step_idx)
        for s in run.microplan:
            if s.idx == idx:
                # Ensure cited spans exist (when provided).
                missing = [c for c in s.cites if c not in run.spans]
                if missing:
                    raise EnforcementError(f"Microplan step cites unknown spans: {missing}")
                untrusted = [
                    c for c in s.cites if not bool((run.spans[c].meta or {}).get("trusted", False))
                ]
                if untrusted:
                    raise EnforcementError(f"Microplan step cites untrusted spans: {untrusted}")
                return s
        raise EnforcementError(f"Unknown microplan step idx: {idx}")

    # ---------------------------------------------------------------------
    # Unified approvals (grants)
    # ---------------------------------------------------------------------

    def _prune_expired_grants(self, *, run: RunState) -> None:
        now = time.time()

        # Pending
        expired_pending = [
            tok for tok, g in (run.pending_grants or {}).items() if now > float(g.expires_at or 0.0)
        ]
        for tok in expired_pending:
            run.pending_grants.pop(tok, None)

        # Granted
        expired_scopes = [
            s for s, exp in (run.granted_scopes or {}).items() if now > float(exp or 0.0)
        ]
        for s in expired_scopes:
            run.granted_scopes.pop(s, None)

    def request_grant(
        self,
        *,
        run: RunState,
        scopes: List[str],
        summary: str = "",
        ttl_s: float = 3600.0,
    ) -> PendingGrant:
        self._prune_expired_grants(run=run)
        cleaned = [str(s).strip() for s in (scopes or []) if str(s).strip()]
        if not cleaned:
            raise EnforcementError("scopes must be a non-empty list")
        now = time.time()
        token = secrets.token_urlsafe(16)
        g = PendingGrant(
            token=token,
            scopes=cleaned,
            summary=str(summary or "").strip(),
            created_at=now,
            expires_at=now + float(ttl_s or 0.0),
        )
        run.pending_grants[token] = g
        return g

    def grant(self, *, run: RunState, token: str) -> List[str]:
        self._prune_expired_grants(run=run)
        tok = str(token or "").strip()
        if not tok:
            raise EnforcementError("token is required")
        if tok not in run.pending_grants:
            raise EnforcementError("Unknown or expired token")
        g = run.pending_grants.pop(tok)
        if time.time() > float(g.expires_at or 0.0):
            raise EnforcementError("Token expired")

        granted_now: List[str] = []
        for s in g.scopes:
            run.granted_scopes[str(s)] = float(g.expires_at)
            granted_now.append(str(s))
        return granted_now

    def has_scope(self, *, run: RunState, scope: str) -> bool:
        self._prune_expired_grants(run=run)
        s = str(scope or "").strip()
        if not s:
            return False
        exp = (run.granted_scopes or {}).get(s)
        if not exp:
            return False
        return time.time() <= float(exp)

    def list_grants(self, *, run: RunState) -> Dict[str, Any]:
        self._prune_expired_grants(run=run)
        granted = [
            {"scope": s, "expires_at": float(exp)}
            for s, exp in sorted((run.granted_scopes or {}).items(), key=lambda kv: kv[0])
        ]
        pending = [
            {
                "token": tok,
                "scopes": list(g.scopes),
                "summary": g.summary,
                "created_at": float(g.created_at),
                "expires_at": float(g.expires_at),
            }
            for tok, g in sorted((run.pending_grants or {}).items(), key=lambda kv: kv[0])
        ]
        return {"granted": granted, "pending": pending}

    def mint_write_token(
        self,
        *,
        run: RunState,
        path: str,
        contents: str,
        change_summary: str,
        motivation: str,
        mode: str,
        rationale: str = "",
        pre_image_sha256: str,
        diff_sid: str,
        verification: Dict[str, Any],
        step_idx: int,
        ttl_s: float = 300.0,
    ) -> PendingWrite:
        now = time.time()
        token = secrets.token_urlsafe(24)
        pw = PendingWrite(
            token=token,
            path=str(path),
            contents=str(contents),
            change_summary=str(change_summary or ""),
            motivation=str(motivation or ""),
            mode=str(mode or "verified"),
            rationale=str(rationale or ""),
            pre_image_sha256=str(pre_image_sha256 or ""),
            diff_sid=str(diff_sid or ""),
            created_at=now,
            expires_at=now + float(ttl_s),
            verification=dict(verification),
            step_idx=int(step_idx),
        )
        run.pending_writes[token] = pw
        return pw

    def pop_write_token(self, *, run: RunState, token: str) -> PendingWrite:
        t = str(token or "").strip()
        if not t:
            raise EnforcementError("token is required")
        if t not in run.pending_writes:
            raise EnforcementError("Unknown or expired token")
        pw = run.pending_writes.pop(t)
        if time.time() > pw.expires_at:
            raise EnforcementError("Token expired")
        return pw
