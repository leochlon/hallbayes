from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


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


ALLOWED_SPAN_KINDS = {
    "anchor",
    "evidence",
    "observation",
    "derived",
    "assumption",
    "decision",
    "audit",
}
CITABLE_SPAN_KINDS = {"evidence", "observation", "derived"}
ACTIVE_SPAN_STATUSES = {"active"}
ALLOWED_SPAN_STATUSES = {
    "active",
    "superseded",
    "stale",
    "tombstoned",
    "redacted",
    "quarantined",
}
ALLOWED_SPAN_SENSITIVITY = {"normal", "secret", "pii", "unknown"}
ALLOWED_CLAIM_KINDS = {"fact", "hypothesis", "decision", "assumption", "unknown"}
ALLOWED_CLAIM_STATUSES = {
    "open",
    "supported",
    "contradicted",
    "insufficient",
    "downgraded",
    "closed",
}
ALLOWED_CLAIM_EVIDENCE_RELATIONS = {
    "supports",
    "contradicts",
    "background",
    "insufficient",
}
ALLOWED_AUDIT_KINDS = {
    "audit_trace_budget_run",
    "detect_hallucination_run",
    "audit_claims",
    "manual",
}
SECRET_RE = re.compile(
    r"(?i)(-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[=:]\s*['\"]?[^\s'\"]{8,}|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b)"
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="surrogatepass")).hexdigest()


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detect_span_sensitivity(text: str, explicit: Optional[str] = None) -> str:
    """Conservatively classify obvious secrets before spans are listed or packed."""

    ex = str(explicit or "").strip().lower()
    if ex in ALLOWED_SPAN_SENSITIVITY and ex != "unknown":
        return ex
    if SECRET_RE.search(str(text or "")):
        return "secret"
    return ex if ex in ALLOWED_SPAN_SENSITIVITY else "unknown"


def _normalise_kind(*, kind: Optional[str], source: str, meta: Dict[str, Any]) -> str:
    k = str(kind or "").strip().lower()
    if not k:
        mk = str((meta or {}).get("kind") or "").strip().lower()
        src = str(source or "").strip().lower()
        if src == "anchor" or mk in {"anchor", "problem", "deliverable"}:
            k = "anchor"
        elif src in {"distill", "extract", "derived"}:
            k = "derived"
        elif src in {"audit", "verifier"}:
            k = "audit"
        else:
            k = "evidence"
    return k if k in ALLOWED_SPAN_KINDS else "evidence"


def _normalise_status(status: Optional[str]) -> str:
    s = str(status or "active").strip().lower()
    return s if s in ALLOWED_SPAN_STATUSES else "active"


def _normalise_claim_kind(kind: Optional[str]) -> str:
    k = str(kind or "fact").strip().lower()
    return k if k in ALLOWED_CLAIM_KINDS else "unknown"


def _normalise_claim_status(status: Optional[str]) -> str:
    s = str(status or "open").strip().lower()
    return s if s in ALLOWED_CLAIM_STATUSES else "open"


def _normalise_claim_relation(relation: Optional[str]) -> str:
    r = str(relation or "supports").strip().lower()
    return r if r in ALLOWED_CLAIM_EVIDENCE_RELATIONS else "background"


def _normalise_audit_kind(kind: Optional[str]) -> str:
    k = str(kind or "manual").strip().lower()
    return k if k in ALLOWED_AUDIT_KINDS else "manual"


def _validate_claim_target(value: Any) -> float:
    try:
        target = float(value if value is not None else 0.95)
    except Exception as exc:
        raise EnforcementError(
            f"target must be a finite probability in (0, 1), got {value!r}"
        ) from exc
    if not (0.0 < target < 1.0):
        raise EnforcementError(f"target must be a finite probability in (0, 1), got {value!r}")
    return target


def _normalise_tags(tags: Optional[Iterable[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in tags or []:
        tag = str(raw or "").strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def mark_run_dirty(
    run: "RunState",
    *,
    all: bool = False,
    meta: bool = False,
    spans: Optional[Iterable[str]] = None,
    attempts: Optional[Iterable[str]] = None,
    claims: Optional[Iterable[str]] = None,
    claim_links: Optional[Iterable[str]] = None,
    audits: Optional[Iterable[str]] = None,
) -> None:
    """Mark ledger rows that must be durably re-committed.

    The SQLite ledger persists hot writes incrementally.  RunStore mutation
    methods call this helper so ``persist_run`` can avoid reserializing and
    reinserting every child row on every span append.  Direct users that mutate
    ``RunState`` fields outside RunStore should either call this helper or force
    a full checkpoint through the ledger layer.
    """

    if all:
        run.ledger_dirty_all = True
        meta = True
    if meta:
        run.ledger_dirty_meta = True
    for sid in spans or []:
        value = str(sid or "").strip()
        if value:
            run.ledger_dirty_spans.add(value)
    for attempt_id in attempts or []:
        value = str(attempt_id or "").strip()
        if value:
            run.ledger_dirty_attempts.add(value)
    for cid in claims or []:
        value = str(cid or "").strip()
        if value:
            run.ledger_dirty_claims.add(value)
    for link_id in claim_links or []:
        value = str(link_id or "").strip()
        if value:
            run.ledger_dirty_claim_links.add(value)
    for audit_id in audits or []:
        value = str(audit_id or "").strip()
        if value:
            run.ledger_dirty_audits.add(value)


def clear_run_dirty(run: "RunState") -> None:
    run.ledger_dirty_all = False
    run.ledger_dirty_meta = False
    run.ledger_dirty_spans.clear()
    run.ledger_dirty_attempts.clear()
    run.ledger_dirty_claims.clear()
    run.ledger_dirty_claim_links.clear()
    run.ledger_dirty_audits.clear()


@dataclass(frozen=True)
class SpanRecord:
    """Run-local evidence object.

    v1 Berry treated a span as only ``sid/text/source/meta``.  v2 keeps those
    fields for compatibility, but adds first-class provenance, trust, lineage,
    sensitivity, and lifecycle fields so the verifier can resolve evidence from
    the server-owned run ledger instead of trusting caller-supplied snippets.
    """

    sid: str
    text: str
    source: str
    created_at: float
    meta: Dict[str, Any] = field(default_factory=dict)

    # v2 provenance and policy fields.  Defaults keep old serialized runs loadable.
    eid: str = ""
    kind: str = ""
    source_type: str = ""
    media_type: str = "text/plain"
    text_sha256: str = ""
    locator: Dict[str, Any] = field(default_factory=dict)
    snapshot: Dict[str, Any] = field(default_factory=dict)
    parents: List[str] = field(default_factory=list)
    transform: Optional[Dict[str, Any]] = None
    trust: str = ""
    status: str = "active"
    sensitivity: str = "unknown"
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        meta = dict(self.meta or {})
        source = str(self.source or self.source_type or "manual")
        source_type = str(self.source_type or source or "manual")
        kind = _normalise_kind(kind=self.kind, source=source, meta=meta)
        text = str(self.text or "")
        text_sha = str(self.text_sha256 or _sha256_text(text))
        locator = dict(self.locator or {})
        snapshot = dict(self.snapshot or {})
        parents = [str(p).strip() for p in (self.parents or []) if str(p).strip()]
        transform = dict(self.transform) if isinstance(self.transform, dict) else self.transform
        trust = str(self.trust or "").strip().lower()
        if not trust:
            trust = "manual" if kind == "anchor" else "derived" if kind == "derived" else "primary"
        sensitivity = detect_span_sensitivity(text, self.sensitivity)
        tags = _normalise_tags(self.tags)
        status = _normalise_status(self.status)
        eid = str(self.eid or "").strip()
        if not eid:
            eid = _stable_json_hash(
                {
                    "text_sha256": text_sha,
                    "source_type": source_type,
                    "media_type": str(self.media_type or "text/plain"),
                    "locator": locator,
                    "snapshot": snapshot,
                    "parents": parents,
                    "transform": transform,
                }
            )

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "media_type", str(self.media_type or "text/plain"))
        object.__setattr__(self, "text_sha256", text_sha)
        object.__setattr__(self, "locator", locator)
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "trust", trust)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "eid", eid)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_SPAN_STATUSES

    @property
    def is_sensitive(self) -> bool:
        return self.sensitivity in {"secret", "pii"}

    @property
    def is_citable(self) -> bool:
        if not self.is_active or self.kind not in CITABLE_SPAN_KINDS:
            return False
        if self.kind == "derived":
            t = (self.transform or {}).get("type") if isinstance(self.transform, dict) else None
            return str(t or "").strip().lower() in {"regex_extract", "line_range", "byte_range"}
        return True

    def preview(self, *, limit: int = 160, redact_sensitive: bool = True) -> str:
        if redact_sensitive and self.is_sensitive:
            return "[REDACTED sensitive span]"
        return self.text.strip().replace("\n", " ")[: max(0, int(limit))]

    def to_public_dict(
        self, *, include_text: bool = False, preview_chars: int = 160, redact_sensitive: bool = True
    ) -> Dict[str, Any]:
        data = {
            "sid": self.sid,
            "eid": self.eid,
            "source": self.source,
            "source_type": self.source_type,
            "kind": self.kind,
            "trust": self.trust,
            "status": self.status,
            "sensitivity": self.sensitivity,
            "created_at": self.created_at,
            "chars": len(self.text),
            "text_sha256": self.text_sha256,
            "locator": dict(self.locator or {}),
            "snapshot": dict(self.snapshot or {}),
            "parents": list(self.parents or []),
            "transform": (
                dict(self.transform or {}) if isinstance(self.transform, dict) else self.transform
            ),
            "tags": list(self.tags or []),
            "preview": self.preview(limit=preview_chars, redact_sensitive=redact_sensitive),
            "meta": dict(self.meta or {}),
            # Back-compat convenience fields used by older clients.
            "source_legacy": self.source,
        }
        if include_text:
            data["text"] = (
                "[REDACTED sensitive span]" if redact_sensitive and self.is_sensitive else self.text
            )
        return data


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
class ClaimRecord:
    """Structured factual unit tracked across evidence gathering and audits."""

    cid: str
    text: str
    kind: str = "fact"
    status: str = "open"
    target: float = 0.95
    created_at: float = 0.0
    updated_at: float = 0.0
    source: str = "manual"
    latest_audit_id: str = ""
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        now = time.time()
        text = str(self.text or "").strip()
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "kind", _normalise_claim_kind(self.kind))
        object.__setattr__(self, "status", _normalise_claim_status(self.status))
        object.__setattr__(self, "target", _validate_claim_target(self.target))
        object.__setattr__(self, "created_at", float(self.created_at or now))
        object.__setattr__(self, "updated_at", float(self.updated_at or self.created_at or now))
        object.__setattr__(self, "source", str(self.source or "manual").strip() or "manual")
        object.__setattr__(self, "latest_audit_id", str(self.latest_audit_id or "").strip())
        object.__setattr__(self, "tags", _normalise_tags(self.tags))
        object.__setattr__(self, "meta", dict(self.meta or {}))

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "claim_id": self.cid,
            "text": self.text,
            "kind": self.kind,
            "status": self.status,
            "target": self.target,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "latest_audit_id": self.latest_audit_id,
            "tags": list(self.tags or []),
            "meta": dict(self.meta or {}),
        }


@dataclass(frozen=True)
class ClaimEvidenceLink:
    """Typed relation between a claim and a run-owned evidence span."""

    link_id: str
    cid: str
    sid: str
    relation: str = "supports"
    created_at: float = 0.0
    created_by: str = "manual"
    audit_id: str = ""
    note: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cid", str(self.cid or "").strip())
        object.__setattr__(self, "sid", str(self.sid or "").strip())
        object.__setattr__(self, "relation", _normalise_claim_relation(self.relation))
        object.__setattr__(self, "created_at", float(self.created_at or time.time()))
        object.__setattr__(self, "created_by", str(self.created_by or "manual").strip() or "manual")
        object.__setattr__(self, "audit_id", str(self.audit_id or "").strip())
        object.__setattr__(self, "note", str(self.note or "").strip())
        object.__setattr__(self, "meta", dict(self.meta or {}))

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "cid": self.cid,
            "claim_id": self.cid,
            "sid": self.sid,
            "relation": self.relation,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "audit_id": self.audit_id,
            "note": self.note,
            "meta": dict(self.meta or {}),
        }


@dataclass(frozen=True)
class AuditRecord:
    """Verifier audit metadata for reproducible claim/evidence decisions."""

    audit_id: str
    kind: str
    created_at: float
    claim_ids: List[str] = field(default_factory=list)
    input_sids: List[str] = field(default_factory=list)
    materialized_sids: List[str] = field(default_factory=list)
    evidence_pack_id: str = ""
    evidence_pack_hash: str = ""
    verifier_model: str = ""
    policy: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    audit_sid: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _normalise_audit_kind(self.kind))
        object.__setattr__(self, "created_at", float(self.created_at or time.time()))
        object.__setattr__(
            self,
            "claim_ids",
            [str(v).strip() for v in (self.claim_ids or []) if str(v).strip()],
        )
        object.__setattr__(
            self,
            "input_sids",
            [str(v).strip() for v in (self.input_sids or []) if str(v).strip()],
        )
        object.__setattr__(
            self,
            "materialized_sids",
            [str(v).strip() for v in (self.materialized_sids or []) if str(v).strip()],
        )
        object.__setattr__(self, "evidence_pack_id", str(self.evidence_pack_id or "").strip())
        object.__setattr__(self, "evidence_pack_hash", str(self.evidence_pack_hash or "").strip())
        object.__setattr__(self, "verifier_model", str(self.verifier_model or "").strip())
        object.__setattr__(self, "policy", dict(self.policy or {}))
        object.__setattr__(self, "result", dict(self.result or {}))
        object.__setattr__(self, "audit_sid", str(self.audit_sid or "").strip())
        object.__setattr__(self, "meta", dict(self.meta or {}))

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "kind": self.kind,
            "created_at": self.created_at,
            "claim_ids": list(self.claim_ids or []),
            "input_sids": list(self.input_sids or []),
            "materialized_sids": list(self.materialized_sids or []),
            "evidence_pack_id": self.evidence_pack_id,
            "evidence_pack_hash": self.evidence_pack_hash,
            "verifier_model": self.verifier_model,
            "policy": dict(self.policy or {}),
            "result": dict(self.result or {}),
            "audit_sid": self.audit_sid,
            "meta": dict(self.meta or {}),
        }


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
    claims: Dict[str, ClaimRecord] = field(default_factory=dict)
    claim_order: List[str] = field(default_factory=list)
    next_claim_idx: int = 0
    claim_evidence_links: List[ClaimEvidenceLink] = field(default_factory=list)
    next_claim_link_idx: int = 0
    audits: List[AuditRecord] = field(default_factory=list)
    next_audit_idx: int = 0

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
    # If `baseline_kind == 'git'`, repo evidence should be read from `baseline_ref`
    # (a git commit hash) rather than the working tree, to prevent
    # evidence-poisoning via self-authored files.
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

    # Incremental ledger bookkeeping. These fields are intentionally not serialized
    # into run payloads; they describe what the in-memory object has changed since
    # the last durable commit. Newly created runs start dirty so the first persist
    # writes a full baseline event. Loaded runs are marked clean by the ledger.
    ledger_dirty_all: bool = field(default=True, repr=False)
    ledger_dirty_meta: bool = field(default=True, repr=False)
    ledger_dirty_spans: Set[str] = field(default_factory=set, repr=False)
    ledger_dirty_attempts: Set[str] = field(default_factory=set, repr=False)
    ledger_dirty_claims: Set[str] = field(default_factory=set, repr=False)
    ledger_dirty_claim_links: Set[str] = field(default_factory=set, repr=False)
    ledger_dirty_audits: Set[str] = field(default_factory=set, repr=False)
    ledger_committed_head_event_hash: str = field(default="", repr=False)
    ledger_committed_run_meta_sha256: str = field(default="", repr=False)
    ledger_committed_row_hashes: Dict[str, Dict[str, str]] = field(default_factory=dict, repr=False)


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
        run.claims.clear()
        run.claim_order.clear()
        run.next_claim_idx = 0
        run.claim_evidence_links.clear()
        run.next_claim_link_idx = 0
        run.audits.clear()
        run.next_audit_idx = 0
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
        mark_run_dirty(run, all=True)
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
        mark_run_dirty(run, meta=True, attempts=[rec.attempt_id])
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

    # ------------------------------------------------------------------
    # Claim / evidence graph
    # ------------------------------------------------------------------

    def create_claim(
        self,
        *,
        run: RunState,
        text: str,
        kind: str = "fact",
        target: float = 0.95,
        status: str = "open",
        source: str = "manual",
        tags: Optional[Sequence[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        cid: Optional[str] = None,
    ) -> ClaimRecord:
        claim_text = str(text or "").strip()
        if not claim_text:
            raise EnforcementError("claim text is required")
        clean_cid = str(cid or "").strip()
        if clean_cid:
            if clean_cid in run.claims:
                raise EnforcementError(f"Claim id already exists: {clean_cid}")
            if clean_cid.startswith("C") and clean_cid[1:].isdigit():
                run.next_claim_idx = max(run.next_claim_idx, int(clean_cid[1:]) + 1)
        else:
            clean_cid = f"C{run.next_claim_idx}"
            run.next_claim_idx += 1
        rec = ClaimRecord(
            cid=clean_cid,
            text=claim_text,
            kind=kind,
            status=status,
            target=target,
            created_at=time.time(),
            updated_at=time.time(),
            source=source,
            tags=list(tags or []),
            meta=dict(meta or {}),
        )
        run.claims[rec.cid] = rec
        run.claim_order.append(rec.cid)
        mark_run_dirty(run, meta=True, claims=[rec.cid])
        return rec

    def get_claim(self, *, run: RunState, cid: str) -> ClaimRecord:
        key = str(cid or "").strip()
        if not key:
            raise EnforcementError("cid is required")
        if key not in run.claims:
            raise EnforcementError(f"Unknown claim id: {key}")
        return run.claims[key]

    def update_claim(
        self,
        *,
        run: RunState,
        cid: str,
        status: Optional[str] = None,
        kind: Optional[str] = None,
        target: Optional[float] = None,
        latest_audit_id: Optional[str] = None,
        tags_add: Optional[Sequence[str]] = None,
        tags_remove: Optional[Sequence[str]] = None,
        meta_update: Optional[Dict[str, Any]] = None,
    ) -> ClaimRecord:
        rec = self.get_claim(run=run, cid=cid)
        tags = list(rec.tags or [])
        remove = {str(t).strip() for t in (tags_remove or []) if str(t).strip()}
        tags = [t for t in tags if t not in remove]
        for raw in tags_add or []:
            tag = str(raw or "").strip()
            if tag and tag not in tags:
                tags.append(tag)
        meta = dict(rec.meta or {})
        if meta_update:
            meta.update(dict(meta_update))
        updated = replace(
            rec,
            kind=_normalise_claim_kind(kind) if kind is not None else rec.kind,
            status=_normalise_claim_status(status) if status is not None else rec.status,
            target=_validate_claim_target(target) if target is not None else rec.target,
            latest_audit_id=(
                str(latest_audit_id or "").strip()
                if latest_audit_id is not None
                else rec.latest_audit_id
            ),
            updated_at=time.time(),
            tags=tags,
            meta=meta,
        )
        run.claims[rec.cid] = updated
        mark_run_dirty(run, meta=True, claims=[rec.cid])
        return updated

    def list_claims(
        self,
        *,
        run: RunState,
        limit: int = 200,
        status: Optional[Sequence[str]] = None,
        kinds: Optional[Sequence[str]] = None,
        include_evidence: bool = True,
    ) -> List[Dict[str, Any]]:
        status_set = {str(s).strip().lower() for s in (status or []) if str(s).strip()}
        kind_set = {str(k).strip().lower() for k in (kinds or []) if str(k).strip()}
        out: List[Dict[str, Any]] = []
        limit_n = max(1, int(limit or 200))
        for cid in run.claim_order:
            if len(out) >= limit_n:
                break
            rec = run.claims.get(cid)
            if rec is None:
                continue
            if status_set and rec.status not in status_set:
                continue
            if kind_set and rec.kind not in kind_set:
                continue
            data = rec.to_public_dict()
            if include_evidence:
                data["evidence"] = [
                    link.to_public_dict()
                    for link in run.claim_evidence_links
                    if link.cid == rec.cid
                ]
            out.append(data)
        return out

    def link_claim_evidence(
        self,
        *,
        run: RunState,
        cid: str,
        sid: str,
        relation: str = "supports",
        created_by: str = "manual",
        audit_id: str = "",
        note: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> ClaimEvidenceLink:
        claim = self.get_claim(run=run, cid=cid)
        span = self.get_span(run=run, sid=sid)
        rel = _normalise_claim_relation(relation)
        if rel in {"supports", "contradicts"} and not span.is_citable:
            raise EnforcementError(
                f"Span {span.sid} is not citable as {rel} evidence "
                f"(kind={span.kind}, status={span.status}, sensitivity={span.sensitivity})"
            )
        # De-duplicate identical active graph edges; update audit/note metadata by creating
        # a new link only when the relation or audit differs.  This keeps repeated audit
        # calls from accumulating an unbounded pile of identical manual edges.
        for existing in run.claim_evidence_links:
            if (
                existing.cid == claim.cid
                and existing.sid == span.sid
                and existing.relation == rel
                and existing.audit_id == str(audit_id or "").strip()
            ):
                return existing
        link = ClaimEvidenceLink(
            link_id=f"L{run.next_claim_link_idx}",
            cid=claim.cid,
            sid=span.sid,
            relation=rel,
            created_at=time.time(),
            created_by=created_by,
            audit_id=audit_id,
            note=note,
            meta=dict(meta or {}),
        )
        run.next_claim_link_idx += 1
        run.claim_evidence_links.append(link)
        mark_run_dirty(run, meta=True, claim_links=[link.link_id])
        return link

    def list_claim_evidence(
        self,
        *,
        run: RunState,
        cid: Optional[str] = None,
        sid: Optional[str] = None,
        relation: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        rel_set = {str(r).strip().lower() for r in (relation or []) if str(r).strip()}
        cid_key = str(cid or "").strip()
        sid_key = str(sid or "").strip()
        out: List[Dict[str, Any]] = []
        for link in run.claim_evidence_links:
            if cid_key and link.cid != cid_key:
                continue
            if sid_key and link.sid != sid_key:
                continue
            if rel_set and link.relation not in rel_set:
                continue
            out.append(link.to_public_dict())
        return out

    def record_audit(
        self,
        *,
        run: RunState,
        kind: str,
        claim_ids: Optional[Sequence[str]] = None,
        input_sids: Optional[Sequence[str]] = None,
        materialized_sids: Optional[Sequence[str]] = None,
        evidence_pack_id: str = "",
        evidence_pack_hash: str = "",
        verifier_model: str = "",
        policy: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        audit_sid: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        cleaned_claims: List[str] = []
        for raw in claim_ids or []:
            cid = str(raw or "").strip()
            if not cid:
                continue
            if cid not in run.claims:
                raise EnforcementError(f"Unknown claim id: {cid}")
            if cid not in cleaned_claims:
                cleaned_claims.append(cid)
        cleaned_inputs: List[str] = []
        for raw in input_sids or []:
            sid = str(raw or "").strip()
            if not sid:
                continue
            if sid not in run.spans:
                raise EnforcementError(f"Unknown input span id: {sid}")
            if sid not in cleaned_inputs:
                cleaned_inputs.append(sid)
        cleaned_materialized: List[str] = []
        for raw in materialized_sids or []:
            sid = str(raw or "").strip()
            if not sid:
                continue
            if sid not in run.spans:
                raise EnforcementError(f"Unknown materialized span id: {sid}")
            if sid not in cleaned_materialized:
                cleaned_materialized.append(sid)
        if audit_sid:
            self.get_span(run=run, sid=audit_sid)
        audit = AuditRecord(
            audit_id=f"V{run.next_audit_idx}",
            kind=kind,
            created_at=time.time(),
            claim_ids=cleaned_claims,
            input_sids=cleaned_inputs,
            materialized_sids=cleaned_materialized,
            evidence_pack_id=evidence_pack_id,
            evidence_pack_hash=evidence_pack_hash,
            verifier_model=verifier_model,
            policy=dict(policy or {}),
            result=dict(result or {}),
            audit_sid=audit_sid,
            meta=dict(meta or {}),
        )
        run.next_audit_idx += 1
        run.audits.append(audit)
        mark_run_dirty(run, meta=True, audits=[audit.audit_id])
        return audit

    def list_audits(
        self,
        *,
        run: RunState,
        claim_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        cid = str(claim_id or "").strip()
        out: List[Dict[str, Any]] = []
        for audit in reversed(run.audits):
            if len(out) >= max(1, int(limit or 100)):
                break
            if cid and cid not in audit.claim_ids:
                continue
            out.append(audit.to_public_dict())
        return out

    def claim_steps(
        self,
        *,
        run: RunState,
        claim_ids: Optional[Sequence[str]] = None,
        max_claims: int = 25,
        evidence_relations: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build verifier steps from the structured claim/evidence graph."""

        relation_set = {
            _normalise_claim_relation(r) for r in (evidence_relations or ["supports", "background"])
        }
        requested = [str(cid).strip() for cid in (claim_ids or []) if str(cid).strip()]
        if not requested:
            requested = [
                cid
                for cid in run.claim_order
                if run.claims.get(cid) and run.claims[cid].status in {"open", "insufficient"}
            ]
        out: List[Dict[str, Any]] = []
        for cid in requested:
            claim = self.get_claim(run=run, cid=cid)
            cites: List[str] = []
            for link in run.claim_evidence_links:
                if link.cid != cid or link.relation not in relation_set:
                    continue
                span = run.spans.get(link.sid)
                # Background edges are graph annotations, not proof.  They may
                # legally point at anchors or decisions, but verifier steps must
                # only cite spans that would pass the evidence-pack policy.
                if span is None or not span.is_citable:
                    continue
                if link.sid not in cites:
                    cites.append(link.sid)
            out.append(
                {
                    "idx": len(out),
                    "claim_id": claim.cid,
                    "claim": claim.text,
                    "cites": cites,
                    "confidence": claim.target,
                }
            )
            if len(out) >= max(1, int(max_claims or 25)):
                break
        return out

    def add_span(
        self,
        *,
        run: RunState,
        text: str,
        source: str,
        meta: Optional[Dict[str, Any]] = None,
        kind: Optional[str] = None,
        source_type: Optional[str] = None,
        media_type: str = "text/plain",
        locator: Optional[Dict[str, Any]] = None,
        snapshot: Optional[Dict[str, Any]] = None,
        parents: Optional[Sequence[str]] = None,
        transform: Optional[Dict[str, Any]] = None,
        trust: Optional[str] = None,
        status: str = "active",
        sensitivity: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> SpanRecord:
        t = str(text or "")
        if not t.strip():
            raise EnforcementError("Span text is empty")

        cleaned_parents: List[str] = []
        for raw in parents or []:
            psid = str(raw or "").strip()
            if not psid:
                continue
            if psid not in run.spans:
                raise EnforcementError(f"Unknown parent span id: {psid}")
            if psid not in cleaned_parents:
                cleaned_parents.append(psid)

        inherited_sensitivity: Optional[str] = sensitivity
        if not inherited_sensitivity and cleaned_parents:
            parent_sens = {run.spans[p].sensitivity for p in cleaned_parents}
            if "secret" in parent_sens:
                inherited_sensitivity = "secret"
            elif "pii" in parent_sens:
                inherited_sensitivity = "pii"

        sid = f"S{run.next_span_idx}"
        run.next_span_idx += 1
        rec = SpanRecord(
            sid=sid,
            text=t,
            source=str(source or source_type or "manual"),
            created_at=time.time(),
            meta=dict(meta or {}),
            kind=kind or "",
            source_type=str(source_type or source or "manual"),
            media_type=str(media_type or "text/plain"),
            locator=dict(locator or {}),
            snapshot=dict(snapshot or {}),
            parents=cleaned_parents,
            transform=dict(transform or {}) if transform else None,
            trust=str(trust or ""),
            status=status,
            sensitivity=inherited_sensitivity or "unknown",
            tags=list(tags or []),
        )
        run.spans[sid] = rec
        run.span_order.append(sid)
        run.spans_version += 1
        mark_run_dirty(run, meta=True, spans=[sid])
        return rec

    def list_spans(
        self,
        *,
        run: RunState,
        limit: int = 200,
        kinds: Optional[Sequence[str]] = None,
        source_types: Optional[Sequence[str]] = None,
        trust: Optional[Sequence[str]] = None,
        status: Optional[Sequence[str]] = None,
        include_sensitive_preview: bool = False,
    ) -> List[Dict[str, Any]]:
        kind_set = {str(k).strip().lower() for k in (kinds or []) if str(k).strip()}
        source_set = {str(k).strip().lower() for k in (source_types or []) if str(k).strip()}
        trust_set = {str(k).strip().lower() for k in (trust or []) if str(k).strip()}
        status_set = {str(k).strip().lower() for k in (status or []) if str(k).strip()}

        out: List[Dict[str, Any]] = []
        limit_n = max(1, int(limit or 200))
        for sid in run.span_order:
            if len(out) >= limit_n:
                break
            s = run.spans[sid]
            if kind_set and s.kind not in kind_set:
                continue
            if source_set and s.source_type not in source_set:
                continue
            if trust_set and s.trust not in trust_set:
                continue
            if status_set and s.status not in status_set:
                continue
            out.append(
                s.to_public_dict(
                    include_text=False,
                    preview_chars=160,
                    redact_sensitive=not include_sensitive_preview,
                )
            )
        return out

    def get_span(self, *, run: RunState, sid: str) -> SpanRecord:
        key = str(sid or "").strip()
        if not key:
            raise EnforcementError("sid is required")
        if key not in run.spans:
            raise EnforcementError(f"Unknown span id: {key}")
        return run.spans[key]

    def mark_span(
        self,
        *,
        run: RunState,
        sid: str,
        status: Optional[str] = None,
        sensitivity: Optional[str] = None,
        tags_add: Optional[Sequence[str]] = None,
        tags_remove: Optional[Sequence[str]] = None,
        trust: Optional[str] = None,
    ) -> SpanRecord:
        rec = self.get_span(run=run, sid=sid)
        tags = list(rec.tags or [])
        remove = {str(t).strip() for t in (tags_remove or []) if str(t).strip()}
        tags = [t for t in tags if t not in remove]
        for raw in tags_add or []:
            tag = str(raw or "").strip()
            if tag and tag not in tags:
                tags.append(tag)
        updated = replace(
            rec,
            status=_normalise_status(status) if status is not None else rec.status,
            sensitivity=detect_span_sensitivity(rec.text, sensitivity)
            if sensitivity is not None
            else rec.sensitivity,
            tags=tags,
            trust=str(trust or rec.trust).strip().lower(),
        )
        run.spans[rec.sid] = updated
        run.spans_version += 1
        mark_run_dirty(run, meta=True, spans=[rec.sid])
        return updated

    def extract_span(
        self,
        *,
        run: RunState,
        parent_sid: str,
        selector: Dict[str, Any],
        reason: str = "",
        source: str = "extract",
        max_lines: int = 200,
        tags: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        parent = self.get_span(run=run, sid=parent_sid)
        sel = dict(selector or {})
        typ = str(sel.get("type") or sel.get("selector_type") or "regex").strip().lower()
        max_n = max(1, int(max_lines or 200))
        lines = parent.text.splitlines()
        matched_lines: List[int] = []

        if typ in {"regex", "regex_extract"}:
            pattern = str(sel.get("pattern") or "")
            if not pattern:
                raise EnforcementError("selector.pattern is required for regex extraction")
            fl = 0
            flags = str(sel.get("flags") or "i")
            if "i" in flags:
                fl |= re.IGNORECASE
            if "m" in flags:
                fl |= re.MULTILINE
            try:
                rx = re.compile(pattern, fl)
            except Exception as exc:
                raise EnforcementError(f"Invalid regex pattern: {exc}")
            for i, line in enumerate(lines, start=1):
                if rx.search(line):
                    matched_lines.append(i)
                    if len(matched_lines) >= max_n:
                        break
        elif typ in {"line_range", "lines"}:
            start = max(1, int(sel.get("start_line") or sel.get("start") or 1))
            end = max(start, int(sel.get("end_line") or sel.get("end") or start))
            matched_lines = list(range(start, min(end, len(lines)) + 1))[:max_n]
        else:
            raise EnforcementError(f"Unsupported selector type: {typ}")

        if not matched_lines:
            return {
                "matched": False,
                "sid": None,
                "parent_sid": parent.sid,
                "matches": [],
                "message": "No lines matched; no evidence span created.",
            }

        text = "\n".join(lines[i - 1] for i in matched_lines if 1 <= i <= len(lines)).strip()
        if not text:
            return {
                "matched": False,
                "sid": None,
                "parent_sid": parent.sid,
                "matches": [],
                "message": "Selected lines were empty; no evidence span created.",
            }

        transform = {
            "type": "line_range" if typ in {"line_range", "lines"} else "regex_extract",
            "selector": sel,
            "reason": str(reason or ""),
            "matched_lines": matched_lines,
        }
        locator = dict(parent.locator or {})
        locator.update({"parent_sid": parent.sid, "matched_lines": matched_lines})
        rec = self.add_span(
            run=run,
            text=text,
            source=str(source or "extract"),
            source_type="extract",
            kind="derived",
            media_type=parent.media_type,
            locator=locator,
            snapshot=dict(parent.snapshot or {}),
            parents=[parent.sid],
            transform=transform,
            trust="derived",
            sensitivity=parent.sensitivity,
            tags=list(tags or []) + ["derived"],
            meta={"parent_sid": parent.sid, "selector": sel, "reason": str(reason or "")},
        )
        return {
            "matched": True,
            "sid": rec.sid,
            "parent_sid": parent.sid,
            "matches": [{"line": i, "text": lines[i - 1]} for i in matched_lines],
            "span": rec.to_public_dict(include_text=False),
        }

    def _evidence_exclusion_reason(
        self,
        span: SpanRecord,
        *,
        allowed_kinds: Sequence[str],
        allow_sensitive: bool,
        include_stale: bool,
        allow_untrusted: bool,
    ) -> Optional[str]:
        allowed = {str(k).strip().lower() for k in allowed_kinds if str(k).strip()}
        if span.status != "active":
            if span.status == "stale" and include_stale:
                pass
            else:
                return f"status:{span.status}"
        if span.kind not in allowed:
            return f"kind:{span.kind}"
        if span.is_sensitive and not allow_sensitive:
            return f"sensitivity:{span.sensitivity}"
        if span.trust in {"quarantined", "untrusted"} and not allow_untrusted:
            return f"trust:{span.trust}"
        if span.kind == "derived" and not span.is_citable:
            return "derived_not_extractively_citable"
        return None

    def resolve_evidence_pack(
        self,
        *,
        run: RunState,
        sids: Sequence[str],
        max_chars: int = 12000,
        allowed_kinds: Optional[Sequence[str]] = None,
        allow_sensitive: bool = False,
        include_stale: bool = False,
        allow_untrusted: bool = False,
        include_derived_parents: bool = True,
    ) -> Dict[str, Any]:
        """Resolve run-owned SIDs into a verifier-safe evidence pack.

        The pack is fail-closed: callers get explicit exclusions for unknown,
        inactive, sensitive, non-evidence, or non-extractive derived spans.
        """

        input_sids: List[str] = []
        for raw in sids or []:
            sid = str(raw or "").strip()
            if sid and sid not in input_sids:
                input_sids.append(sid)

        allowed = list(allowed_kinds or ["evidence", "observation", "derived"])
        excluded: List[Dict[str, str]] = []
        materialized: List[SpanRecord] = []
        seen: set[str] = set()

        def add_candidate(sid: str, *, cited_by: Optional[str] = None) -> None:
            if sid in seen:
                return
            if sid not in run.spans:
                excluded.append({"sid": sid, "reason": "unknown"})
                return
            span = run.spans[sid]
            reason = self._evidence_exclusion_reason(
                span,
                allowed_kinds=allowed,
                allow_sensitive=allow_sensitive,
                include_stale=include_stale,
                allow_untrusted=allow_untrusted,
            )
            if reason:
                item = {"sid": sid, "reason": reason}
                if cited_by:
                    item["cited_by"] = cited_by
                excluded.append(item)
                # If a non-citable derived summary is cited, surface its primary
                # parents so the user can audit against primary evidence instead.
                if include_derived_parents and span.kind == "derived":
                    for parent in span.parents:
                        add_candidate(parent, cited_by=sid)
                return
            seen.add(sid)
            materialized.append(span)

        for sid in input_sids:
            add_candidate(sid)

        pack_spans: List[Dict[str, str]] = []
        text_parts: List[str] = []
        chars_used = 0
        truncated = False
        cap = max(500, int(max_chars or 12000))
        for span in materialized:
            block_prefix = f"[{span.sid}] "
            block = block_prefix + span.text
            if chars_used + len(block) > cap:
                remaining = cap - chars_used - len(block_prefix) - len("\n...[TRUNCATED SPAN]")
                if remaining <= 80:
                    excluded.append({"sid": span.sid, "reason": "prompt_budget_exceeded"})
                    truncated = True
                    continue
                span_text = span.text[:remaining].rstrip() + "\n...[TRUNCATED SPAN]"
                block = block_prefix + span_text
                truncated = True
            else:
                span_text = span.text
            chars_used += len(block)
            pack_spans.append({"sid": span.sid, "text": span_text})
            text_parts.append(block)

        text = "\n".join(text_parts).strip()
        pack_basis = {
            "run_id": run.run_id,
            "input_sids": input_sids,
            "materialized": [
                {
                    "sid": s.sid,
                    "eid": s.eid,
                    "text_sha256": s.text_sha256,
                    "status": s.status,
                    "kind": s.kind,
                    "sensitivity": s.sensitivity,
                }
                for s in materialized
                if any(ps["sid"] == s.sid for ps in pack_spans)
            ],
            "text_sha256": _sha256_text(text),
            "max_chars": cap,
        }
        return {
            "run_id": run.run_id,
            "input_sids": input_sids,
            "materialized_sids": [s["sid"] for s in pack_spans],
            "excluded": excluded,
            "spans": pack_spans,
            "text": text,
            "chars": len(text),
            "text_sha256": _sha256_text(text),
            "pack_id": _stable_json_hash(pack_basis),
            "truncated": truncated,
            "policy": {
                "allowed_kinds": allowed,
                "allow_sensitive": bool(allow_sensitive),
                "include_stale": bool(include_stale),
                "allow_untrusted": bool(allow_untrusted),
                "include_derived_parents": bool(include_derived_parents),
                "max_chars": cap,
            },
        }

    def query_evidence(
        self,
        *,
        run: RunState,
        query: str,
        limit: int = 10,
        kinds: Optional[Sequence[str]] = None,
        source_types: Optional[Sequence[str]] = None,
        trust: Optional[Sequence[str]] = None,
        status: Optional[Sequence[str]] = None,
        include_derived: bool = False,
        include_stale: bool = False,
    ) -> List[Dict[str, Any]]:
        tokens = [t for t in re.split(r"[^a-zA-Z0-9_]+", (query or "").lower()) if t]
        if not tokens:
            return []
        kind_set = {str(k).strip().lower() for k in (kinds or []) if str(k).strip()}
        source_set = {str(k).strip().lower() for k in (source_types or []) if str(k).strip()}
        trust_set = {str(k).strip().lower() for k in (trust or []) if str(k).strip()}
        status_set = {str(k).strip().lower() for k in (status or []) if str(k).strip()}

        scored: List[tuple[float, SpanRecord, List[str]]] = []
        for sid in run.span_order:
            rec = run.spans[sid]
            if rec.kind == "derived" and not include_derived:
                continue
            if rec.status != "active" and not include_stale:
                continue
            if kind_set and rec.kind not in kind_set:
                continue
            if source_set and rec.source_type not in source_set:
                continue
            if trust_set and rec.trust not in trust_set:
                continue
            if status_set and rec.status not in status_set:
                continue
            text_l = rec.text.lower()
            matched = [tok for tok in tokens if tok in text_l]
            if not matched:
                continue
            score = sum(text_l.count(tok) for tok in matched) + (0.1 * len(set(matched)))
            scored.append((float(score), rec, matched))
        scored.sort(key=lambda item: (-item[0], item[1].sid))
        out: List[Dict[str, Any]] = []
        for score, rec, matched in scored[: max(1, int(limit or 10))]:
            item = rec.to_public_dict(include_text=False, preview_chars=220)
            item.update({"score": score, "why_matched": matched})
            out.append(item)
        return out

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
