from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .enforcement import (
    AttemptRecord,
    AuditRecord,
    ClaimEvidenceLink,
    ClaimRecord,
    EnforcementError,
    RunState,
    SpanRecord,
    clear_run_dirty,
    mark_run_dirty,
)
from .paths import ensure_berry_home

LEDGER_SCHEMA_VERSION = 4
_SQLITE_USER_VERSION = 4
_TABLE_SPECS = {
    "spans": ("sid", "span"),
    "attempts": ("attempt_id", "attempt"),
    "claims": ("cid", "claim"),
    "claim_evidence_links": ("link_id", "claim/evidence link"),
    "audits": ("audit_id", "audit"),
}
_EXPORT_MODE_ENV = "BERRY_LEDGER_EXPORT_MODE"
_WRITE_CONNECTIONS: Dict[Tuple[int, str], sqlite3.Connection] = {}
_SCHEMA_READY: set[Tuple[int, str]] = set()


# ---------------------------------------------------------------------------
# Paths and durable file helpers
# ---------------------------------------------------------------------------


def runs_dir() -> Path:
    d = ensure_berry_home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(run_id: str) -> Path:
    rid = str(run_id or "").strip()
    if not rid:
        raise EnforcementError("run_id is required")
    d = runs_dir() / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_json_path(run_id: str) -> Path:
    return run_dir(run_id) / "run.json"


def run_sqlite_path(run_id: str) -> Path:
    return run_dir(run_id) / "run.sqlite"


def ledger_head_path(run_id: str) -> Path:
    return run_dir(run_id) / "ledger_head.json"


def evidence_tsv_path(run_id: str) -> Path:
    return run_dir(run_id) / "evidence.tsv"


def attempts_tsv_path(run_id: str) -> Path:
    return run_dir(run_id) / "attempts.tsv"


def claims_tsv_path(run_id: str) -> Path:
    return run_dir(run_id) / "claims.tsv"


def claim_evidence_tsv_path(run_id: str) -> Path:
    return run_dir(run_id) / "claim_evidence.tsv"


def audits_tsv_path(run_id: str) -> Path:
    return run_dir(run_id) / "audits.tsv"


def ledger_events_jsonl_path(run_id: str) -> Path:
    return run_dir(run_id) / "ledger_events.jsonl"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _append_text_durable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="surrogatepass")).hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _json_loads_object(raw: str, *, context: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except Exception as exc:
        raise EnforcementError(f"Invalid JSON in {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnforcementError(f"Invalid JSON in {context}: expected object")
    return value


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def span_to_payload(rec: SpanRecord) -> Dict[str, Any]:
    transform = dict(rec.transform or {}) if isinstance(rec.transform, dict) else rec.transform
    return {
        "sid": rec.sid,
        "text": rec.text,
        "source": rec.source,
        "created_at": float(rec.created_at),
        "meta": dict(rec.meta or {}),
        "eid": rec.eid,
        "kind": rec.kind,
        "source_type": rec.source_type,
        "media_type": rec.media_type,
        "text_sha256": rec.text_sha256,
        "locator": dict(rec.locator or {}),
        "snapshot": dict(rec.snapshot or {}),
        "parents": list(rec.parents or []),
        "transform": transform,
        "trust": rec.trust,
        "status": rec.status,
        "sensitivity": rec.sensitivity,
        "tags": list(rec.tags or []),
    }


def attempt_to_payload(rec: AttemptRecord) -> Dict[str, Any]:
    return {
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
        "git_state": rec.git_state,
        "objective_metric": rec.objective_metric,
        "objective_value": rec.objective_value,
        "result_summary": rec.result_summary,
        "next_step": rec.next_step,
    }


def claim_to_payload(rec: ClaimRecord) -> Dict[str, Any]:
    return rec.to_public_dict()


def claim_evidence_to_payload(rec: ClaimEvidenceLink) -> Dict[str, Any]:
    return rec.to_public_dict()


def audit_to_payload(rec: AuditRecord) -> Dict[str, Any]:
    return rec.to_public_dict()


def run_meta_to_payload(run: RunState) -> Dict[str, Any]:
    """Small SQLite run payload committed on every hot write.

    Child objects live in normalized tables.  Keeping this payload small is what
    removes the previous O(n²) append behavior: adding S599 no longer requires
    serializing S0..S598 into the authoritative ``runs`` row.
    """

    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "payload_kind": "ledger_head",
        "run_id": run.run_id,
        "created_at": float(run.created_at),
        "deliverable_sid": run.deliverable_sid,
        "next_span_idx": int(run.next_span_idx),
        "next_attempt_idx": int(run.next_attempt_idx),
        "next_claim_idx": int(run.next_claim_idx),
        "next_claim_link_idx": int(run.next_claim_link_idx),
        "next_audit_idx": int(run.next_audit_idx),
        "spans_version": int(run.spans_version),
        "baseline_kind": run.baseline_kind,
        "baseline_ref": run.baseline_ref,
        "counts": {
            "spans": len(run.spans),
            "attempts": len(run.attempts),
            "claims": len(run.claims),
            "claim_evidence_links": len(run.claim_evidence_links),
            "audits": len(run.audits),
        },
        "ledger": {
            "source_of_truth": "run.sqlite",
            "sqlite_schema_version": _SQLITE_USER_VERSION,
            "integrity": "incremental_row_hash_event_chain",
        },
    }


def run_to_payload(run: RunState) -> Dict[str, Any]:
    """Full compatibility snapshot used for explicit/sync exports and legacy migration."""

    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "payload_kind": "full_export",
        "run_id": run.run_id,
        "created_at": float(run.created_at),
        "deliverable_sid": run.deliverable_sid,
        "next_span_idx": int(run.next_span_idx),
        "next_attempt_idx": int(run.next_attempt_idx),
        "next_claim_idx": int(run.next_claim_idx),
        "next_claim_link_idx": int(run.next_claim_link_idx),
        "next_audit_idx": int(run.next_audit_idx),
        "spans_version": int(run.spans_version),
        "baseline_kind": run.baseline_kind,
        "baseline_ref": run.baseline_ref,
        "span_order": list(run.span_order),
        "claim_order": list(run.claim_order),
        "spans": {sid: span_to_payload(rec) for sid, rec in run.spans.items()},
        "attempts": [attempt_to_payload(rec) for rec in run.attempts],
        "claims": {cid: claim_to_payload(rec) for cid, rec in run.claims.items()},
        "claim_evidence_links": [
            claim_evidence_to_payload(rec) for rec in run.claim_evidence_links
        ],
        "audits": [audit_to_payload(rec) for rec in run.audits],
        "ledger": {
            "source_of_truth": "run.sqlite",
            "sqlite_schema_version": _SQLITE_USER_VERSION,
        },
    }


def _numeric_suffix(identifier: str, prefix: str) -> Optional[int]:
    value = str(identifier or "").strip()
    if value.startswith(prefix) and value[len(prefix) :].isdigit():
        return int(value[len(prefix) :])
    return None


def _bump_counter_from_ids(current: int, ids: Iterable[str], prefix: str) -> int:
    out = int(current or 0)
    for identifier in ids:
        suffix = _numeric_suffix(identifier, prefix)
        if suffix is not None:
            out = max(out, suffix + 1)
    return out


def run_from_payload(
    raw: Dict[str, Any], *, fallback_run_id: Optional[str] = None, mark_clean: bool = True
) -> RunState:
    rid = str(raw.get("run_id") or fallback_run_id or "").strip()
    if not rid:
        raise EnforcementError("Invalid persisted run: missing run_id")

    run = RunState(run_id=rid, created_at=float(raw.get("created_at") or time.time()))
    run.deliverable_sid = raw.get("deliverable_sid")
    run.next_span_idx = int(raw.get("next_span_idx") or 0)
    run.next_attempt_idx = int(raw.get("next_attempt_idx") or 0)
    run.next_claim_idx = int(raw.get("next_claim_idx") or 0)
    run.next_claim_link_idx = int(raw.get("next_claim_link_idx") or 0)
    run.next_audit_idx = int(raw.get("next_audit_idx") or 0)
    run.spans_version = int(raw.get("spans_version") or 0)
    run.baseline_kind = str(raw.get("baseline_kind") or "fs")
    run.baseline_ref = raw.get("baseline_ref")

    spans_raw = raw.get("spans") or {}
    if isinstance(spans_raw, dict):
        span_iter: Iterable[tuple[str, Any]] = spans_raw.items()
    elif isinstance(spans_raw, list):
        span_iter = [
            (str(rec.get("sid") or f"S{i}"), rec)
            for i, rec in enumerate(spans_raw)
            if isinstance(rec, dict)
        ]
    else:
        span_iter = []
    for sid, rec in span_iter:
        if not isinstance(rec, dict):
            continue
        s = SpanRecord(
            sid=str(rec.get("sid") or sid),
            text=str(rec.get("text") or ""),
            source=str(rec.get("source") or rec.get("source_type") or "manual"),
            created_at=float(rec.get("created_at") or time.time()),
            meta=dict(rec.get("meta") or {}),
            eid=str(rec.get("eid") or ""),
            kind=str(rec.get("kind") or ""),
            source_type=str(rec.get("source_type") or rec.get("source") or "manual"),
            media_type=str(rec.get("media_type") or "text/plain"),
            text_sha256=str(rec.get("text_sha256") or ""),
            locator=dict(rec.get("locator") or {}),
            snapshot=dict(rec.get("snapshot") or {}),
            parents=[str(v).strip() for v in (rec.get("parents") or []) if str(v).strip()],
            transform=(
                dict(rec.get("transform") or {})
                if isinstance(rec.get("transform"), dict)
                else rec.get("transform")
            ),
            trust=str(rec.get("trust") or ""),
            status=str(rec.get("status") or "active"),
            sensitivity=str(rec.get("sensitivity") or "unknown"),
            tags=[str(v).strip() for v in (rec.get("tags") or []) if str(v).strip()],
        )
        if s.text.strip():
            run.spans[s.sid] = s

    order = raw.get("span_order")
    if isinstance(order, list):
        run.span_order = [str(x) for x in order if str(x) in run.spans]
    else:
        run.span_order = sorted(
            run.spans.keys(),
            key=lambda sid: (
                _numeric_suffix(sid, "S") is None,
                _numeric_suffix(sid, "S") or 0,
                sid,
            ),
        )
    for sid in run.spans:
        if sid not in run.span_order:
            run.span_order.append(sid)
    run.next_span_idx = _bump_counter_from_ids(run.next_span_idx, run.spans.keys(), "S")

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
                    git_state=str(rec.get("git_state") or ""),
                    objective_metric=str(rec.get("objective_metric") or ""),
                    objective_value=str(rec.get("objective_value") or ""),
                    result_summary=str(rec.get("result_summary") or ""),
                    next_step=str(rec.get("next_step") or ""),
                )
            )
    run.next_attempt_idx = _bump_counter_from_ids(
        run.next_attempt_idx, [a.attempt_id for a in run.attempts], "A"
    )

    claims_raw = raw.get("claims") or {}
    if isinstance(claims_raw, dict):
        claim_iter: Iterable[tuple[str, Any]] = claims_raw.items()
    elif isinstance(claims_raw, list):
        claim_iter = [
            (str(rec.get("cid") or rec.get("claim_id") or f"C{i}"), rec)
            for i, rec in enumerate(claims_raw)
            if isinstance(rec, dict)
        ]
    else:
        claim_iter = []
    for cid, rec in claim_iter:
        if not isinstance(rec, dict):
            continue
        text = str(rec.get("text") or rec.get("claim") or "").strip()
        if not text:
            continue
        c = ClaimRecord(
            cid=str(rec.get("cid") or rec.get("claim_id") or cid),
            text=text,
            kind=str(rec.get("kind") or "fact"),
            status=str(rec.get("status") or "open"),
            target=float(rec.get("target") or rec.get("confidence") or 0.95),
            created_at=float(rec.get("created_at") or time.time()),
            updated_at=float(rec.get("updated_at") or rec.get("created_at") or time.time()),
            source=str(rec.get("source") or "manual"),
            latest_audit_id=str(rec.get("latest_audit_id") or ""),
            tags=[str(v).strip() for v in (rec.get("tags") or []) if str(v).strip()],
            meta=dict(rec.get("meta") or {}),
        )
        run.claims[c.cid] = c
    claim_order = raw.get("claim_order")
    if isinstance(claim_order, list):
        run.claim_order = [str(x) for x in claim_order if str(x) in run.claims]
    else:
        run.claim_order = sorted(
            run.claims.keys(),
            key=lambda cid: (
                _numeric_suffix(cid, "C") is None,
                _numeric_suffix(cid, "C") or 0,
                cid,
            ),
        )
    for cid in run.claims:
        if cid not in run.claim_order:
            run.claim_order.append(cid)
    run.next_claim_idx = _bump_counter_from_ids(run.next_claim_idx, run.claims.keys(), "C")

    links_raw = raw.get("claim_evidence_links") or raw.get("claim_links") or []
    if isinstance(links_raw, list):
        for rec in links_raw:
            if not isinstance(rec, dict):
                continue
            cid = str(rec.get("cid") or rec.get("claim_id") or "").strip()
            sid = str(rec.get("sid") or "").strip()
            if cid not in run.claims or sid not in run.spans:
                continue
            run.claim_evidence_links.append(
                ClaimEvidenceLink(
                    link_id=str(rec.get("link_id") or f"L{len(run.claim_evidence_links)}"),
                    cid=cid,
                    sid=sid,
                    relation=str(rec.get("relation") or "supports"),
                    created_at=float(rec.get("created_at") or time.time()),
                    created_by=str(rec.get("created_by") or "manual"),
                    audit_id=str(rec.get("audit_id") or ""),
                    note=str(rec.get("note") or ""),
                    meta=dict(rec.get("meta") or {}),
                )
            )
    run.next_claim_link_idx = _bump_counter_from_ids(
        run.next_claim_link_idx, [link.link_id for link in run.claim_evidence_links], "L"
    )

    audits_raw = raw.get("audits") or []
    if isinstance(audits_raw, list):
        for rec in audits_raw:
            if not isinstance(rec, dict):
                continue
            run.audits.append(
                AuditRecord(
                    audit_id=str(rec.get("audit_id") or f"V{len(run.audits)}"),
                    kind=str(rec.get("kind") or "manual"),
                    created_at=float(rec.get("created_at") or time.time()),
                    claim_ids=[
                        str(v).strip()
                        for v in (rec.get("claim_ids") or [])
                        if str(v).strip() in run.claims
                    ],
                    input_sids=[
                        str(v).strip()
                        for v in (rec.get("input_sids") or [])
                        if str(v).strip() in run.spans
                    ],
                    materialized_sids=[
                        str(v).strip()
                        for v in (rec.get("materialized_sids") or [])
                        if str(v).strip() in run.spans
                    ],
                    evidence_pack_id=str(rec.get("evidence_pack_id") or ""),
                    evidence_pack_hash=str(rec.get("evidence_pack_hash") or ""),
                    verifier_model=str(rec.get("verifier_model") or ""),
                    policy=dict(rec.get("policy") or {}),
                    result=dict(rec.get("result") or {}),
                    audit_sid=str(rec.get("audit_sid") or ""),
                    meta=dict(rec.get("meta") or {}),
                )
            )
    run.next_audit_idx = _bump_counter_from_ids(
        run.next_audit_idx, [audit.audit_id for audit in run.audits], "V"
    )
    if mark_clean:
        clear_run_dirty(run)
    else:
        mark_run_dirty(run, all=True)
    return run


# ---------------------------------------------------------------------------
# SQLite source-of-truth with incremental tamper-evident events
# ---------------------------------------------------------------------------


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _connect_for_write(path: Path) -> sqlite3.Connection:
    key = (os.getpid(), str(path.resolve()))
    con = _WRITE_CONNECTIONS.get(key)
    if con is not None and not path.exists():
        try:
            con.close()
        except Exception:
            pass
        _WRITE_CONNECTIONS.pop(key, None)
        _SCHEMA_READY.discard(key)
        con = None
    if con is None:
        con = _connect(path)
        _WRITE_CONNECTIONS[key] = con
    if key not in _SCHEMA_READY:
        _ensure_schema(con)
        _SCHEMA_READY.add(key)
    return con


def close_cached_connections() -> None:
    for con in list(_WRITE_CONNECTIONS.values()):
        try:
            con.close()
        except Exception:
            pass
    _WRITE_CONNECTIONS.clear()
    _SCHEMA_READY.clear()


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}


def _ensure_column(con: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    if column not in _table_columns(con, table):
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            head_event_hash TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS spans (
            run_id TEXT NOT NULL,
            sid TEXT NOT NULL,
            position INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL DEFAULT '',
            text_sha256 TEXT NOT NULL,
            eid TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            source_type TEXT NOT NULL,
            PRIMARY KEY (run_id, sid),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_spans_run_kind_status ON spans(run_id, kind, status);
        CREATE TABLE IF NOT EXISTS attempts (
            run_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (run_id, attempt_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS claims (
            run_id TEXT NOT NULL,
            cid TEXT NOT NULL,
            position INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            target REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (run_id, cid),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_claims_run_status ON claims(run_id, status, kind);
        CREATE TABLE IF NOT EXISTS claim_evidence_links (
            run_id TEXT NOT NULL,
            link_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL DEFAULT '',
            cid TEXT NOT NULL,
            sid TEXT NOT NULL,
            relation TEXT NOT NULL,
            audit_id TEXT NOT NULL,
            PRIMARY KEY (run_id, link_id),
            FOREIGN KEY (run_id, cid) REFERENCES claims(run_id, cid) ON DELETE CASCADE,
            FOREIGN KEY (run_id, sid) REFERENCES spans(run_id, sid) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_claim_links_claim
            ON claim_evidence_links(run_id, cid, relation);
        CREATE INDEX IF NOT EXISTS idx_claim_links_span
            ON claim_evidence_links(run_id, sid, relation);
        CREATE TABLE IF NOT EXISTS audits (
            run_id TEXT NOT NULL,
            audit_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            created_at REAL NOT NULL,
            evidence_pack_id TEXT NOT NULL,
            evidence_pack_hash TEXT NOT NULL,
            PRIMARY KEY (run_id, audit_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_audits_run_kind ON audits(run_id, kind, created_at);
        CREATE TABLE IF NOT EXISTS ledger_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ts REAL NOT NULL,
            event_type TEXT NOT NULL,
            object_type TEXT NOT NULL,
            object_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            prev_event_hash TEXT NOT NULL,
            event_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ledger_events_run ON ledger_events(run_id, event_id);
        PRAGMA user_version = 4;
        """
    )
    # Existing schema-3 ledgers need additive columns.  Keep these ALTERs outside
    # the CREATE block so repeated starts are cheap and migration is automatic.
    _ensure_column(con, "runs", "head_event_hash", "TEXT NOT NULL DEFAULT ''")
    for table in _TABLE_SPECS:
        _ensure_column(con, table, "payload_sha256", "TEXT NOT NULL DEFAULT ''")
    con.execute(f"PRAGMA user_version = {_SQLITE_USER_VERSION}")


def _append_event(
    con: sqlite3.Connection,
    *,
    run_id: str,
    event_type: str,
    object_type: str,
    object_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload_json = _json_dumps(payload)
    payload_sha = _sha256_text(payload_json)
    row = con.execute(
        "SELECT event_hash FROM ledger_events WHERE run_id = ? ORDER BY event_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    prev = str(row["event_hash"] if row else "GENESIS")
    ts = time.time()
    event_hash = _sha256_text(
        _json_dumps(
            {
                "run_id": run_id,
                "ts": ts,
                "event_type": event_type,
                "object_type": object_type,
                "object_id": object_id,
                "payload_sha256": payload_sha,
                "prev_event_hash": prev,
            }
        )
    )
    cur = con.execute(
        """
        INSERT INTO ledger_events(
            run_id, ts, event_type, object_type, object_id,
            payload_json, payload_sha256, prev_event_hash, event_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            ts,
            event_type,
            object_type,
            object_id,
            payload_json,
            payload_sha,
            prev,
            event_hash,
        ),
    )
    return {
        "event_id": int(cur.lastrowid or 0),
        "run_id": run_id,
        "ts": ts,
        "event_type": event_type,
        "object_type": object_type,
        "object_id": object_id,
        "payload": payload,
        "payload_sha256": payload_sha,
        "prev_event_hash": prev,
        "event_hash": event_hash,
    }


def _counts(run: RunState) -> Dict[str, int]:
    return {
        "spans": len(run.spans),
        "attempts": len(run.attempts),
        "claims": len(run.claims),
        "claim_evidence_links": len(run.claim_evidence_links),
        "audits": len(run.audits),
    }


def _position_map(values: Iterable[str]) -> Dict[str, int]:
    return {str(v): pos for pos, v in enumerate(values)}


def _attempt_position_map(run: RunState) -> Dict[str, int]:
    return {rec.attempt_id: pos for pos, rec in enumerate(run.attempts)}


def _link_position_map(run: RunState) -> Dict[str, int]:
    return {rec.link_id: pos for pos, rec in enumerate(run.claim_evidence_links)}


def _audit_position_map(run: RunState) -> Dict[str, int]:
    return {rec.audit_id: pos for pos, rec in enumerate(run.audits)}


def _db_counts(con: sqlite3.Connection, run_id: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for table in _TABLE_SPECS:
        out[table] = int(
            con.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)).fetchone()[0]
        )
    return out


def _run_row_exists(con: sqlite3.Connection, run_id: str) -> bool:
    return con.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is not None


def _row_payload_sha(payload: Dict[str, Any]) -> Tuple[str, str]:
    payload_json = _json_dumps(payload)
    return payload_json, _sha256_text(payload_json)


def _upsert_span(con: sqlite3.Connection, run: RunState, sid: str, position: int) -> Dict[str, Any]:
    span = run.spans.get(sid)
    if span is None:
        con.execute("DELETE FROM spans WHERE run_id = ? AND sid = ?", (run.run_id, sid))
        return {"table": "spans", "id": sid, "op": "delete"}
    sp = span_to_payload(span)
    payload_json, payload_sha = _row_payload_sha(sp)
    con.execute(
        """
        INSERT INTO spans(
            run_id, sid, position, payload_json, payload_sha256, text_sha256,
            eid, kind, status, source_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, sid) DO UPDATE SET
            position=excluded.position,
            payload_json=excluded.payload_json,
            payload_sha256=excluded.payload_sha256,
            text_sha256=excluded.text_sha256,
            eid=excluded.eid,
            kind=excluded.kind,
            status=excluded.status,
            source_type=excluded.source_type
        """,
        (
            run.run_id,
            span.sid,
            position,
            payload_json,
            payload_sha,
            span.text_sha256,
            span.eid,
            span.kind,
            span.status,
            span.source_type,
        ),
    )
    return {
        "table": "spans",
        "id": sid,
        "op": "upsert",
        "payload_sha256": payload_sha,
        "position": position,
    }


def _upsert_attempt(
    con: sqlite3.Connection,
    run: RunState,
    attempt_id: str,
    by_id: Dict[str, AttemptRecord],
    position: int,
) -> Dict[str, Any]:
    rec = by_id.get(attempt_id)
    if rec is None:
        con.execute(
            "DELETE FROM attempts WHERE run_id = ? AND attempt_id = ?", (run.run_id, attempt_id)
        )
        return {"table": "attempts", "id": attempt_id, "op": "delete"}
    payload_json, payload_sha = _row_payload_sha(attempt_to_payload(rec))
    con.execute(
        """
        INSERT INTO attempts(run_id, attempt_id, position, payload_json, payload_sha256)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(run_id, attempt_id) DO UPDATE SET
            position=excluded.position,
            payload_json=excluded.payload_json,
            payload_sha256=excluded.payload_sha256
        """,
        (run.run_id, rec.attempt_id, position, payload_json, payload_sha),
    )
    return {
        "table": "attempts",
        "id": attempt_id,
        "op": "upsert",
        "payload_sha256": payload_sha,
        "position": position,
    }


def _upsert_claim(
    con: sqlite3.Connection, run: RunState, cid: str, position: int
) -> Dict[str, Any]:
    claim = run.claims.get(cid)
    if claim is None:
        con.execute("DELETE FROM claims WHERE run_id = ? AND cid = ?", (run.run_id, cid))
        return {"table": "claims", "id": cid, "op": "delete"}
    cp = claim_to_payload(claim)
    payload_json, payload_sha = _row_payload_sha(cp)
    con.execute(
        """
        INSERT INTO claims(
            run_id, cid, position, payload_json, payload_sha256, kind, status, target, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, cid) DO UPDATE SET
            position=excluded.position,
            payload_json=excluded.payload_json,
            payload_sha256=excluded.payload_sha256,
            kind=excluded.kind,
            status=excluded.status,
            target=excluded.target,
            updated_at=excluded.updated_at
        """,
        (
            run.run_id,
            claim.cid,
            position,
            payload_json,
            payload_sha,
            claim.kind,
            claim.status,
            claim.target,
            claim.updated_at,
        ),
    )
    return {
        "table": "claims",
        "id": cid,
        "op": "upsert",
        "payload_sha256": payload_sha,
        "position": position,
    }


def _upsert_link(
    con: sqlite3.Connection,
    run: RunState,
    link_id: str,
    by_id: Dict[str, ClaimEvidenceLink],
    position: int,
) -> Dict[str, Any]:
    link = by_id.get(link_id)
    if link is None:
        con.execute(
            "DELETE FROM claim_evidence_links WHERE run_id = ? AND link_id = ?",
            (run.run_id, link_id),
        )
        return {"table": "claim_evidence_links", "id": link_id, "op": "delete"}
    if link.cid not in run.claims or link.sid not in run.spans:
        raise EnforcementError(
            f"Cannot persist dangling claim/evidence link {link.link_id}: {link.cid}->{link.sid}"
        )
    lp = claim_evidence_to_payload(link)
    payload_json, payload_sha = _row_payload_sha(lp)
    con.execute(
        """
        INSERT INTO claim_evidence_links(
            run_id, link_id, position, payload_json, payload_sha256, cid, sid, relation, audit_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, link_id) DO UPDATE SET
            position=excluded.position,
            payload_json=excluded.payload_json,
            payload_sha256=excluded.payload_sha256,
            cid=excluded.cid,
            sid=excluded.sid,
            relation=excluded.relation,
            audit_id=excluded.audit_id
        """,
        (
            run.run_id,
            link.link_id,
            position,
            payload_json,
            payload_sha,
            link.cid,
            link.sid,
            link.relation,
            link.audit_id,
        ),
    )
    return {
        "table": "claim_evidence_links",
        "id": link_id,
        "op": "upsert",
        "payload_sha256": payload_sha,
        "position": position,
    }


def _upsert_audit(
    con: sqlite3.Connection,
    run: RunState,
    audit_id: str,
    by_id: Dict[str, AuditRecord],
    position: int,
) -> Dict[str, Any]:
    audit = by_id.get(audit_id)
    if audit is None:
        con.execute("DELETE FROM audits WHERE run_id = ? AND audit_id = ?", (run.run_id, audit_id))
        return {"table": "audits", "id": audit_id, "op": "delete"}
    dangling_claims = [cid for cid in audit.claim_ids if cid not in run.claims]
    dangling_sids = [
        sid for sid in audit.input_sids + audit.materialized_sids if sid not in run.spans
    ]
    if dangling_claims or dangling_sids:
        raise EnforcementError(
            f"Cannot persist dangling audit {audit.audit_id}: claims={dangling_claims}, sids={dangling_sids}"
        )
    ap = audit_to_payload(audit)
    payload_json, payload_sha = _row_payload_sha(ap)
    con.execute(
        """
        INSERT INTO audits(
            run_id, audit_id, position, payload_json, payload_sha256, kind, created_at,
            evidence_pack_id, evidence_pack_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, audit_id) DO UPDATE SET
            position=excluded.position,
            payload_json=excluded.payload_json,
            payload_sha256=excluded.payload_sha256,
            kind=excluded.kind,
            created_at=excluded.created_at,
            evidence_pack_id=excluded.evidence_pack_id,
            evidence_pack_hash=excluded.evidence_pack_hash
        """,
        (
            run.run_id,
            audit.audit_id,
            position,
            payload_json,
            payload_sha,
            audit.kind,
            audit.created_at,
            audit.evidence_pack_id,
            audit.evidence_pack_hash,
        ),
    )
    return {
        "table": "audits",
        "id": audit_id,
        "op": "upsert",
        "payload_sha256": payload_sha,
        "position": position,
    }


def _delete_all_child_rows(con: sqlite3.Connection, run_id: str) -> None:
    for table in ["claim_evidence_links", "audits", "attempts", "claims", "spans"]:
        con.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))


def _collect_and_apply_changes(
    con: sqlite3.Connection, run: RunState, *, full_snapshot: bool
) -> List[Dict[str, Any]]:
    if full_snapshot:
        _delete_all_child_rows(con, run.run_id)

    span_pos = _position_map(run.span_order)
    attempt_by_id = {rec.attempt_id: rec for rec in run.attempts}
    attempt_pos = _attempt_position_map(run)
    claim_pos = _position_map(run.claim_order)
    link_by_id = {rec.link_id: rec for rec in run.claim_evidence_links}
    link_pos = _link_position_map(run)
    audit_by_id = {rec.audit_id: rec for rec in run.audits}
    audit_pos = _audit_position_map(run)

    if full_snapshot:
        span_ids = list(run.span_order)
        attempt_ids = [rec.attempt_id for rec in run.attempts]
        claim_ids = list(run.claim_order)
        link_ids = [rec.link_id for rec in run.claim_evidence_links]
        audit_ids = [rec.audit_id for rec in run.audits]
    else:
        span_ids = sorted(run.ledger_dirty_spans, key=lambda sid: span_pos.get(sid, 10**12))
        attempt_ids = sorted(
            run.ledger_dirty_attempts, key=lambda aid: attempt_pos.get(aid, 10**12)
        )
        claim_ids = sorted(run.ledger_dirty_claims, key=lambda cid: claim_pos.get(cid, 10**12))
        link_ids = sorted(run.ledger_dirty_claim_links, key=lambda lid: link_pos.get(lid, 10**12))
        audit_ids = sorted(run.ledger_dirty_audits, key=lambda aid: audit_pos.get(aid, 10**12))

    changes: List[Dict[str, Any]] = []
    for sid in span_ids:
        changes.append(_upsert_span(con, run, sid, span_pos.get(sid, -1)))
    for attempt_id in attempt_ids:
        changes.append(
            _upsert_attempt(con, run, attempt_id, attempt_by_id, attempt_pos.get(attempt_id, -1))
        )
    for cid in claim_ids:
        changes.append(_upsert_claim(con, run, cid, claim_pos.get(cid, -1)))
    for link_id in link_ids:
        changes.append(_upsert_link(con, run, link_id, link_by_id, link_pos.get(link_id, -1)))
    for audit_id in audit_ids:
        changes.append(_upsert_audit(con, run, audit_id, audit_by_id, audit_pos.get(audit_id, -1)))
    return changes


def _should_force_full(con: sqlite3.Connection, run: RunState) -> bool:
    if bool(run.ledger_dirty_all):
        return True
    if not _run_row_exists(con, run.run_id):
        return True
    # This cheap count check catches direct append/delete mutations that bypassed
    # RunStore's dirty sets.  Normal appends have explicit dirty child IDs and
    # should remain O(1), even though the committed DB count is temporarily lower
    # than the in-memory count until this transaction applies the new row.
    has_child_dirty = bool(
        run.ledger_dirty_spans
        or run.ledger_dirty_attempts
        or run.ledger_dirty_claims
        or run.ledger_dirty_claim_links
        or run.ledger_dirty_audits
    )
    if has_child_dirty:
        return False
    try:
        return _db_counts(con, run.run_id) != _counts(run)
    except Exception:
        return True


def _has_pending_dirty_rows(run: RunState) -> bool:
    return bool(
        run.ledger_dirty_all
        or run.ledger_dirty_meta
        or run.ledger_dirty_spans
        or run.ledger_dirty_attempts
        or run.ledger_dirty_claims
        or run.ledger_dirty_claim_links
        or run.ledger_dirty_audits
    )


def _dirty_ids_by_table(run: RunState) -> Dict[str, set[str]]:
    return {
        "spans": set(run.ledger_dirty_spans),
        "attempts": set(run.ledger_dirty_attempts),
        "claims": set(run.ledger_dirty_claims),
        "claim_evidence_links": set(run.ledger_dirty_claim_links),
        "audits": set(run.ledger_dirty_audits),
    }


def _current_run_row_hashes(run: RunState) -> Dict[str, Dict[str, str]]:
    hashes: Dict[str, Dict[str, str]] = {table: {} for table in _TABLE_SPECS}
    for sid, span in run.spans.items():
        _payload_json, payload_sha = _row_payload_sha(span_to_payload(span))
        hashes["spans"][sid] = payload_sha
    for rec in run.attempts:
        _payload_json, payload_sha = _row_payload_sha(attempt_to_payload(rec))
        hashes["attempts"][rec.attempt_id] = payload_sha
    for cid, claim in run.claims.items():
        _payload_json, payload_sha = _row_payload_sha(claim_to_payload(claim))
        hashes["claims"][cid] = payload_sha
    for link in run.claim_evidence_links:
        _payload_json, payload_sha = _row_payload_sha(claim_evidence_to_payload(link))
        hashes["claim_evidence_links"][link.link_id] = payload_sha
    for audit in run.audits:
        _payload_json, payload_sha = _row_payload_sha(audit_to_payload(audit))
        hashes["audits"][audit.audit_id] = payload_sha
    return hashes


def _install_committed_state(
    run: RunState,
    *,
    head_event_hash: str,
    run_meta_sha256: str,
    row_hashes: Dict[str, Dict[str, str]],
) -> None:
    run.ledger_committed_head_event_hash = str(head_event_hash or "")
    run.ledger_committed_run_meta_sha256 = str(run_meta_sha256 or "")
    run.ledger_committed_row_hashes = {
        table: {str(k): str(v) for k, v in (row_hashes.get(table) or {}).items()}
        for table in _TABLE_SPECS
    }


def _install_committed_state_from_replay(run: RunState, replay: Dict[str, Any]) -> None:
    expected = replay.get("expected_tables") or {}
    row_hashes: Dict[str, Dict[str, str]] = {table: {} for table in _TABLE_SPECS}
    for table in _TABLE_SPECS:
        for object_id, item in (expected.get(table) or {}).items():
            row_hashes[table][str(object_id)] = str(item.get("payload_sha256") or "")
    _install_committed_state(
        run,
        head_event_hash=str(replay.get("head_event_hash") or ""),
        run_meta_sha256=str(replay.get("run_meta_sha256") or ""),
        row_hashes=row_hashes,
    )


def _install_committed_state_from_event(run: RunState, event: Dict[str, Any]) -> None:
    if bool(event.get("full_snapshot")) or not run.ledger_committed_row_hashes:
        row_hashes = _current_run_row_hashes(run)
    else:
        row_hashes = {
            table: dict(run.ledger_committed_row_hashes.get(table) or {}) for table in _TABLE_SPECS
        }
        for change in event.get("changes") or []:
            if not isinstance(change, dict):
                continue
            table = str(change.get("table") or "")
            object_id = str(change.get("id") or "")
            if table not in row_hashes or not object_id:
                continue
            if str(change.get("op") or "upsert") == "delete":
                row_hashes[table].pop(object_id, None)
            else:
                row_hashes[table][object_id] = str(change.get("payload_sha256") or "")
    _install_committed_state(
        run,
        head_event_hash=str(event.get("event_hash") or ""),
        run_meta_sha256=str(event.get("run_meta_sha256") or ""),
        row_hashes=row_hashes,
    )


def _validate_write_base(con: sqlite3.Connection, run: RunState) -> None:
    """Fail closed if the on-disk base changed since this RunState was loaded.

    This keeps hot writes O(number of dirty rows).  We do not replay the full
    event log on every span append, but we do check the committed run head and
    every row that this commit is about to overwrite/delete.  Full verification
    still happens on load/export, where untouched-row tampering is detected by
    replaying the event chain and comparing the normalized tables.
    """

    row = con.execute(
        "SELECT payload_sha256, head_event_hash FROM runs WHERE run_id = ?",
        (run.run_id,),
    ).fetchone()
    if row is None:
        return
    expected_head = str(run.ledger_committed_head_event_hash or "")
    if expected_head and str(row["head_event_hash"] or "") != expected_head:
        raise EnforcementError(
            "SQLite ledger changed since this run was loaded; reload before writing"
        )
    expected_meta = str(run.ledger_committed_run_meta_sha256 or "")
    if expected_meta and str(row["payload_sha256"] or "") != expected_meta:
        raise EnforcementError(
            "SQLite run metadata changed since this run was loaded; reload before writing"
        )

    dirty = _dirty_ids_by_table(run)
    for table, (id_col, label) in _TABLE_SPECS.items():
        expected_hashes = run.ledger_committed_row_hashes.get(table) or {}
        ids = set(dirty.get(table) or set())
        if run.ledger_dirty_all:
            ids.update(expected_hashes.keys())
        if not ids:
            continue
        placeholders = ",".join("?" for _ in ids)
        rows = con.execute(
            f"SELECT {id_col} AS object_id, payload_json, payload_sha256 FROM {table} "
            f"WHERE run_id = ? AND {id_col} IN ({placeholders})",
            (run.run_id, *sorted(ids)),
        ).fetchall()
        actual = {str(r["object_id"]): r for r in rows}
        for object_id in sorted(ids):
            expected_sha = str(expected_hashes.get(object_id) or "")
            current = actual.get(object_id)
            if expected_sha:
                if current is None:
                    raise EnforcementError(
                        f"SQLite {label} row {object_id} was removed since this run was loaded"
                    )
                payload_json = str(current["payload_json"] or "")
                stored_sha = str(current["payload_sha256"] or "")
                if _sha256_text(payload_json) != stored_sha or stored_sha != expected_sha:
                    raise EnforcementError(
                        f"SQLite {label} row {object_id} changed since this run was loaded"
                    )
            elif current is not None and not run.ledger_dirty_all:
                raise EnforcementError(
                    f"SQLite {label} row {object_id} already exists; reload before writing"
                )


def _mark_dirty_from_db_diff(con: sqlite3.Connection, run: RunState) -> None:
    """Detect unmarked direct RunState edits without slowing the normal hot path.

    RunStore marks changed rows explicitly.  If a caller bypasses RunStore and
    calls persist_run with an apparently clean run, we first verify the existing
    SQLite ledger against the event log so external tampering cannot be silently
    healed by an in-memory object. Only then do we diff the clean in-memory run
    against the verified durable state.
    """

    replay = _verify_event_hash_chain_con(con, run.run_id)
    if replay.get("saw_incremental"):
        row = con.execute(
            "SELECT payload_sha256, head_event_hash FROM runs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()
        if row is None:
            mark_run_dirty(run, all=True)
            return
        if str(row["head_event_hash"] or "") != str(replay.get("head_event_hash") or ""):
            raise EnforcementError("SQLite ledger head event hash mismatch")
        if str(row["payload_sha256"] or "") != str(replay.get("run_meta_sha256") or ""):
            raise EnforcementError("SQLite ledger head event does not match run metadata")
        _verify_tables_against_event_log(con, run_id=run.run_id, replay=replay)

    span_pos = _position_map(run.span_order)
    actual_spans = _table_hashes(con, table="spans", id_col="sid", run_id=run.run_id)
    for sid, span in run.spans.items():
        _payload_json, payload_sha = _row_payload_sha(span_to_payload(span))
        expected = {"payload_sha256": payload_sha, "position": span_pos.get(sid, -1)}
        if actual_spans.get(sid) != expected:
            run.ledger_dirty_spans.add(sid)
    for sid in set(actual_spans) - set(run.spans):
        run.ledger_dirty_spans.add(sid)

    attempt_pos = _attempt_position_map(run)
    attempt_by_id = {rec.attempt_id: rec for rec in run.attempts}
    actual_attempts = _table_hashes(con, table="attempts", id_col="attempt_id", run_id=run.run_id)
    for attempt_id, rec in attempt_by_id.items():
        _payload_json, payload_sha = _row_payload_sha(attempt_to_payload(rec))
        expected = {"payload_sha256": payload_sha, "position": attempt_pos.get(attempt_id, -1)}
        if actual_attempts.get(attempt_id) != expected:
            run.ledger_dirty_attempts.add(attempt_id)
    for attempt_id in set(actual_attempts) - set(attempt_by_id):
        run.ledger_dirty_attempts.add(attempt_id)

    claim_pos = _position_map(run.claim_order)
    actual_claims = _table_hashes(con, table="claims", id_col="cid", run_id=run.run_id)
    for cid, claim in run.claims.items():
        _payload_json, payload_sha = _row_payload_sha(claim_to_payload(claim))
        expected = {"payload_sha256": payload_sha, "position": claim_pos.get(cid, -1)}
        if actual_claims.get(cid) != expected:
            run.ledger_dirty_claims.add(cid)
    for cid in set(actual_claims) - set(run.claims):
        run.ledger_dirty_claims.add(cid)

    link_pos = _link_position_map(run)
    link_by_id = {rec.link_id: rec for rec in run.claim_evidence_links}
    actual_links = _table_hashes(
        con, table="claim_evidence_links", id_col="link_id", run_id=run.run_id
    )
    for link_id, link in link_by_id.items():
        _payload_json, payload_sha = _row_payload_sha(claim_evidence_to_payload(link))
        expected = {"payload_sha256": payload_sha, "position": link_pos.get(link_id, -1)}
        if actual_links.get(link_id) != expected:
            run.ledger_dirty_claim_links.add(link_id)
    for link_id in set(actual_links) - set(link_by_id):
        run.ledger_dirty_claim_links.add(link_id)

    audit_pos = _audit_position_map(run)
    audit_by_id = {rec.audit_id: rec for rec in run.audits}
    actual_audits = _table_hashes(con, table="audits", id_col="audit_id", run_id=run.run_id)
    for audit_id, audit in audit_by_id.items():
        _payload_json, payload_sha = _row_payload_sha(audit_to_payload(audit))
        expected = {"payload_sha256": payload_sha, "position": audit_pos.get(audit_id, -1)}
        if actual_audits.get(audit_id) != expected:
            run.ledger_dirty_audits.add(audit_id)
    for audit_id in set(actual_audits) - set(audit_by_id):
        run.ledger_dirty_audits.add(audit_id)

    _meta_json, meta_sha = _row_payload_sha(run_meta_to_payload(run))
    row = con.execute("SELECT payload_sha256 FROM runs WHERE run_id = ?", (run.run_id,)).fetchone()
    if row is None or str(row["payload_sha256"] or "") != meta_sha:
        run.ledger_dirty_meta = True


def _commit_incremental(con: sqlite3.Connection, run: RunState) -> Dict[str, Any]:
    full_snapshot = _should_force_full(con, run)
    meta_payload = run_meta_to_payload(run)
    meta_json, meta_sha = _row_payload_sha(meta_payload)
    now = time.time()
    con.execute(
        """
        INSERT INTO runs(run_id, payload_json, payload_sha256, head_event_hash, created_at, updated_at)
        VALUES (?, ?, ?, '', ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            payload_json=excluded.payload_json,
            payload_sha256=excluded.payload_sha256,
            created_at=excluded.created_at,
            updated_at=excluded.updated_at
        """,
        (run.run_id, meta_json, meta_sha, float(run.created_at), now),
    )
    changes = _collect_and_apply_changes(con, run, full_snapshot=full_snapshot)
    # Persist a commit event even for metadata-only commits.  No-op calls with a
    # clean run are filtered before opening the transaction.
    event_payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "format": "incremental_v1",
        "run_id": run.run_id,
        "run_meta_sha256": meta_sha,
        "full_snapshot": bool(full_snapshot),
        "changes": changes,
        "counts": _counts(run),
        "spans_version": int(run.spans_version),
    }
    event = _append_event(
        con,
        run_id=run.run_id,
        event_type="state_committed",
        object_type="run",
        object_id=run.run_id,
        payload=event_payload,
    )
    con.execute(
        "UPDATE runs SET head_event_hash = ?, updated_at = ? WHERE run_id = ?",
        (event["event_hash"], time.time(), run.run_id),
    )
    event["changes"] = changes
    event["run_meta_sha256"] = meta_sha
    event["full_snapshot"] = bool(full_snapshot)
    return event


def _export_mode(value: Optional[str] = None) -> str:
    raw = (
        str(value if value is not None else os.environ.get(_EXPORT_MODE_ENV, "off")).strip().lower()
    )
    if raw in {"1", "true", "yes", "sync", "full"}:
        return "sync"
    if raw in {"0", "false", "no", "off", "none"}:
        return "off"
    return "hot"


def persist_run(run: RunState, *, export_mode: Optional[str] = None) -> None:
    """Persist a run using incremental SQLite commits plus lightweight exports.

    Integrity is preserved without the previous full-snapshot-per-write cost:
    each changed normalized row stores its own payload hash; each commit appends
    a hash-chained event containing the changed row hashes and current metadata
    hash; load reconstructs the expected table state by replaying the event log
    and comparing it to the SQLite rows. Default commits write only SQLite.
    Set ``BERRY_LEDGER_EXPORT_MODE=hot`` for append-only inspection mirrors or
    ``BERRY_LEDGER_EXPORT_MODE=sync`` to regenerate
    full JSON/TSV snapshots on every commit for legacy consumers.
    """

    try:
        mode = _export_mode(export_mode)
        db_path = run_sqlite_path(run.run_id)
        event: Optional[Dict[str, Any]] = None
        # If the run is clean, still check for a missing DB.  This happens when a
        # caller loads an object from a legacy JSON file and asks to persist after
        # deleting the SQLite ledger.
        con = _connect_for_write(db_path)
        try:
            if not _has_pending_dirty_rows(run) and _run_row_exists(con, run.run_id):
                _mark_dirty_from_db_diff(con, run)
                if not _has_pending_dirty_rows(run):
                    return
            con.execute("BEGIN IMMEDIATE")
            _validate_write_base(con, run)
            event = _commit_incremental(con, run)
            con.commit()
            _install_committed_state_from_event(run, event)
            clear_run_dirty(run)
        except Exception:
            try:
                con.rollback()
            except Exception:
                pass
            raise
        if mode == "sync":
            _write_full_exports(run)
        elif mode == "hot":
            assert event is not None
            _write_hot_exports(run, event)
    except EnforcementError:
        raise
    except Exception as exc:
        raise EnforcementError(
            f"Failed to persist run ledger: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Verification and loading
# ---------------------------------------------------------------------------


def _verify_event_hash_chain_con(con: sqlite3.Connection, run_id: str) -> Dict[str, Any]:
    rows = con.execute(
        """
        SELECT event_id, run_id, ts, event_type, object_type, object_id,
               payload_json, payload_sha256, prev_event_hash, event_hash
        FROM ledger_events WHERE run_id = ? ORDER BY event_id ASC
        """,
        (run_id,),
    ).fetchall()
    prev = "GENESIS"
    expected: Dict[str, Dict[str, Dict[str, Any]]] = {table: {} for table in _TABLE_SPECS}
    latest_meta_sha = ""
    saw_incremental = False
    for row in rows:
        payload_json = str(row["payload_json"] or "")
        payload_sha = _sha256_text(payload_json)
        if payload_sha != row["payload_sha256"]:
            raise EnforcementError(f"Ledger event {row['event_id']} payload hash mismatch")
        if str(row["prev_event_hash"] or "") != prev:
            raise EnforcementError(f"Ledger event {row['event_id']} previous hash mismatch")
        expected_event_hash = _sha256_text(
            _json_dumps(
                {
                    "run_id": row["run_id"],
                    "ts": float(row["ts"]),
                    "event_type": row["event_type"],
                    "object_type": row["object_type"],
                    "object_id": row["object_id"],
                    "payload_sha256": row["payload_sha256"],
                    "prev_event_hash": row["prev_event_hash"],
                }
            )
        )
        if expected_event_hash != row["event_hash"]:
            raise EnforcementError(f"Ledger event {row['event_id']} event hash mismatch")
        payload = _json_loads_object(payload_json, context=f"ledger_events.{row['event_id']}")
        if payload.get("format") == "incremental_v1" or "changes" in payload:
            saw_incremental = True
            latest_meta_sha = str(payload.get("run_meta_sha256") or latest_meta_sha)
            if bool(payload.get("full_snapshot")):
                expected = {table: {} for table in _TABLE_SPECS}
            for change in payload.get("changes") or []:
                if not isinstance(change, dict):
                    raise EnforcementError(f"Ledger event {row['event_id']} has invalid change")
                table = str(change.get("table") or "")
                object_id = str(change.get("id") or "")
                if table not in expected or not object_id:
                    raise EnforcementError(f"Ledger event {row['event_id']} has invalid table/id")
                op = str(change.get("op") or "upsert")
                if op == "delete":
                    expected[table].pop(object_id, None)
                    continue
                if op != "upsert":
                    raise EnforcementError(f"Ledger event {row['event_id']} has invalid op {op!r}")
                payload_sha256 = str(change.get("payload_sha256") or "")
                try:
                    position = int(change["position"])
                except Exception as exc:
                    raise EnforcementError(
                        f"Ledger event {row['event_id']} has invalid position"
                    ) from exc
                if not payload_sha256:
                    raise EnforcementError(
                        f"Ledger event {row['event_id']} missing row payload hash"
                    )
                expected[table][object_id] = {
                    "payload_sha256": payload_sha256,
                    "position": position,
                }
            counts = payload.get("counts")
            if isinstance(counts, dict):
                for table in _TABLE_SPECS:
                    if table in counts and int(counts.get(table) or 0) != len(expected[table]):
                        raise EnforcementError(
                            f"Ledger event {row['event_id']} {table} count mismatch"
                        )
        prev = str(row["event_hash"])
    return {
        "events": len(rows),
        "head_event_hash": None if not rows else prev,
        "run_meta_sha256": latest_meta_sha,
        "expected_tables": expected,
        "saw_incremental": saw_incremental,
    }


def verify_event_hash_chain(path: Path, run_id: str) -> Dict[str, Any]:
    """Verify the per-run event hash chain and replay incremental state hashes."""

    if not path.exists():
        raise FileNotFoundError(path)
    with _connect(path) as con:
        _ensure_schema(con)
        return _verify_event_hash_chain_con(con, run_id)


def _table_hashes(
    con: sqlite3.Connection, *, table: str, id_col: str, run_id: str
) -> Dict[str, Dict[str, Any]]:
    rows = con.execute(
        "SELECT "
        f"{id_col} AS object_id, position, payload_json, payload_sha256 FROM {table} "
        "WHERE run_id = ? ORDER BY position ASC",
        (run_id,),
    ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    label = _TABLE_SPECS[table][1]
    for row in rows:
        object_id = str(row["object_id"])
        payload_json = str(row["payload_json"] or "")
        actual_sha = _sha256_text(payload_json)
        stored_sha = str(row["payload_sha256"] or "")
        if not stored_sha:
            raise EnforcementError(
                f"SQLite {label} table is inconsistent with ledger event log: "
                "missing row payload hash"
            )
        if actual_sha != stored_sha:
            raise EnforcementError(
                f"SQLite {label} table is inconsistent with ledger event log: payload hash mismatch"
            )
        out[object_id] = {"payload_sha256": stored_sha, "position": int(row["position"])}
    return out


def _verify_tables_against_event_log(
    con: sqlite3.Connection, *, run_id: str, replay: Dict[str, Any]
) -> None:
    expected = replay.get("expected_tables") or {}
    for table, (id_col, label) in _TABLE_SPECS.items():
        actual = _table_hashes(con, table=table, id_col=id_col, run_id=run_id)
        exp = expected.get(table) or {}
        if actual != exp:
            raise EnforcementError(f"SQLite {label} table is inconsistent with ledger event log")


def _table_payloads(
    con: sqlite3.Connection, *, table: str, id_col: str, run_id: str
) -> List[Dict[str, Any]]:
    rows = con.execute(
        "SELECT "
        f"{id_col} AS object_id, payload_json FROM {table} "
        "WHERE run_id = ? ORDER BY position ASC",
        (run_id,),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            _json_loads_object(
                str(row["payload_json"] or "{}"), context=f"{table}.{row['object_id']}"
            )
        )
    return out


def _full_payload_from_sqlite(
    con: sqlite3.Connection, *, run_id: str, meta_payload: Dict[str, Any]
) -> Dict[str, Any]:
    spans_list = _table_payloads(con, table="spans", id_col="sid", run_id=run_id)
    attempts = _table_payloads(con, table="attempts", id_col="attempt_id", run_id=run_id)
    claims_list = _table_payloads(con, table="claims", id_col="cid", run_id=run_id)
    links = _table_payloads(con, table="claim_evidence_links", id_col="link_id", run_id=run_id)
    audits = _table_payloads(con, table="audits", id_col="audit_id", run_id=run_id)
    payload = dict(meta_payload)
    payload["payload_kind"] = "full_from_sqlite"
    payload["span_order"] = [str(item.get("sid")) for item in spans_list]
    payload["claim_order"] = [str(item.get("cid") or item.get("claim_id")) for item in claims_list]
    payload["spans"] = {str(item.get("sid")): item for item in spans_list}
    payload["attempts"] = attempts
    payload["claims"] = {str(item.get("cid") or item.get("claim_id")): item for item in claims_list}
    payload["claim_evidence_links"] = links
    payload["audits"] = audits
    return payload


# Legacy schema-3 verification.  Keep this code so already-created v3 ledgers
# migrate safely rather than being trusted by shape alone.


def _legacy_table_payloads(
    con: sqlite3.Connection, *, table: str, id_col: str, run_id: str
) -> Dict[str, Dict[str, Any]]:
    rows = con.execute(
        "SELECT "
        f"{id_col} AS object_id, payload_json FROM {table} "
        "WHERE run_id = ? ORDER BY position ASC",
        (run_id,),
    ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        out[str(row["object_id"])] = _json_loads_object(
            str(row["payload_json"] or "{}"), context=f"{table}.{row['object_id']}"
        )
    return out


def _verify_legacy_normalized_tables(
    con: sqlite3.Connection, *, run_id: str, payload: Dict[str, Any]
) -> None:
    expected_spans = dict(payload.get("spans") or {})
    actual_spans = _legacy_table_payloads(con, table="spans", id_col="sid", run_id=run_id)
    if {k: _json_dumps(v) for k, v in actual_spans.items()} != {
        k: _json_dumps(v) for k, v in expected_spans.items()
    }:
        raise EnforcementError("SQLite span table is inconsistent with run payload")

    expected_claims = dict(payload.get("claims") or {})
    actual_claims = _legacy_table_payloads(con, table="claims", id_col="cid", run_id=run_id)
    if {k: _json_dumps(v) for k, v in actual_claims.items()} != {
        k: _json_dumps(v) for k, v in expected_claims.items()
    }:
        raise EnforcementError("SQLite claim table is inconsistent with run payload")

    expected_attempts = {
        str(item.get("attempt_id") or f"A{i}"): item
        for i, item in enumerate(payload.get("attempts") or [])
        if isinstance(item, dict)
    }
    actual_attempts = _legacy_table_payloads(
        con, table="attempts", id_col="attempt_id", run_id=run_id
    )
    if {k: _json_dumps(v) for k, v in actual_attempts.items()} != {
        k: _json_dumps(v) for k, v in expected_attempts.items()
    }:
        raise EnforcementError("SQLite attempt table is inconsistent with run payload")

    expected_links = {
        str(item.get("link_id") or f"L{i}"): item
        for i, item in enumerate(payload.get("claim_evidence_links") or [])
        if isinstance(item, dict)
    }
    actual_links = _legacy_table_payloads(
        con, table="claim_evidence_links", id_col="link_id", run_id=run_id
    )
    if {k: _json_dumps(v) for k, v in actual_links.items()} != {
        k: _json_dumps(v) for k, v in expected_links.items()
    }:
        raise EnforcementError("SQLite claim/evidence link table is inconsistent with run payload")

    expected_audits = {
        str(item.get("audit_id") or f"V{i}"): item
        for i, item in enumerate(payload.get("audits") or [])
        if isinstance(item, dict)
    }
    actual_audits = _legacy_table_payloads(con, table="audits", id_col="audit_id", run_id=run_id)
    if {k: _json_dumps(v) for k, v in actual_audits.items()} != {
        k: _json_dumps(v) for k, v in expected_audits.items()
    }:
        raise EnforcementError("SQLite audit table is inconsistent with run payload")


def _verify_legacy_head_event_matches_payload(
    con: sqlite3.Connection, *, run_id: str, payload_json: str
) -> None:
    row = con.execute(
        """
        SELECT payload_json FROM ledger_events
        WHERE run_id = ? AND event_type = 'snapshot_committed'
        ORDER BY event_id DESC LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise EnforcementError(f"SQLite ledger for run {run_id!r} has no snapshot event")
    event_payload = _json_loads_object(
        str(row["payload_json"] or "{}"), context="ledger_events.snapshot_committed.payload_json"
    )
    expected_payload_sha = str(event_payload.get("payload_sha256") or "")
    actual_payload_sha = _sha256_text(payload_json)
    if expected_payload_sha != actual_payload_sha:
        raise EnforcementError("SQLite ledger head event does not match run payload")


def _load_from_sqlite(path: Path, run_id: str) -> RunState:
    needs_migration = False
    with _connect(path) as con:
        _ensure_schema(con)
        row = con.execute(
            "SELECT payload_json, payload_sha256, head_event_hash FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(path)
        payload_json = str(row["payload_json"] or "")
        if _sha256_text(payload_json) != str(row["payload_sha256"] or ""):
            raise EnforcementError(f"Run payload hash mismatch in {path}")
        payload = _json_loads_object(payload_json, context=f"{path}:runs.payload_json")

        if "spans" in payload or payload.get("payload_kind") == "full_export":
            # Schema-3 full-payload ledger. Verify using the old cross-table and
            # head-event checks, then migrate to incremental v4 after closing the DB.
            _verify_legacy_normalized_tables(con, run_id=run_id, payload=payload)
            _verify_legacy_head_event_matches_payload(con, run_id=run_id, payload_json=payload_json)
            _verify_event_hash_chain_con(con, run_id)
            run = run_from_payload(payload, fallback_run_id=run_id, mark_clean=False)
            needs_migration = True
        else:
            replay = _verify_event_hash_chain_con(con, run_id)
            if not replay.get("saw_incremental"):
                raise EnforcementError(
                    f"SQLite ledger for run {run_id!r} has no incremental state events"
                )
            if str(row["head_event_hash"] or "") != str(replay.get("head_event_hash") or ""):
                raise EnforcementError("SQLite ledger head event hash mismatch")
            if str(row["payload_sha256"] or "") != str(replay.get("run_meta_sha256") or ""):
                raise EnforcementError("SQLite ledger head event does not match run metadata")
            _verify_tables_against_event_log(con, run_id=run_id, replay=replay)
            full_payload = _full_payload_from_sqlite(con, run_id=run_id, meta_payload=payload)
            run = run_from_payload(full_payload, fallback_run_id=run_id, mark_clean=True)
            _install_committed_state_from_replay(run, replay)
    if needs_migration:
        persist_run(run)
    return run


def _load_from_json(path: Path, run_id: str) -> RunState:
    raw = _json_loads_object(path.read_text(encoding="utf-8"), context=str(path))
    if raw.get("payload_kind") == "ledger_head" and not raw.get("spans"):
        raise EnforcementError(
            f"{path} is a lightweight SQLite ledger head export, not a recoverable JSON run. "
            "Restore run.sqlite or enable BERRY_LEDGER_EXPORT_MODE=sync for full JSON exports."
        )
    return run_from_payload(raw, fallback_run_id=run_id, mark_clean=False)


def load_persisted_run(run_id: str) -> RunState:
    rid = str(run_id or "").strip()
    if not rid:
        raise EnforcementError("run_id is required")
    sqlite_path = run_sqlite_path(rid)
    if sqlite_path.exists():
        return _load_from_sqlite(sqlite_path, rid)
    json_path = run_json_path(rid)
    if json_path.exists():
        run = _load_from_json(json_path, rid)
        # JSON-only runs are legacy compatibility input.  Migrate them into the
        # SQLite ledger immediately.  Fail closed if migration cannot be committed.
        persist_run(run)
        return run
    raise FileNotFoundError(run_dir(rid))


# ---------------------------------------------------------------------------
# Inspection exports
# ---------------------------------------------------------------------------


def _write_hot_exports(run: RunState, event: Dict[str, Any]) -> None:
    """Write O(1) inspection artifacts for the latest commit.

    These files are not the source of truth; SQLite is.  They intentionally avoid
    full-run rewrites on each span append.  Use ``BERRY_LEDGER_EXPORT_MODE=sync``
    to regenerate the legacy full snapshots.
    """

    head = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "payload_kind": "ledger_head",
        "run_id": run.run_id,
        "source_of_truth": "run.sqlite",
        "ledger_path": str(run_sqlite_path(run.run_id)),
        "head_event_hash": event["event_hash"],
        "last_event_id": event["event_id"],
        "last_event_type": event["event_type"],
        "run_meta_sha256": event["run_meta_sha256"],
        "full_snapshot": event["full_snapshot"],
        "counts": _counts(run),
        "spans_version": int(run.spans_version),
        "updated_at": time.time(),
        "exports": {
            "mode": "hot",
            "description": "Lightweight head and append-only TSV/JSONL mirrors; run.sqlite is authoritative.",
            "sync_mode": f"Set {_EXPORT_MODE_ENV}=sync to regenerate full JSON/TSV snapshots on every commit.",
        },
    }
    text = json.dumps(head, indent=2, sort_keys=True) + "\n"
    atomic_write_text(run_json_path(run.run_id), text)
    atomic_write_text(ledger_head_path(run.run_id), text)
    _append_event_jsonl(run.run_id, event)
    _append_changed_tsv_rows(run, event.get("changes") or [], int(event["event_id"]))


def _append_event_jsonl(run_id: str, event: Dict[str, Any]) -> None:
    row = {
        "event_id": event["event_id"],
        "run_id": event["run_id"],
        "ts": event["ts"],
        "event_type": event["event_type"],
        "object_type": event["object_type"],
        "object_id": event["object_id"],
        "payload_sha256": event["payload_sha256"],
        "prev_event_hash": event["prev_event_hash"],
        "event_hash": event["event_hash"],
        "payload": event["payload"],
    }
    _append_text_durable(ledger_events_jsonl_path(run_id), json.dumps(row, sort_keys=True) + "\n")


def _ensure_tsv_header(path: Path, header: List[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        fh.flush()
        os.fsync(fh.fileno())


def _append_tsv_rows(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    if not rows:
        _ensure_tsv_header(path, header)
        return
    _ensure_tsv_header(path, header)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def _append_changed_tsv_rows(run: RunState, changes: List[Dict[str, Any]], event_id: int) -> None:
    by_table: Dict[str, List[str]] = {table: [] for table in _TABLE_SPECS}
    for change in changes:
        if change.get("op") != "upsert":
            continue
        table = str(change.get("table") or "")
        object_id = str(change.get("id") or "")
        if table in by_table and object_id:
            by_table[table].append(object_id)

    evidence_rows: List[List[Any]] = []
    for sid in by_table["spans"]:
        rec = run.spans.get(sid)
        if not rec:
            continue
        locator = ""
        if rec.locator:
            path_val = rec.locator.get("path") or rec.locator.get("rel_path") or ""
            start = rec.locator.get("start_line", "")
            end = rec.locator.get("end_line", "")
            locator = f"{path_val}:{start}-{end}" if path_val else json.dumps(rec.locator)
        evidence_rows.append(
            [
                event_id,
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
    _append_tsv_rows(
        evidence_tsv_path(run.run_id),
        [
            "event_id",
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
        ],
        evidence_rows,
    )

    attempt_by_id = {rec.attempt_id: rec for rec in run.attempts}
    _append_tsv_rows(
        attempts_tsv_path(run.run_id),
        [
            "event_id",
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
        ],
        [
            [
                event_id,
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
            for aid in by_table["attempts"]
            for rec in [attempt_by_id.get(aid)]
            if rec is not None
        ],
    )

    _append_tsv_rows(
        claims_tsv_path(run.run_id),
        [
            "event_id",
            "ts",
            "run_id",
            "cid",
            "kind",
            "status",
            "target",
            "latest_audit_id",
            "evidence_sids",
            "claim",
        ],
        [
            [
                event_id,
                f"{float(rec.created_at):.6f}",
                run.run_id,
                rec.cid,
                rec.kind,
                rec.status,
                f"{float(rec.target):.4f}",
                rec.latest_audit_id,
                ",".join(link.sid for link in run.claim_evidence_links if link.cid == rec.cid),
                rec.text,
            ]
            for cid in by_table["claims"]
            for rec in [run.claims.get(cid)]
            if rec is not None
        ],
    )

    link_by_id = {rec.link_id: rec for rec in run.claim_evidence_links}
    _append_tsv_rows(
        claim_evidence_tsv_path(run.run_id),
        [
            "event_id",
            "ts",
            "run_id",
            "link_id",
            "cid",
            "sid",
            "relation",
            "audit_id",
            "created_by",
            "note",
        ],
        [
            [
                event_id,
                f"{float(rec.created_at):.6f}",
                run.run_id,
                rec.link_id,
                rec.cid,
                rec.sid,
                rec.relation,
                rec.audit_id,
                rec.created_by,
                rec.note,
            ]
            for lid in by_table["claim_evidence_links"]
            for rec in [link_by_id.get(lid)]
            if rec is not None
        ],
    )

    audit_by_id = {rec.audit_id: rec for rec in run.audits}
    _append_tsv_rows(
        audits_tsv_path(run.run_id),
        [
            "event_id",
            "ts",
            "run_id",
            "audit_id",
            "kind",
            "claim_ids",
            "input_sids",
            "materialized_sids",
            "evidence_pack_id",
            "evidence_pack_hash",
            "audit_sid",
        ],
        [
            [
                event_id,
                f"{float(rec.created_at):.6f}",
                run.run_id,
                rec.audit_id,
                rec.kind,
                ",".join(rec.claim_ids),
                ",".join(rec.input_sids),
                ",".join(rec.materialized_sids),
                rec.evidence_pack_id,
                rec.evidence_pack_hash,
                rec.audit_sid,
            ]
            for aid in by_table["audits"]
            for rec in [audit_by_id.get(aid)]
            if rec is not None
        ],
    )


def _write_full_exports(run: RunState) -> None:
    payload = run_to_payload(run)
    run_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write_text(run_json_path(run.run_id), run_json)
    atomic_write_text(ledger_head_path(run.run_id), run_json)
    _write_evidence_tsv(run)
    _write_attempts_tsv(run)
    _write_claims_tsv(run)
    _write_claim_evidence_tsv(run)
    _write_audits_tsv(run)
    _write_ledger_events_jsonl_snapshot(run.run_id)


def export_run(run_id: str) -> Dict[str, Any]:
    """Regenerate full JSON/TSV inspection exports from the authoritative ledger."""

    run = load_persisted_run(run_id)
    _write_full_exports(run)
    return {
        "run_id": run.run_id,
        "mode": "sync",
        "run_json": str(run_json_path(run.run_id)),
        "evidence_tsv": str(evidence_tsv_path(run.run_id)),
        "attempts_tsv": str(attempts_tsv_path(run.run_id)),
        "claims_tsv": str(claims_tsv_path(run.run_id)),
        "claim_evidence_tsv": str(claim_evidence_tsv_path(run.run_id)),
        "audits_tsv": str(audits_tsv_path(run.run_id)),
        "ledger_events_jsonl": str(ledger_events_jsonl_path(run.run_id)),
    }


def _write_ledger_events_jsonl_snapshot(run_id: str) -> None:
    path = ledger_events_jsonl_path(run_id)
    rows: List[Dict[str, Any]] = []
    with _connect(run_sqlite_path(run_id)) as con:
        _ensure_schema(con)
        for row in con.execute(
            """
            SELECT event_id, run_id, ts, event_type, object_type, object_id,
                   payload_json, payload_sha256, prev_event_hash, event_hash
            FROM ledger_events WHERE run_id = ? ORDER BY event_id ASC
            """,
            (run_id,),
        ):
            rows.append(
                {
                    "event_id": int(row["event_id"]),
                    "run_id": row["run_id"],
                    "ts": float(row["ts"]),
                    "event_type": row["event_type"],
                    "object_type": row["object_type"],
                    "object_id": row["object_id"],
                    "payload_sha256": row["payload_sha256"],
                    "prev_event_hash": row["prev_event_hash"],
                    "event_hash": row["event_hash"],
                    "payload": _json_loads_object(
                        row["payload_json"], context=f"{path}:ledger_events.payload_json"
                    ),
                }
            )
    atomic_write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


# Full TSV snapshot writers, used only by explicit/sync exports.


def _write_evidence_tsv(run: RunState) -> None:
    path = evidence_tsv_path(run.run_id)
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
                rec = run.spans.get(sid)
                if not rec:
                    continue
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
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _write_attempts_tsv(run: RunState) -> None:
    path = attempts_tsv_path(run.run_id)
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
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _write_claims_tsv(run: RunState) -> None:
    path = claims_tsv_path(run.run_id)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(
                [
                    "ts",
                    "run_id",
                    "cid",
                    "kind",
                    "status",
                    "target",
                    "latest_audit_id",
                    "evidence_sids",
                    "claim",
                ]
            )
            for cid in run.claim_order:
                rec = run.claims.get(cid)
                if not rec:
                    continue
                evidence = [link.sid for link in run.claim_evidence_links if link.cid == rec.cid]
                writer.writerow(
                    [
                        f"{float(rec.created_at):.6f}",
                        run.run_id,
                        rec.cid,
                        rec.kind,
                        rec.status,
                        f"{float(rec.target):.4f}",
                        rec.latest_audit_id,
                        ",".join(evidence),
                        rec.text,
                    ]
                )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _write_claim_evidence_tsv(run: RunState) -> None:
    path = claim_evidence_tsv_path(run.run_id)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(
                [
                    "ts",
                    "run_id",
                    "link_id",
                    "cid",
                    "sid",
                    "relation",
                    "audit_id",
                    "created_by",
                    "note",
                ]
            )
            for rec in run.claim_evidence_links:
                writer.writerow(
                    [
                        f"{float(rec.created_at):.6f}",
                        run.run_id,
                        rec.link_id,
                        rec.cid,
                        rec.sid,
                        rec.relation,
                        rec.audit_id,
                        rec.created_by,
                        rec.note,
                    ]
                )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _write_audits_tsv(run: RunState) -> None:
    path = audits_tsv_path(run.run_id)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(
                [
                    "ts",
                    "run_id",
                    "audit_id",
                    "kind",
                    "claim_ids",
                    "input_sids",
                    "materialized_sids",
                    "evidence_pack_id",
                    "evidence_pack_hash",
                    "audit_sid",
                ]
            )
            for rec in run.audits:
                writer.writerow(
                    [
                        f"{float(rec.created_at):.6f}",
                        run.run_id,
                        rec.audit_id,
                        rec.kind,
                        ",".join(rec.claim_ids),
                        ",".join(rec.input_sids),
                        ",".join(rec.materialized_sids),
                        rec.evidence_pack_id,
                        rec.evidence_pack_hash,
                        rec.audit_sid,
                    ]
                )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
