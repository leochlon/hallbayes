from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .enforcement import (
    AttemptRecord,
    AuditRecord,
    ClaimEvidenceLink,
    ClaimRecord,
    EnforcementError,
    RunState,
    SpanRecord,
)
from .paths import ensure_berry_home

LEDGER_SCHEMA_VERSION = 3
_SQLITE_USER_VERSION = 3


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


def run_to_payload(run: RunState) -> Dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
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


def run_from_payload(raw: Dict[str, Any], *, fallback_run_id: Optional[str] = None) -> RunState:
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
        run.next_claim_link_idx,
        [link.link_id for link in run.claim_evidence_links],
        "L",
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
    return run


# ---------------------------------------------------------------------------
# SQLite source-of-truth with tamper-evident snapshot events
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


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS spans (
            run_id TEXT NOT NULL,
            sid TEXT NOT NULL,
            position INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
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
            PRIMARY KEY (run_id, attempt_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS claims (
            run_id TEXT NOT NULL,
            cid TEXT NOT NULL,
            position INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
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
        PRAGMA user_version = 3;
        """
    )


def _append_event(
    con: sqlite3.Connection,
    *,
    run_id: str,
    event_type: str,
    object_type: str,
    object_id: str,
    payload: Dict[str, Any],
) -> None:
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
    con.execute(
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


def _content_manifest(run: RunState) -> Dict[str, Any]:
    payload = run_to_payload(run)
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "run_id": run.run_id,
        "payload_sha256": _sha256_text(_json_dumps(payload)),
        "spans": len(run.spans),
        "attempts": len(run.attempts),
        "claims": len(run.claims),
        "claim_evidence_links": len(run.claim_evidence_links),
        "audits": len(run.audits),
        "spans_version": run.spans_version,
        "span_text_hashes": {
            sid: run.spans[sid].text_sha256 for sid in run.span_order if sid in run.spans
        },
        "claim_statuses": {
            cid: run.claims[cid].status for cid in run.claim_order if cid in run.claims
        },
        "audit_ids": [audit.audit_id for audit in run.audits],
    }


def _replace_table_rows(con: sqlite3.Connection, run: RunState, payload: Dict[str, Any]) -> None:
    run_json = _json_dumps(payload)
    now = time.time()
    con.execute(
        """
        INSERT INTO runs(run_id, payload_json, payload_sha256, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            payload_json=excluded.payload_json,
            payload_sha256=excluded.payload_sha256,
            created_at=excluded.created_at,
            updated_at=excluded.updated_at
        """,
        (run.run_id, run_json, _sha256_text(run_json), float(run.created_at), now),
    )

    for table in ["claim_evidence_links", "audits", "attempts", "claims", "spans"]:
        con.execute(f"DELETE FROM {table} WHERE run_id = ?", (run.run_id,))

    for pos, sid in enumerate(run.span_order):
        span = run.spans.get(sid)
        if not span:
            continue
        sp = span_to_payload(span)
        con.execute(
            """
            INSERT INTO spans(
                run_id, sid, position, payload_json, text_sha256, eid, kind, status, source_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                span.sid,
                pos,
                _json_dumps(sp),
                span.text_sha256,
                span.eid,
                span.kind,
                span.status,
                span.source_type,
            ),
        )

    for pos, attempt in enumerate(run.attempts):
        con.execute(
            "INSERT INTO attempts(run_id, attempt_id, position, payload_json) VALUES (?, ?, ?, ?)",
            (run.run_id, attempt.attempt_id, pos, _json_dumps(attempt_to_payload(attempt))),
        )

    for pos, cid in enumerate(run.claim_order):
        claim = run.claims.get(cid)
        if not claim:
            continue
        cp = claim_to_payload(claim)
        con.execute(
            """
            INSERT INTO claims(
                run_id, cid, position, payload_json, kind, status, target, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                claim.cid,
                pos,
                _json_dumps(cp),
                claim.kind,
                claim.status,
                claim.target,
                claim.updated_at,
            ),
        )

    for pos, link in enumerate(run.claim_evidence_links):
        if link.cid not in run.claims or link.sid not in run.spans:
            raise EnforcementError(
                "Cannot persist dangling claim/evidence link "
                f"{link.link_id}: {link.cid}->{link.sid}"
            )
        lp = claim_evidence_to_payload(link)
        con.execute(
            """
            INSERT INTO claim_evidence_links(
                run_id, link_id, position, payload_json, cid, sid, relation, audit_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                link.link_id,
                pos,
                _json_dumps(lp),
                link.cid,
                link.sid,
                link.relation,
                link.audit_id,
            ),
        )

    for pos, audit in enumerate(run.audits):
        dangling_claims = [cid for cid in audit.claim_ids if cid not in run.claims]
        dangling_sids = [
            sid for sid in audit.input_sids + audit.materialized_sids if sid not in run.spans
        ]
        if dangling_claims or dangling_sids:
            raise EnforcementError(
                f"Cannot persist dangling audit {audit.audit_id}: "
                f"claims={dangling_claims}, sids={dangling_sids}"
            )
        ap = audit_to_payload(audit)
        con.execute(
            """
            INSERT INTO audits(
                run_id, audit_id, position, payload_json, kind, created_at,
                evidence_pack_id, evidence_pack_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                audit.audit_id,
                pos,
                _json_dumps(ap),
                audit.kind,
                audit.created_at,
                audit.evidence_pack_id,
                audit.evidence_pack_hash,
            ),
        )


def _table_payloads(
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


def _verify_normalized_tables(
    con: sqlite3.Connection, *, run_id: str, payload: Dict[str, Any]
) -> None:
    """Ensure normalized SQLite tables have not drifted from the run payload."""

    expected_spans = dict(payload.get("spans") or {})
    actual_spans = _table_payloads(con, table="spans", id_col="sid", run_id=run_id)
    if {k: _json_dumps(v) for k, v in actual_spans.items()} != {
        k: _json_dumps(v) for k, v in expected_spans.items()
    }:
        raise EnforcementError("SQLite span table is inconsistent with run payload")

    expected_claims = dict(payload.get("claims") or {})
    actual_claims = _table_payloads(con, table="claims", id_col="cid", run_id=run_id)
    if {k: _json_dumps(v) for k, v in actual_claims.items()} != {
        k: _json_dumps(v) for k, v in expected_claims.items()
    }:
        raise EnforcementError("SQLite claim table is inconsistent with run payload")

    expected_attempts = {
        str(item.get("attempt_id") or f"A{i}"): item
        for i, item in enumerate(payload.get("attempts") or [])
        if isinstance(item, dict)
    }
    actual_attempts = _table_payloads(con, table="attempts", id_col="attempt_id", run_id=run_id)
    if {k: _json_dumps(v) for k, v in actual_attempts.items()} != {
        k: _json_dumps(v) for k, v in expected_attempts.items()
    }:
        raise EnforcementError("SQLite attempt table is inconsistent with run payload")

    expected_links = {
        str(item.get("link_id") or f"L{i}"): item
        for i, item in enumerate(payload.get("claim_evidence_links") or [])
        if isinstance(item, dict)
    }
    actual_links = _table_payloads(
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
    actual_audits = _table_payloads(con, table="audits", id_col="audit_id", run_id=run_id)
    if {k: _json_dumps(v) for k, v in actual_audits.items()} != {
        k: _json_dumps(v) for k, v in expected_audits.items()
    }:
        raise EnforcementError("SQLite audit table is inconsistent with run payload")


def _verify_head_event_matches_payload(
    con: sqlite3.Connection, *, run_id: str, payload_json: str
) -> None:
    """Ensure the latest snapshot event commits the current run payload hash."""

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
        str(row["payload_json"] or "{}"),
        context="ledger_events.snapshot_committed.payload_json",
    )
    expected_payload_sha = str(event_payload.get("payload_sha256") or "")
    actual_payload_sha = _sha256_text(payload_json)
    if expected_payload_sha != actual_payload_sha:
        raise EnforcementError("SQLite ledger head event does not match run payload")


def persist_run(run: RunState) -> None:
    """Persist a run into SQLite source-of-truth plus inspection exports.

    The SQLite write is a single IMMEDIATE transaction.  A tamper-evident
    snapshot event is appended on every successful commit.  JSON/TSV files are
    compatibility exports; they are not the source of truth once ``run.sqlite``
    exists, but export failures still fail closed so users do not rely on stale
    inspection ledgers.
    """

    try:
        payload = run_to_payload(run)
        db_path = run_sqlite_path(run.run_id)
        with _connect(db_path) as con:
            _ensure_schema(con)
            con.execute("BEGIN IMMEDIATE")
            _replace_table_rows(con, run, payload)
            _append_event(
                con,
                run_id=run.run_id,
                event_type="snapshot_committed",
                object_type="run",
                object_id=run.run_id,
                payload=_content_manifest(run),
            )
            con.commit()
        _write_exports(run, payload)
    except EnforcementError:
        raise
    except Exception as exc:
        raise EnforcementError(
            f"Failed to persist run ledger: {type(exc).__name__}: {exc}"
        ) from exc


def verify_event_hash_chain(path: Path, run_id: str) -> Dict[str, Any]:
    """Verify the per-run event hash chain and payload hashes."""

    if not path.exists():
        raise FileNotFoundError(path)
    with _connect(path) as con:
        _ensure_schema(con)
        rows = con.execute(
            """
            SELECT event_id, run_id, ts, event_type, object_type, object_id,
                   payload_json, payload_sha256, prev_event_hash, event_hash
            FROM ledger_events WHERE run_id = ? ORDER BY event_id ASC
            """,
            (run_id,),
        ).fetchall()
    prev = "GENESIS"
    for row in rows:
        payload_sha = _sha256_text(str(row["payload_json"] or ""))
        if payload_sha != row["payload_sha256"]:
            raise EnforcementError(f"Ledger event {row['event_id']} payload hash mismatch")
        if str(row["prev_event_hash"] or "") != prev:
            raise EnforcementError(f"Ledger event {row['event_id']} previous hash mismatch")
        expected = _sha256_text(
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
        if expected != row["event_hash"]:
            raise EnforcementError(f"Ledger event {row['event_id']} event hash mismatch")
        prev = str(row["event_hash"])
    return {"events": len(rows), "head_event_hash": None if not rows else prev}


def _load_from_sqlite(path: Path, run_id: str) -> RunState:
    with _connect(path) as con:
        _ensure_schema(con)
        row = con.execute(
            "SELECT payload_json, payload_sha256 FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(path)
        payload_json = str(row["payload_json"] or "")
        if _sha256_text(payload_json) != str(row["payload_sha256"] or ""):
            raise EnforcementError(f"Run payload hash mismatch in {path}")
        payload = _json_loads_object(payload_json, context=f"{path}:runs.payload_json")
        _verify_normalized_tables(con, run_id=run_id, payload=payload)
        _verify_head_event_matches_payload(con, run_id=run_id, payload_json=payload_json)
    verify_event_hash_chain(path, run_id)
    return run_from_payload(payload, fallback_run_id=run_id)


def _load_from_json(path: Path, run_id: str) -> RunState:
    raw = _json_loads_object(path.read_text(encoding="utf-8"), context=str(path))
    return run_from_payload(raw, fallback_run_id=run_id)


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
        # JSON-only runs are legacy compatibility input.  Migrate them into
        # the SQLite ledger immediately so subsequent loads get hash-chain
        # verification and normalized graph tables.  Fail closed if the
        # migration cannot be committed.
        persist_run(run)
        return run
    raise FileNotFoundError(run_dir(rid))


# ---------------------------------------------------------------------------
# Inspection exports
# ---------------------------------------------------------------------------


def _write_exports(run: RunState, payload: Dict[str, Any]) -> None:
    run_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write_text(run_json_path(run.run_id), run_json)
    _write_evidence_tsv(run)
    _write_attempts_tsv(run)
    _write_claims_tsv(run)
    _write_claim_evidence_tsv(run)
    _write_audits_tsv(run)
    _write_ledger_events_jsonl(run.run_id)


def _write_ledger_events_jsonl(run_id: str) -> None:
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
                        row["payload_json"],
                        context=f"{path}:ledger_events.payload_json",
                    ),
                }
            )
    atomic_write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


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
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
