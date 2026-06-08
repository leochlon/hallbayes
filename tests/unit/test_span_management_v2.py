from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from berry.enforcement import EnforcementError, RunStore, SpanRecord
from berry.mcp_server import _load_persisted_run, _persist_run, _run_json_path, _run_sqlite_path
from berry.run_ledger import load_persisted_run, verify_event_hash_chain


def test_span_record_v2_classifies_anchor_and_redacts_secrets() -> None:
    store = RunStore()
    run = store.start_run(run_id="r")

    anchor = store.add_span(
        run=run,
        text="Build a verifier.",
        source="anchor",
        meta={"kind": "problem"},
    )
    secret = store.add_span(
        run=run,
        text="OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567890",
        source="manual",
    )

    assert anchor.kind == "anchor"
    assert anchor.is_citable is False
    assert secret.sensitivity == "secret"
    assert secret.is_sensitive is True

    listed = store.list_spans(run=run)
    by_sid = {s["sid"]: s for s in listed}
    assert by_sid[anchor.sid]["kind"] == "anchor"
    assert by_sid[secret.sid]["preview"] == "[REDACTED sensitive span]"
    assert by_sid[secret.sid]["text_sha256"] == secret.text_sha256


def test_resolve_evidence_pack_excludes_non_citable_unsafe_and_unknown_spans() -> None:
    store = RunStore()
    run = store.start_run(run_id="r")
    anchor = store.add_span(run=run, text="Desired outcome", source="anchor")
    evidence = store.add_span(run=run, text="The test passed with exit code 0.", source="test")
    secret = store.add_span(run=run, text="password=supersecret123", source="log")
    stale = store.add_span(run=run, text="Old result", source="test")
    store.mark_span(run=run, sid=stale.sid, status="stale")
    tombstoned = store.add_span(run=run, text="Deleted result", source="test")
    store.mark_span(run=run, sid=tombstoned.sid, status="tombstoned")

    pack = store.resolve_evidence_pack(
        run=run,
        sids=[anchor.sid, evidence.sid, secret.sid, stale.sid, tombstoned.sid, "S999"],
    )

    assert pack["materialized_sids"] == [evidence.sid]
    reasons = {(e["sid"], e["reason"]) for e in pack["excluded"]}
    assert (anchor.sid, "kind:anchor") in reasons
    assert (secret.sid, "sensitivity:secret") in reasons
    assert (stale.sid, "status:stale") in reasons
    assert (tombstoned.sid, "status:tombstoned") in reasons
    assert ("S999", "unknown") in reasons

    stale_allowed = store.resolve_evidence_pack(run=run, sids=[stale.sid], include_stale=True)
    assert stale_allowed["materialized_sids"] == [stale.sid]
    tombstone_allowed = store.resolve_evidence_pack(
        run=run, sids=[tombstoned.sid], include_stale=True
    )
    assert tombstone_allowed["materialized_sids"] == []
    assert tombstone_allowed["excluded"] == [{"sid": tombstoned.sid, "reason": "status:tombstoned"}]
    assert pack["pack_id"]
    assert pack["text_sha256"]


def test_extract_span_no_match_does_not_create_citable_placeholder() -> None:
    store = RunStore()
    run = store.start_run(run_id="r")
    parent = store.add_span(
        run=run,
        text="alpha\nbeta\ngamma",
        source="file",
        source_type="file",
        locator={"path": "x.py", "start_line": 1, "end_line": 3},
    )
    before = list(run.span_order)

    no_match = store.extract_span(
        run=run,
        parent_sid=parent.sid,
        selector={"type": "regex", "pattern": "delta"},
    )

    assert no_match["matched"] is False
    assert no_match["sid"] is None
    assert run.span_order == before

    match = store.extract_span(
        run=run,
        parent_sid=parent.sid,
        selector={"type": "regex", "pattern": "beta"},
        reason="narrow to relevant assertion",
    )

    assert match["matched"] is True
    child = store.get_span(run=run, sid=match["sid"])
    assert child.text == "beta"
    assert child.kind == "derived"
    assert child.parents == [parent.sid]
    assert child.transform and child.transform["type"] == "regex_extract"
    assert child.is_citable is True


def test_resolve_evidence_pack_surfaces_parents_for_non_extract_derived_summary() -> None:
    store = RunStore()
    run = store.start_run(run_id="r")
    parent = store.add_span(run=run, text="Primary source says the value is 42.", source="source")
    summary = store.add_span(
        run=run,
        text="The value is 42.",
        source="summary",
        source_type="summary",
        kind="derived",
        parents=[parent.sid],
        transform={"type": "summary"},
        trust="derived",
    )

    pack = store.resolve_evidence_pack(run=run, sids=[summary.sid])

    assert pack["materialized_sids"] == [parent.sid]
    assert any(
        e["sid"] == summary.sid and e["reason"] == "derived_not_extractively_citable"
        for e in pack["excluded"]
    )


def test_query_evidence_filters_kind_source_status_and_excludes_derived_by_default() -> None:
    store = RunStore()
    run = store.start_run(run_id="r")
    file_span = store.add_span(
        run=run,
        text="Graph extraction completed successfully.",
        source="file",
        source_type="file",
        kind="evidence",
    )
    store.add_span(
        run=run,
        text="Graph extraction summary.",
        source="summary",
        source_type="summary",
        kind="derived",
        parents=[file_span.sid],
        transform={"type": "summary"},
    )

    default_results = store.query_evidence(run=run, query="graph extraction")
    assert [r["sid"] for r in default_results] == [file_span.sid]

    file_results = store.query_evidence(
        run=run,
        query="graph extraction",
        source_types=["file"],
    )
    assert [r["sid"] for r in file_results] == [file_span.sid]

    with_derived = store.query_evidence(
        run=run,
        query="graph extraction",
        include_derived=True,
    )
    assert len(with_derived) == 2


def test_persisted_run_round_trips_v2_span_fields(tmp_berry_home: Path) -> None:
    store = RunStore()
    run = store.start_run(run_id="roundtrip")
    run.baseline_kind = "git"
    run.baseline_ref = "abc123"
    parent = store.add_span(
        run=run,
        text="line 1\nline 2",
        source="file",
        source_type="file",
        kind="evidence",
        locator={"path": "src/x.py", "start_line": 1, "end_line": 2},
        snapshot={"file_sha256": "filehash"},
        tags=["file"],
    )
    child = store.extract_span(
        run=run,
        parent_sid=parent.sid,
        selector={"type": "line_range", "start_line": 2, "end_line": 2},
    )["sid"]

    _persist_run(run)
    loaded = _load_persisted_run("roundtrip")

    assert loaded.baseline_kind == "git"
    assert loaded.baseline_ref == "abc123"
    assert loaded.spans[parent.sid].locator["path"] == "src/x.py"
    assert loaded.spans[parent.sid].snapshot["file_sha256"] == "filehash"
    assert loaded.spans[child].parents == [parent.sid]
    assert loaded.spans[child].kind == "derived"

    payload = json.loads((tmp_berry_home / "runs" / "roundtrip" / "run.json").read_text())
    assert payload["schema_version"] == 3
    assert payload["spans"][parent.sid]["text_sha256"] == parent.text_sha256
    assert _run_sqlite_path("roundtrip").exists()
    assert verify_event_hash_chain(_run_sqlite_path("roundtrip"), "roundtrip")["events"] == 1
    assert (tmp_berry_home / "runs" / "roundtrip" / "ledger_events.jsonl").exists()


def test_persist_run_fails_closed_on_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    store = RunStore()
    run = store.start_run(run_id="fail")
    store.add_span(run=run, text="evidence", source="manual")

    def boom(path, text):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr("berry.run_ledger.atomic_write_text", boom)
    with pytest.raises(EnforcementError, match="Failed to persist run ledger"):
        _persist_run(run)


def test_v1_span_record_construction_still_gets_v2_defaults() -> None:
    rec = SpanRecord(
        sid="S0",
        text="hello",
        source="manual",
        created_at=1.0,
        meta={},
    )

    assert rec.kind == "evidence"
    assert rec.source_type == "manual"
    assert rec.status == "active"
    assert rec.text_sha256
    assert rec.eid


def test_sqlite_ledger_is_source_of_truth_when_json_export_is_missing(tmp_berry_home: Path) -> None:
    store = RunStore()
    run = store.start_run(run_id="sqlite-source")
    span = store.add_span(run=run, text="Measured accuracy is 0.91.", source="metric")
    claim = store.create_claim(run=run, text="Accuracy is 0.91.", target=0.95)
    store.link_claim_evidence(run=run, cid=claim.cid, sid=span.sid, relation="supports")
    store.record_audit(
        run=run,
        kind="manual",
        claim_ids=[claim.cid],
        input_sids=[span.sid],
        materialized_sids=[span.sid],
        evidence_pack_id="pack",
        evidence_pack_hash="hash",
        result={"status": "passed"},
    )

    _persist_run(run)
    sqlite_path = _run_sqlite_path("sqlite-source")
    assert sqlite_path.exists()
    with sqlite3.connect(sqlite_path) as con:
        assert con.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM claim_evidence_links").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM audits").fetchone()[0] == 1

    _run_json_path("sqlite-source").unlink()
    loaded = load_persisted_run("sqlite-source")
    assert loaded.spans[span.sid].text == span.text
    assert loaded.claims[claim.cid].text == claim.text
    assert loaded.claim_evidence_links[0].sid == span.sid
    assert loaded.audits[0].evidence_pack_id == "pack"


def test_sqlite_event_log_tampering_fails_closed(tmp_berry_home: Path) -> None:
    store = RunStore()
    run = store.start_run(run_id="tamper")
    store.add_span(run=run, text="durable fact", source="manual")
    _persist_run(run)

    sqlite_path = _run_sqlite_path("tamper")
    with sqlite3.connect(sqlite_path) as con:
        con.execute(
            "UPDATE ledger_events SET payload_json = ? WHERE event_id = 1",
            ('{"tampered":true}',),
        )
        con.commit()

    with pytest.raises(EnforcementError, match="payload hash mismatch|head event"):
        _load_persisted_run("tamper")


def test_legacy_json_without_sqlite_still_loads_and_migrates(tmp_berry_home: Path) -> None:
    run_dir = tmp_berry_home / "runs" / "legacy"
    run_dir.mkdir(parents=True)
    payload = {
        "schema_version": 2,
        "run_id": "legacy",
        "created_at": 1.0,
        "next_span_idx": 1,
        "span_order": ["S0"],
        "spans": {
            "S0": {
                "sid": "S0",
                "text": "Legacy evidence.",
                "source": "manual",
                "created_at": 1.0,
            }
        },
        "attempts": [],
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_persisted_run("legacy")
    assert loaded.spans["S0"].kind == "evidence"
    assert loaded.claims == {}
    assert loaded.next_claim_idx == 0
    assert _run_sqlite_path("legacy").exists()


def test_sqlite_normalized_table_tampering_fails_closed(tmp_berry_home: Path) -> None:
    store = RunStore()
    run = store.start_run(run_id="table-tamper")
    span = store.add_span(run=run, text="original evidence", source="manual")
    _persist_run(run)

    sqlite_path = _run_sqlite_path("table-tamper")
    with sqlite3.connect(sqlite_path) as con:
        con.execute(
            "UPDATE spans SET payload_json = ? WHERE run_id = ? AND sid = ?",
            (json.dumps({"sid": span.sid, "text": "tampered"}), "table-tamper", span.sid),
        )
        con.commit()

    with pytest.raises(EnforcementError, match="span table is inconsistent"):
        _load_persisted_run("table-tamper")


def test_sqlite_head_event_must_commit_current_run_payload(tmp_berry_home: Path) -> None:
    store = RunStore()
    run = store.start_run(run_id="head-tamper")
    store.add_span(run=run, text="unchanged evidence", source="manual")
    _persist_run(run)

    sqlite_path = _run_sqlite_path("head-tamper")
    with sqlite3.connect(sqlite_path) as con:
        payload_json = con.execute(
            "SELECT payload_json FROM runs WHERE run_id = ?", ("head-tamper",)
        ).fetchone()[0]
        payload = json.loads(payload_json)
        payload["baseline_ref"] = "tampered-but-child-tables-still-match"
        mutated = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        con.execute(
            "UPDATE runs SET payload_json = ?, payload_sha256 = ? WHERE run_id = ?",
            (mutated, hashlib.sha256(mutated.encode("utf-8")).hexdigest(), "head-tamper"),
        )
        con.commit()

    with pytest.raises(EnforcementError, match="head event does not match"):
        _load_persisted_run("head-tamper")
