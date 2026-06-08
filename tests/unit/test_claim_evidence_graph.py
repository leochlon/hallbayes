from __future__ import annotations

from pathlib import Path

import pytest

from berry.enforcement import EnforcementError, RunStore
from berry.mcp_server import _load_persisted_run, _persist_run


def test_claim_graph_create_link_list_update_and_persist(tmp_berry_home: Path) -> None:
    store = RunStore()
    run = store.start_run(run_id="claims")
    span = store.add_span(run=run, text="The implementation added a SQLite ledger.", source="diff")
    claim = store.create_claim(
        run=run,
        text="The implementation added a SQLite ledger.",
        kind="fact",
        target=0.95,
        tags=["storage"],
    )
    link = store.link_claim_evidence(run=run, cid=claim.cid, sid=span.sid, relation="supports")

    listed = store.list_claims(run=run)
    assert listed[0]["cid"] == claim.cid
    assert listed[0]["evidence"][0]["link_id"] == link.link_id

    audit = store.record_audit(
        run=run,
        kind="manual",
        claim_ids=[claim.cid],
        input_sids=[span.sid],
        materialized_sids=[span.sid],
        evidence_pack_id="pack1",
        evidence_pack_hash="hash1",
        verifier_model="scripted",
        result={"status": "passed"},
    )
    updated = store.update_claim(
        run=run,
        cid=claim.cid,
        status="supported",
        target=0.9,
        tags_add=["audited"],
        latest_audit_id=audit.audit_id,
    )
    assert updated.status == "supported"
    assert updated.target == 0.9
    assert "audited" in updated.tags
    assert updated.latest_audit_id == audit.audit_id

    _persist_run(run)
    loaded = _load_persisted_run("claims")
    assert loaded.claims[claim.cid].status == "supported"
    assert loaded.claim_evidence_links[0].relation == "supports"
    assert loaded.audits[0].claim_ids == [claim.cid]


def test_claim_graph_rejects_unknown_or_non_citable_support_edges() -> None:
    store = RunStore()
    run = store.start_run(run_id="claims")
    claim = store.create_claim(run=run, text="The task was completed.")
    anchor = store.add_span(run=run, text="Please complete the task.", source="anchor")

    with pytest.raises(EnforcementError, match="Unknown span id"):
        store.link_claim_evidence(run=run, cid=claim.cid, sid="S999", relation="supports")

    with pytest.raises(EnforcementError, match="not citable"):
        store.link_claim_evidence(run=run, cid=claim.cid, sid=anchor.sid, relation="supports")

    background = store.link_claim_evidence(
        run=run, cid=claim.cid, sid=anchor.sid, relation="background"
    )
    assert background.relation == "background"


def test_claim_steps_use_open_claims_and_linked_supporting_evidence() -> None:
    store = RunStore()
    run = store.start_run(run_id="claims")
    span = store.add_span(run=run, text="Benchmark latency is 42 ms.", source="benchmark")
    unsupported = store.create_claim(run=run, text="This claim has no evidence.")
    supported = store.create_claim(run=run, text="Benchmark latency is 42 ms.")
    store.link_claim_evidence(run=run, cid=supported.cid, sid=span.sid, relation="supports")

    steps = store.claim_steps(run=run, claim_ids=[unsupported.cid, supported.cid])

    assert [step["claim_id"] for step in steps] == [unsupported.cid, supported.cid]
    assert steps[0]["cites"] == []
    assert steps[1]["cites"] == [span.sid]


def test_claim_steps_do_not_pack_non_citable_background_anchors() -> None:
    store = RunStore()
    run = store.start_run(run_id="claims")
    claim = store.create_claim(run=run, text="The migration was implemented.")
    anchor = store.add_span(run=run, text="Please implement the migration.", source="anchor")
    evidence = store.add_span(
        run=run,
        text="run.sqlite is written during persist_run.",
        source="diff",
    )

    store.link_claim_evidence(run=run, cid=claim.cid, sid=anchor.sid, relation="background")
    store.link_claim_evidence(run=run, cid=claim.cid, sid=evidence.sid, relation="supports")

    steps = store.claim_steps(
        run=run,
        claim_ids=[claim.cid],
        evidence_relations=["supports", "background"],
    )

    assert steps == [
        {
            "idx": 0,
            "claim_id": claim.cid,
            "claim": claim.text,
            "cites": [evidence.sid],
            "confidence": claim.target,
        }
    ]
