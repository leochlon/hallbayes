"""RAGTruth grounding benchmark for Berry v2 (realistic RAG faithfulness).

Real LLM responses + the context given to the model + human span-level grounding
labels. Task = does the response contain content not supported by the context
(information sufficiency / grounding), which is what Berry measures.

answer = output (the RAG response), evidence span = context, context_mode="all"
(no per-sentence citations needed). Response label from hallucination_labels_processed:
hallucinated(1) iff evident_conflict>0 or baseless_info>0, else grounded(0).

Env: OPENAI_BASE_URL, OPENAI_API_KEY. Usage:
  python bench/run_ragtruth.py --data /tmp/ragtruth.json --n 400 --out /tmp/rt_out.jsonl
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from berry.hallucination_detector.core import run_detect_hallucination

MODEL = "gpt-4.1-mini"


def label_of(r: dict) -> int:
    p = r["hallucination_labels_processed"]
    if isinstance(p, str):
        p = json.loads(p)
    return 1 if (p.get("evident_conflict", 0) or p.get("baseless_info", 0)) else 0


def berry(output: str, context: str) -> dict:
    r = run_detect_hallucination(
        answer=output,
        spans=[{"sid": "S1", "text": context}],
        verifier_model=MODEL,
        context_mode="all",
        max_concurrency=3,
        timeout_s=120,
    )
    s = r.get("summary", {})
    gaps = [(d.get("budget_gap") or {}).get("min") for d in (r.get("details") or [])]
    gaps = [g for g in gaps if g is not None]
    return {
        "score": max(gaps) if gaps else 0.0,  # most-ungrounded claim
        "flagged": (s.get("flagged_claims") or 0) > 0,
        "claims": s.get("claims_total"),
    }


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    w = t = 0
    for p in pos:
        for n in neg:
            w += p > n
            t += p == n
    return (w + 0.5 * t) / (len(pos) * len(neg))


def aurc(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    bad, risks = 0, []
    for k, i in enumerate(order, 1):
        bad += labels[i]
        risks.append(bad / k)
    return sum(risks) / len(order)


def prf(pred, labels):
    tp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 0)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {
        "acc": round((tp + tn) / max(1, len(labels)), 3),
        "prec": round(prec, 3),
        "rec": round(rec, 3),
        "f1": round(2 * prec * rec / max(1e-9, prec + rec), 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    rows = json.load(open(args.data))[: args.n]
    out = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(berry, r["output"], r["context"]): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = {
                "id": rows[i].get("id"),
                "label": label_of(rows[i]),
                "task": rows[i].get("task_type"),
                **fut.result(),
            }
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(rows)}", flush=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    labels = [r["label"] for r in out]
    scores = [r["score"] for r in out]
    flagged = [1 if r["flagged"] else 0 for r in out]
    print("\n==== BERRY v2 GROUNDING (RAGTruth, real RAG responses) ====")
    print(
        f"responses={len(out)}  hallucinated={sum(labels)} grounded={len(labels) - sum(labels)} (base rate {sum(labels) / len(labels):.2f})"
    )
    print(f"AUROC = {auroc(scores, labels):.3f}   AURC = {aurc(scores, labels):.3f}")
    print(f"decision @target 0.95 : {prf(flagged, labels)}")


if __name__ == "__main__":
    main()
