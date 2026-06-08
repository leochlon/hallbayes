"""Information-SUFFICIENCY benchmark for Berry v2 on HotpotQA (distractor).

This tests what Berry measures: does the CITED EVIDENCE contain enough
information to support the claim -- not whether the claim is true in the world.

Controlled ablation per HotpotQA item (claim held fixed, only evidence changes):
  SUFFICIENT  : cite the gold supporting-fact sentences  -> Berry should PASS
  INSUFFICIENT: cite distractor paragraphs only          -> Berry should FLAG
Label is ground-truth by construction (sufficiency), so AUROC/AURC measure
sufficiency detection, isolated from correctness.

Env: OPENAI_BASE_URL, OPENAI_API_KEY. Usage:
  python bench/run_suff.py --data /tmp/hotpot_rows.json --out /tmp/suff_out.jsonl
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from berry.hallucination_detector.core import run_detect_hallucination

MODEL = "gpt-4.1-mini"


def build_tasks(rows: list[dict]) -> list[dict]:
    tasks = []
    for r in rows:
        row = r["row"] if "row" in r else r
        q, ans = row["question"], row["answer"]
        ctx, sf = row["context"], row["supporting_facts"]
        titles, sents = ctx["title"], ctx["sentences"]
        gold_titles = set(sf["title"])
        # gold supporting-fact sentences
        gold = []
        for gt, si in zip(sf["title"], sf["sent_id"]):
            if gt in titles:
                j = titles.index(gt)
                if 0 <= si < len(sents[j]):
                    gold.append(sents[j][si].strip())
        # distractor paragraphs (titles not in the gold set)
        distract = []
        for j, t in enumerate(titles):
            if t not in gold_titles:
                distract.append(" ".join(s.strip() for s in sents[j]))
        if not gold or not distract:
            continue
        claim = f'The answer to the question "{q}" is: {ans}.'
        gold_ev = " ".join(gold)
        distract_ev = " ".join(distract[:2])  # a couple of distractor paragraphs
        tasks.append({"claim": claim, "evidence": gold_ev, "insufficient": 0})
        tasks.append({"claim": claim, "evidence": distract_ev, "insufficient": 1})
    return tasks


def berry(task: dict) -> dict:
    r = run_detect_hallucination(
        answer=f"{task['claim']} [S1]",
        spans=[{"sid": "S1", "text": task["evidence"]}],
        verifier_model=MODEL,
        max_concurrency=2,
        timeout_s=90,
    )
    d = (r.get("details") or [{}])[0]
    gap = (d.get("budget_gap") or {}).get("min")
    return {
        "score": float(gap) if gap is not None else 0.0,
        "flagged": bool(d.get("flagged")),
        "status": d.get("status"),
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
    bad = 0
    risks = []
    for k, i in enumerate(order, 1):
        bad += labels[i]
        risks.append(bad / k)
    return sum(risks) / len(order)


def prf(pred, labels):
    tp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 0)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "acc": (tp + tn) / max(1, len(labels)),
        "prec": prec,
        "rec": rec,
        "f1": 2 * prec * rec / max(1e-9, prec + rec),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    raw = json.load(open(args.data))
    rows = raw.get("rows", raw) if isinstance(raw, dict) else raw
    tasks = build_tasks(rows)
    out = [None] * len(tasks)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(berry, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = {**tasks[i], **fut.result()}
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    labels = [r["insufficient"] for r in out]
    scores = [r["score"] for r in out]
    flagged = [1 if r["flagged"] else 0 for r in out]
    nosp = sum(1 for r in out if r["status"] == "no_spans")
    print("\n==== BERRY v2 SUFFICIENCY (HotpotQA ablation) ====")
    print(
        f"items={len(out)} claims ({sum(labels)} insufficient / {len(labels) - sum(labels)} sufficient), no_spans={nosp}"
    )
    print(f"AUROC(sufficient vs insufficient) = {auroc(scores, labels):.3f}")
    print(f"AURC                              = {aurc(scores, labels):.3f}")
    print(f"decision @target 0.95             : {prf(flagged, labels)}")
    # mean score by condition
    suf = [s for s, y in zip(scores, labels) if y == 0]
    insuf = [s for s, y in zip(scores, labels) if y == 1]
    print(
        f"mean budget-gap  sufficient={sum(suf) / len(suf):.2f}  insufficient={sum(insuf) / len(insuf):.2f}"
    )


if __name__ == "__main__":
    main()
