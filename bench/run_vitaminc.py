"""VitaminC sufficiency benchmark for Berry v2, sliced by failure axis.

VitaminC is evidence-relative (same claim flips label as evidence changes), so it
tests grounding, not world-correctness. We separate the two axes:
  SUPPORTS         = evidence sufficient + entails   (Berry should PASS,  y=0)
  NOT ENOUGH INFO  = insufficient                    (Berry should FLAG,  y=1)  <- pure sufficiency
  REFUTES          = sufficient but contradicts      (direction axis; reported separately)

Headline = AUROC on SUPPORTS-vs-NEI (pure sufficiency). Plus the within-claim
contrastive paired delta: same claim, evidence changes -> does Berry's budget move?

Env: OPENAI_BASE_URL, OPENAI_API_KEY. Usage:
  python bench/run_vitaminc.py --data /tmp/vitaminc.json --out /tmp/vc_out.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from berry.hallucination_detector.core import run_detect_hallucination

MODEL = "gpt-4.1-mini"


def berry(claim: str, evidence: str) -> dict:
    r = run_detect_hallucination(
        answer=claim,
        spans=[{"sid": "S1", "text": evidence}],
        verifier_model=MODEL,
        context_mode="all",
        max_concurrency=2,
        timeout_s=90,
    )
    d = (r.get("details") or [{}])[0]
    gap = (d.get("budget_gap") or {}).get("min")
    return {"score": float(gap) if gap is not None else 0.0, "flagged": bool(d.get("flagged"))}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    rows = json.load(open(args.data))[: args.n]
    out = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(berry, r["claim"], r["evidence"]): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = {"label": rows[i]["label"], "claim": rows[i]["claim"], **fut.result()}
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(rows)}", flush=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    def sel(lbls):
        return [r for r in out if r["label"] in lbls]

    sup, nei, ref = sel(["SUPPORTS"]), sel(["NOT ENOUGH INFO"]), sel(["REFUTES"])
    print(
        f"\n==== BERRY v2 on VitaminC (n={len(out)}: SUP {len(sup)} / NEI {len(nei)} / REF {len(ref)}) ===="
    )

    # pure sufficiency: SUPPORTS(0) vs NEI(1)
    s = [r["score"] for r in sup + nei]
    y = [0] * len(sup) + [1] * len(nei)
    print(f"PURE SUFFICIENCY  AUROC(SUPPORTS vs NEI) = {auroc(s, y):.3f}")
    # full grounding: SUPPORTS(0) vs NEI+REFUTES(1)
    s2 = [r["score"] for r in sup + nei + ref]
    y2 = [0] * len(sup) + [1] * (len(nei) + len(ref))
    print(f"FULL GROUNDING    AUROC(SUPPORTS vs NEI+REFUTES) = {auroc(s2, y2):.3f}")
    # per-label behaviour
    for name, grp in [("SUPPORTS", sup), ("NOT ENOUGH INFO", nei), ("REFUTES", ref)]:
        if not grp:
            continue
        flag = sum(1 for r in grp if r["flagged"]) / len(grp)
        ms = sum(r["score"] for r in grp) / len(grp)
        tag = "FP rate" if name == "SUPPORTS" else "recall"
        print(f"  {name:16s}: {tag}={flag:.2f}  mean_gap={ms:.1f}")

    # within-claim contrastive paired delta (same claim, SUPPORTS vs non-SUPPORTS evidence)
    byc = defaultdict(lambda: defaultdict(list))
    for r in out:
        byc[r["claim"]]["sup" if r["label"] == "SUPPORTS" else "insuf"].append(r["score"])
    deltas = []
    for c, g in byc.items():
        if g["sup"] and g["insuf"]:
            deltas.append(sum(g["insuf"]) / len(g["insuf"]) - sum(g["sup"]) / len(g["sup"]))
    if deltas:
        pos = sum(1 for d in deltas if d > 0)
        print(f"\nCONTRASTIVE (same claim, evidence flips): {len(deltas)} paired claims")
        print(
            f"  mean delta budget-gap (insufficient - sufficient) = {sum(deltas) / len(deltas):+.1f}"
        )
        print(
            f"  budget correctly rises when evidence weakens: {pos}/{len(deltas)} = {pos / len(deltas):.2f}"
        )


if __name__ == "__main__":
    main()
