"""RAGTruth grounding benchmark, scored the proper way: atomic-claim decomposition.

Pipeline (how RAGTruth is meant to be scored):
  1. decompose each response into atomic, decontextualized claims (1 LLM call/response)
  2. verify each atomic claim against the context with Berry's verifier
  3. aggregate to a response-level decision and compare to RAGTruth's response label

Contrast with run_ragtruth.py, which used Berry's default sentence split. Atomic
units isolate the single ungrounded fact, so precision should improve.

Env: OPENAI_BASE_URL, OPENAI_API_KEY. Usage:
  python bench/run_ragtruth_atomic.py --data /tmp/ragtruth.json --n 400 --out /tmp/rta.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from berry.hallucination_detector.core import run_detect_hallucination

MODEL = "gpt-4.1-mini"
MAX_CLAIMS = 15

DECOMP_PROMPT = (
    "Break the response into a numbered list of atomic factual claims.\n"
    "Rules: each line states ONE fact; make it self-contained (resolve pronouns, "
    "dates, entities); copy only what the response asserts -- do not add, infer, or "
    "correct anything. Output one claim per line, no commentary.\n\nResponse:\n"
)


def _chat(content: str, max_tokens: int) -> str:
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer proxy"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=90))
    return r["choices"][0]["message"]["content"]


def decompose(output: str) -> list[str]:
    txt = _chat(DECOMP_PROMPT + output, max_tokens=900)
    claims = []
    for line in txt.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip()
        if len(line) > 3:
            claims.append(line)
    return claims[:MAX_CLAIMS] or [output.strip()]


def verify(claim: str, context: str) -> dict:
    r = run_detect_hallucination(
        answer=claim,
        spans=[{"sid": "S1", "text": context}],
        verifier_model=MODEL,
        context_mode="all",
        max_concurrency=1,
        timeout_s=90,
    )
    d = (r.get("details") or [{}])[0]
    gap = (d.get("budget_gap") or {}).get("min")
    return {"score": float(gap) if gap is not None else 0.0, "flagged": bool(d.get("flagged"))}


def label_of(r: dict) -> int:
    p = r["hallucination_labels_processed"]
    if isinstance(p, str):
        p = json.loads(p)
    return 1 if (p.get("evident_conflict", 0) or p.get("baseless_info", 0)) else 0


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

    # Phase 1: decompose every response into atomic claims
    print(f"phase 1: decomposing {len(rows)} responses ...", flush=True)
    claims_by_resp = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(decompose, r["output"]): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            claims_by_resp[futs[fut]] = fut.result()
    total_claims = sum(len(c) for c in claims_by_resp)
    print(
        f"  {total_claims} atomic claims (mean {total_claims / len(rows):.1f}/response)", flush=True
    )

    # Phase 2: verify every atomic claim (flattened)
    flat = [(i, c) for i, cs in enumerate(claims_by_resp) for c in cs]
    print(f"phase 2: verifying {len(flat)} atomic claims ...", flush=True)
    results = [None] * len(flat)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(verify, c, rows[i]["context"]): k for k, (i, c) in enumerate(flat)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(flat)}", flush=True)

    # Phase 3: aggregate per response
    agg = [{"scores": [], "flags": []} for _ in rows]
    for (i, _c), res in zip(flat, results):
        agg[i]["scores"].append(res["score"])
        agg[i]["flags"].append(res["flagged"])
    out = []
    for i, r in enumerate(rows):
        sc = agg[i]["scores"] or [0.0]
        out.append(
            {
                "id": r.get("id"),
                "label": label_of(r),
                "n_claims": len(claims_by_resp[i]),
                "score": max(sc),
                "flagged": any(agg[i]["flags"]),
            }
        )
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    labels = [r["label"] for r in out]
    scores = [r["score"] for r in out]
    flagged = [1 if r["flagged"] else 0 for r in out]
    print("\n==== BERRY v2 GROUNDING on RAGTruth (ATOMIC decomposition) ====")
    print(
        f"responses={len(out)}  base rate={sum(labels) / len(labels):.2f}  atomic claims/resp={total_claims / len(rows):.1f}"
    )
    print(f"AUROC = {auroc(scores, labels):.3f}   AURC = {aurc(scores, labels):.3f}")
    print(f"decision @target 0.95 : {prf(flagged, labels)}")
    print("(compare sentence-split run: AUROC 0.810, prec 0.374, rec 0.937)")


if __name__ == "__main__":
    main()
