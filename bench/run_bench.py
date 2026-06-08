"""Berry v2 verification benchmark on HaluEval QA.

Each HaluEval record -> two evidence-conditioned claims against the same
knowledge span: the right_answer (label=0, supported) and the
hallucinated_answer (label=1, hallucination). We run Berry's real verifier
(detect_hallucination) on each and compare against a naive single-call LLM judge.

Metrics: AUROC of the hallucination score, AURC (area under risk-coverage),
and decision accuracy/precision/recall/F1 at Berry's prespecified target.

Env: OPENAI_BASE_URL (proxy), OPENAI_API_KEY. Usage:
  python bench/run_bench.py --data /tmp/halu_qa.jsonl --n 100 --out /tmp/bench_out.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from berry.hallucination_detector.core import run_detect_hallucination

MODEL = "gpt-4.1-mini"


def berry_score(answer_text: str, knowledge: str) -> dict:
    r = run_detect_hallucination(
        answer=f"{answer_text} [S1]",
        spans=[{"sid": "S1", "text": knowledge}],
        verifier_model=MODEL,
        max_concurrency=2,
        timeout_s=90,
    )
    d = (r.get("details") or [{}])[0]
    gap = (d.get("budget_gap") or {}).get("min")
    return {
        "score": float(gap) if gap is not None else 0.0,  # higher = more hallucination
        "flagged": bool(d.get("flagged")),
        "status": d.get("status"),
    }


def baseline_judge(answer_text: str, knowledge: str) -> int:
    """Naive single-call LLM judge: is the claim supported by the evidence?"""
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Evidence:\n{knowledge}\n\nClaim: {answer_text}\n\n"
                        "Is the claim fully supported by the evidence? Answer yes or no."
                    ),
                }
            ],
            "max_tokens": 3,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer proxy"},
    )
    try:
        r = json.load(urllib.request.urlopen(req, timeout=60))
        txt = r["choices"][0]["message"]["content"].strip().lower()
        return 0 if txt.startswith("y") else 1  # predicted hallucination
    except Exception:
        return 0


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def aurc(scores: list[float], labels: list[int]) -> float:
    """Area under risk-coverage: answer lowest-score claims first; risk = halluc rate among answered."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    n = len(order)
    risks = []
    bad = 0
    for k, i in enumerate(order, 1):
        bad += labels[i]
        risks.append(bad / k)
    return sum(risks) / n


def risk_at_coverage(scores: list[float], labels: list[int], covs=(0.25, 0.5, 0.75, 0.9)) -> dict:
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    n = len(order)
    out = {}
    for c in covs:
        k = max(1, int(round(c * n)))
        answered = order[:k]
        risk = sum(labels[i] for i in answered) / k
        out[f"risk@cov{c}"] = round(risk, 3)
    return out


def prf(pred: list[int], labels: list[int]) -> dict:
    tp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 0)
    acc = (tp + tn) / max(1, len(labels))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.data) if line.strip()][: args.n]
    tasks = []  # (answer_text, knowledge, label)
    for r in recs:
        tasks.append((r["right_answer"], r["knowledge"], 0))
        tasks.append((r["hallucinated_answer"], r["knowledge"], 1))

    rows = [None] * len(tasks)

    def work(idx):
        ans, kn, label = tasks[idx]
        b = berry_score(ans, kn)
        row = {"idx": idx, "label": label, **b}
        if args.baseline:
            row["baseline_pred"] = baseline_judge(ans, kn)
        return idx, row

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, i) for i in range(len(tasks))]):
            idx, row = fut.result()
            rows[idx] = row
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    flagged = [1 if r["flagged"] else 0 for r in rows]
    nosp = sum(1 for r in rows if r["status"] == "no_spans")

    print("\n==== BERRY v2 verifier ====")
    print(f"items={len(rows)} (claims), no_spans={nosp}")
    print(f"AUROC(score)      = {auroc(scores, labels):.3f}")
    print(f"AURC(risk-cov)    = {aurc(scores, labels):.3f}  (lower=better)")
    print(
        f"risk@coverage     : {risk_at_coverage(scores, labels)}  (base rate={sum(labels) / len(labels):.2f})"
    )
    print(f"decision @target  : {prf(flagged, labels)}")
    if args.baseline:
        bpred = [r.get("baseline_pred", 0) for r in rows]
        print("\n==== naive LLM-judge baseline (single call) ====")
        print(f"decision          : {prf(bpred, labels)}")


if __name__ == "__main__":
    main()
