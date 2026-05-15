#!/usr/bin/env python3
"""Calibrate verification thresholds for a Berry verifier backend.

Runs a fixed 20-true + 20-false claim set against the configured backend
and reports the P(YES) distribution plus a suggested
`verification_*_default_target` value for ~/.berry/config.json.

Usage:
    python scripts/calibrate_local.py --backend local \\
        --model gpt-oss-20b \\
        --base-url http://127.0.0.1:1234/v1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from berry.hallucination_detector.core import run_detect_hallucination  # noqa: E402

TRUE_CASES: List[Tuple[str, str]] = [
    ("Paris is the capital of France.", "Paris is the capital of France."),
    ("Water boils at 100 degrees Celsius at 1 atmosphere of pressure.",
     "At standard atmospheric pressure of 1 atm, water boils at 100 degrees Celsius."),
    ("The Earth orbits the Sun.", "The Earth orbits the Sun once per year."),
    ("Mount Everest is the tallest mountain on Earth.",
     "Mount Everest, at 8,848 meters, is the tallest mountain above sea level on Earth."),
    ("Humans have 23 pairs of chromosomes.",
     "A typical human cell contains 23 pairs of chromosomes."),
    ("The Pacific Ocean is the largest ocean.",
     "The Pacific Ocean is the largest and deepest of the world's oceans."),
    ("World War II ended in 1945.",
     "World War II ended in 1945 with the surrender of Japan."),
    ("The chemical symbol for gold is Au.",
     "Gold has the chemical symbol Au and atomic number 79."),
    ("Light travels faster than sound.",
     "Light travels at roughly 300,000 km/s, far faster than the speed of sound in air."),
    ("The Great Wall of China is located in China.",
     "The Great Wall of China stretches across northern China."),
    ("Shakespeare wrote Hamlet.",
     "William Shakespeare is the author of the tragedy Hamlet."),
    ("A triangle has three sides.",
     "By definition, every triangle is a polygon with exactly three sides."),
    ("The human heart has four chambers.",
     "The human heart consists of four chambers: two atria and two ventricles."),
    ("DNA is a double helix.",
     "DNA molecules are structured as a double helix of two complementary strands."),
    ("The Amazon is a river in South America.",
     "The Amazon River flows through several countries in South America."),
    ("Mercury is the planet closest to the Sun.",
     "Mercury is the innermost planet in the Solar System, closest to the Sun."),
    ("Oxygen is required for human respiration.",
     "Human cellular respiration requires oxygen to produce energy from glucose."),
    ("English is spoken in the United Kingdom.",
     "English is the primary language of the United Kingdom."),
    ("The speed of light in a vacuum is approximately 300,000 km/s.",
     "Light travels at approximately 299,792 km/s in a vacuum."),
    ("Cats are mammals.",
     "Domestic cats are small carnivorous mammals of the family Felidae."),
]

FALSE_CASES: List[Tuple[str, str]] = [
    ("Paris is the capital of Germany.", "Paris is the capital of France."),
    ("Water boils at 50 degrees Celsius at 1 atmosphere.",
     "Water boils at 100 degrees Celsius at 1 atm of pressure."),
    ("The Sun orbits the Earth.", "The Earth orbits the Sun once per year."),
    ("Mount Everest is in Australia.",
     "Mount Everest is located on the border between Nepal and Tibet."),
    ("Humans have 50 pairs of chromosomes.",
     "A typical human cell contains 23 pairs of chromosomes."),
    ("The Atlantic Ocean is the largest ocean on Earth.",
     "The Pacific Ocean is the largest and deepest of the world's oceans."),
    ("World War II ended in 1965.",
     "World War II ended in 1945 with the surrender of Japan."),
    ("The chemical symbol for gold is Go.",
     "Gold has the chemical symbol Au and atomic number 79."),
    ("Sound travels faster than light.",
     "Light travels at roughly 300,000 km/s, far faster than the speed of sound in air."),
    ("The Great Wall of China is located in Brazil.",
     "The Great Wall of China stretches across northern China."),
    ("Charles Dickens wrote Hamlet.",
     "William Shakespeare is the author of the tragedy Hamlet."),
    ("A triangle has five sides.",
     "By definition, every triangle is a polygon with exactly three sides."),
    ("The human heart has two chambers.",
     "The human heart consists of four chambers: two atria and two ventricles."),
    ("DNA is a single strand.",
     "DNA molecules are structured as a double helix of two complementary strands."),
    ("The Amazon is a river in Europe.",
     "The Amazon River flows through several countries in South America."),
    ("Jupiter is the planet closest to the Sun.",
     "Mercury is the innermost planet in the Solar System, closest to the Sun."),
    ("Nitrogen alone is sufficient for human cellular respiration.",
     "Human cellular respiration requires oxygen to produce energy from glucose."),
    ("Mandarin is the primary language of the United Kingdom.",
     "English is the primary language of the United Kingdom."),
    ("Light travels at approximately 3 km/s in a vacuum.",
     "Light travels at approximately 299,792 km/s in a vacuum."),
    ("Cats are reptiles.",
     "Domestic cats are small carnivorous mammals of the family Felidae."),
]


def _configure_env(backend: str, base_url: str, api_key: str) -> None:
    backend = (backend or "openai").strip().lower()
    if backend == "local":
        os.environ["BERRY_VERIFIER_BACKEND"] = "local"
        if base_url:
            os.environ["BERRY_LOCAL_BASE_URL"] = base_url
    else:
        os.environ["BERRY_VERIFIER_BACKEND"] = backend
        if base_url:
            os.environ["BERRY_OPENAI_BASE_URL"] = base_url
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "not-needed"


def _score_one(claim: str, span_text: str, *, model: str) -> float:
    result = run_detect_hallucination(
        answer=claim,
        spans=[{"sid": "S1", "text": span_text}],
        verifier_model=model,
        default_target=0.95,
        max_claims=1,
        temperature=0.0,
        top_logprobs=10,
        max_concurrency=1,
        timeout_s=60.0,
        units="bits",
    )
    details = result.get("details") or []
    if not details:
        return float("nan")
    return float((details[0].get("post_yes") or {}).get("p_lower") or 0.0)


def _histogram(label: str, probs: List[float], *, width: int = 40) -> None:
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    counts = [0] * (len(bins) - 1)
    for p in probs:
        for i in range(len(bins) - 1):
            if bins[i] <= p < bins[i + 1]:
                counts[i] += 1
                break
    peak = max(counts) or 1
    print(f"\n{label} (n={len(probs)})")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        bar = "#" * int(round(counts[i] / peak * width))
        print(f"  [{lo:.1f}, {hi:.1f}) | {bar:<{width}} {counts[i]}")
    if probs:
        mean = sum(probs) / len(probs)
        srt = sorted(probs)
        median = srt[len(srt) // 2] if len(srt) % 2 else 0.5 * (srt[len(srt) // 2 - 1] + srt[len(srt) // 2])
        print(f"  mean={mean:.3f}  median={median:.3f}  min={min(probs):.3f}  max={max(probs):.3f}")


def _suggest_threshold(true_p: List[float], false_p: List[float]) -> Tuple[float, float, float]:
    best = (0.95, 0.0, 0.0, -1.0)
    n_true = max(1, len(true_p))
    n_false = max(1, len(false_p))
    T = 0.50
    while T <= 0.991:
        tpr = sum(1 for p in true_p if p >= T) / n_true
        frr = sum(1 for p in false_p if p < T) / n_false
        score = tpr * frr
        if score > best[3] or (abs(score - best[3]) < 1e-9 and T > best[0]):
            best = (round(T, 2), tpr, frr, score)
        T += 0.01
    return best[0], best[1], best[2]


def parse_args(argv: List[str]) -> argparse.Namespace:
    env_backend = (os.environ.get("BERRY_VERIFIER_BACKEND") or "openai").strip().lower()
    p = argparse.ArgumentParser(description="Calibrate Berry verifier thresholds.")
    p.add_argument("--backend", choices=["openai", "local", "gemini", "vertex"],
                   default=env_backend if env_backend in {"openai", "local", "gemini", "vertex"} else "openai")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="")
    p.add_argument("--api-key", default="")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, run only the first N items from each set.")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    _configure_env(args.backend, args.base_url, args.api_key)

    trues = TRUE_CASES if args.limit <= 0 else TRUE_CASES[: args.limit]
    falses = FALSE_CASES if args.limit <= 0 else FALSE_CASES[: args.limit]

    print(f"Backend: {args.backend}  model: {args.model}  base_url: {args.base_url or '(default)'}")
    print(f"Running {len(trues)} true and {len(falses)} false cases...\n")

    true_p: List[float] = []
    for i, (claim, span) in enumerate(trues, 1):
        p = _score_one(claim, span, model=args.model)
        true_p.append(p)
        print(f"  T{i:02d}  P(YES)={p:.3f}  {claim[:60]}")

    false_p: List[float] = []
    for i, (claim, span) in enumerate(falses, 1):
        p = _score_one(claim, span, model=args.model)
        false_p.append(p)
        print(f"  F{i:02d}  P(YES)={p:.3f}  {claim[:60]}")

    _histogram("TRUE  claims P(YES)", true_p)
    _histogram("FALSE claims P(YES)", false_p)

    T, tpr, frr = _suggest_threshold(true_p, false_p)
    overlap = sum(1 for p in false_p if p >= T) + sum(1 for p in true_p if p < T)

    print("\n=== Calibration summary ===")
    print(f"  Suggested threshold T          : {T:.2f}")
    print(f"  True  pass rate  (P>=T)        : {tpr:.2%}")
    print(f"  False reject rate (P<T)        : {frr:.2%}")
    print(f"  Misclassified items            : {overlap}/{len(true_p) + len(false_p)}")
    print(f"  Score (tpr * frr)              : {tpr * frr:.3f}")

    print("\nRecommendation:")
    print(f"  Set verification_write_default_target  to {T:.2f} in ~/.berry/config.json")
    print(f"  Set verification_output_default_target to {T:.2f} in ~/.berry/config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
