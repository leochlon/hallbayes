"""
CLI Front-End: Hallucination Risk Checker
----------------------------------------

Interactive CLI:
- Enter/OpenAI API key (or use `OPENAI_API_KEY`).
- Pick a model (default: `gpt-4o-mini`).
- Type your prompt.
- Run a closed-book risk check and see a clear report with next steps.
"""

from __future__ import annotations

import argparse
import os
import sys
from getpass import getpass
from typing import Optional

from scripts.hallucination_toolkit import (
    OpenAIBackend,
    OpenAIItem,
    OpenAIPlanner,
    generate_answer_if_allowed,
)


DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
]


def prompt_multiline(prompt_text: str = "Enter your prompt. End with an empty line:") -> str:
    print(prompt_text)
    print("(Press Enter twice to finish)")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "" and (not lines or lines[-1].strip() == ""):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def select_model_interactive(default: str = DEFAULT_MODELS[0]) -> str:
    print("Select a model:")
    for i, m in enumerate(DEFAULT_MODELS, 1):
        print(f"  {i}) {m}")
    print(f"  {len(DEFAULT_MODELS)+1}) Custom")
    choice = input(f"Model [default {default}]: ").strip()
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(DEFAULT_MODELS):
            return DEFAULT_MODELS[idx-1]
        if idx == len(DEFAULT_MODELS) + 1:
            custom = input("Enter custom model name: ").strip()
            return custom or default
    return choice or default


def get_api_key_interactive() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    key = getpass("Enter OPENAI_API_KEY: ").strip()
    return key


def advice_for_metric(decision_answer: bool, roh: float, isr: float, b2t: float) -> str:
    lines = []
    if decision_answer:
        if roh <= 0.05:
            lines.append("Low estimated risk. Proceed to answer.")
        elif roh <= 0.20:
            lines.append("Moderate risk. Provide a cautious answer and cite uncertainty.")
        else:
            lines.append("Elevated risk. Consider asking for more context or abstaining.")
        lines.append("Log decision with Δ̄, B2T, ISR, and EDFL RoH bound.")
        lines.append("Optionally generate an answer now and review before sharing.")
    else:
        lines.append("Abstain: the evidence-to-answer margin is insufficient.")
        lines.append("Ask for more context/evidence or simplify the question.")
        lines.append("If evidence exists, switch to evidence_erase skeleton policy.")
        lines.append("Alternatively lower risk targets (smaller h*) only if acceptable.")
    lines.append(f"Diagnostic: ISR={isr:.3f}, B2T={b2t:.3f}, RoH≤{roh:.3f} (EDFL).")
    return "\n- ".join(lines)


def run_once(
    api_key: str,
    model: str,
    prompt: str,
    n_samples: int = 7,
    m: int = 6,
    skeleton_policy: str = "closed_book",
    temperature: float = 0.3,
    h_star: float = 0.05,
    isr_threshold: float = 1.0,
    margin_extra_bits: float = 0.0,
    B_clip: float = 12.0,
    clip_mode: str = "one-sided",
    want_answer: bool = False,
) -> int:
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        backend = OpenAIBackend(model=model)
    except Exception as e:
        print("Could not initialize OpenAI backend:", e)
        print("Install `openai>=1.0.0` and set OPENAI_API_KEY.")
        return 2

    item = OpenAIItem(prompt=prompt, n_samples=n_samples, m=m, skeleton_policy=skeleton_policy)
    planner = OpenAIPlanner(backend, temperature=temperature)

    print("\nRunning hallucination risk evaluation…")
    metrics = planner.run(
        [item],
        h_star=h_star,
        isr_threshold=isr_threshold,
        margin_extra_bits=margin_extra_bits,
        B_clip=B_clip,
        clip_mode=clip_mode,
    )

    print("\nReport:")
    for mtr in metrics:
        decision_text = "Answer" if mtr.decision_answer else "Abstain"
        print(f"- Decision: {decision_text}")
        print(f"- Details: {mtr.rationale}")
        advice = advice_for_metric(mtr.decision_answer, mtr.roh_bound, mtr.isr, mtr.b2t)
        print("- Next steps:\n- " + advice)
        if want_answer:
            answer = generate_answer_if_allowed(backend, item, mtr, max_tokens_answer=256)
            print("\nModel answer:\n" + answer if answer else "\nNo answer generated (model abstained or error).")

    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive front-end for hallucination risk checks")
    p.add_argument("--model", default=None, help="OpenAI model name (default: ask interactively)")
    p.add_argument("--prompt-file", default=None, help="Path to a text file with the prompt")
    p.add_argument("--n-samples", type=int, default=7, help="Samples per prompt variant")
    p.add_argument("--m", type=int, default=6, help="Number of skeleton variants")
    p.add_argument("--skeleton", default="closed_book", choices=["closed_book", "evidence_erase", "auto"], help="Skeleton policy")
    p.add_argument("--temperature", type=float, default=0.3, help="Decision sampling temperature")
    p.add_argument("--h-star", type=float, default=0.05, help="Target error rate when answering")
    p.add_argument("--isr-threshold", type=float, default=1.0, help="Minimum ISR to answer")
    p.add_argument("--margin-extra-bits", type=float, default=0.0, help="Extra Δ margin in nats")
    p.add_argument("--B-clip", type=float, default=12.0, help="Clipping bound for Δ contributions")
    p.add_argument("--clip-mode", default="one-sided", choices=["one-sided", "symmetric"], help="Clipping mode")
    p.add_argument("--answer", action="store_true", help="Attempt to generate an answer if allowed")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("No OPENAI_API_KEY found in environment.")
        api_key = get_api_key_interactive()
        if not api_key:
            print("No API key provided; aborting.")
            return 1

    model = args.model or select_model_interactive()
    if not model:
        print("No model selected; aborting.")
        return 1

    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        except Exception as e:
            print(f"Failed to read prompt file: {e}")
            return 1
    else:
        prompt = prompt_multiline()
    if not prompt:
        print("Empty prompt; aborting.")
        return 1

    return run_once(
        api_key=api_key,
        model=model,
        prompt=prompt,
        n_samples=args.n_samples,
        m=args.m,
        skeleton_policy=args.skeleton,
        temperature=args.temperature,
        h_star=args.h_star,
        isr_threshold=args.isr_threshold,
        margin_extra_bits=args.margin_extra_bits,
        B_clip=args.B_clip,
        clip_mode=args.clip_mode,
        want_answer=bool(args.answer),
    )


if __name__ == "__main__":
    sys.exit(main())
# Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License - see LICENSE file for details
