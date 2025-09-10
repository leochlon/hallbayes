"""
Streamlit Web UI — Closed-Book Hallucination Risk
-------------------------------------------------

Browser UI:
- Enter/OpenAI API key (or rely on env var)
- Pick model, tune evaluation knobs
- Enter prompt, run evaluation
- See decision, Δ̄, B2T, ISR, EDFL RoH bound, next‑step guidance
- Optionally generate an answer (if allowed) and export SLA JSON

Run:
  pip install streamlit openai>=1.0.0
  streamlit run app/web/web_app.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
import sys
from pathlib import Path

# Ensure repo root is on sys.path so we can import `scripts` and semantic modules
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from scripts.hallucination_toolkit import (
    OpenAIBackend,
    OpenAIItem as ClassicItem,
    OpenAIPlanner as ClassicPlanner,
    generate_answer_if_allowed as classic_generate_answer,
    make_sla_certificate,
)

# Semantic (experimental) engine
try:
    from scripts import hallucination_toolkit_semantic as semx
except Exception:
    semx = None


DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
]


def advice_for_metric(decision_answer: bool, roh: float, isr: float, b2t: float) -> list[str]:
    tips: list[str] = []
    if decision_answer:
        if roh <= 0.05:
            tips.append("Low estimated risk. Proceed to answer.")
        elif roh <= 0.20:
            tips.append("Moderate risk. Provide a cautious answer and cite uncertainty.")
        else:
            tips.append("Elevated risk. Consider asking for more context or abstaining.")
        tips.append("Log decision with Δ̄, B2T, ISR, and EDFL RoH bound.")
        tips.append("Optionally generate an answer now and review before sharing.")
    else:
        tips.append("Abstain: the evidence-to-answer margin is insufficient.")
        tips.append("Ask for more context/evidence or simplify the question.")
        tips.append("If evidence exists, switch to evidence_erase skeleton policy.")
        tips.append("Alternatively lower risk targets (smaller h*) only if acceptable.")
    tips.append(f"Diagnostic: ISR={isr:.3f}, B2T={b2t:.3f}, RoH≤{roh:.3f} (EDFL).")
    return tips


def sidebar_controls():
    st.sidebar.header("Configure")

    env_key = os.environ.get("OPENAI_API_KEY", "")

    with st.sidebar.expander("API & Model", expanded=True):
        api_key = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            help="If left empty, the app will try OPENAI_API_KEY from the environment.",
            placeholder=("Using OPENAI_API_KEY from env" if env_key else "sk-..."),
        ).strip()

        model_choice = st.selectbox(
            "Model",
            options=DEFAULT_MODELS + ["Custom…"],
            index=0,
        )
        custom_model = st.text_input("Custom model", value="") if model_choice == "Custom…" else ""
        model = (custom_model.strip() or DEFAULT_MODELS[0]) if model_choice == "Custom…" else model_choice

    engine = st.sidebar.radio("Engine", ["Semantic (experimental)", "Classic"], index=0)

    with st.sidebar.expander("Decision thresholds", expanded=False):
        h_star = st.slider("h* (target error when answering)", 0.005, 0.25, 0.05, 0.005)
        isr_threshold = st.slider("ISR threshold", 0.5, 3.0, 1.0, 0.1)
        margin_extra_bits = st.slider("Extra Δ margin (nats)", 0.0, 4.0, 0.0, 0.1)

    with st.sidebar.expander("Advanced", expanded=False):
        # Match defaults to selected engine; semantic engine defaults to 1.0 (as in test.py)
        default_temp = 1.0 if engine == "Semantic (experimental)" else 0.3
        temperature = st.slider("temperature (decision)", 0.0, 1.0, float(default_temp), 0.05)
        if engine == "Classic":
            skeleton_policy = st.selectbox("Skeleton policy", ["closed_book", "evidence_erase", "auto"], index=0)
            n_samples = st.slider("n_samples (per prompt)", 1, 12, 7)
            m = st.slider("m (skeleton variants)", 2, 12, 6)
            B_clip = st.slider("B_clip", 1.0, 32.0, 12.0, 1.0)
            clip_mode = st.selectbox("clip_mode", ["one-sided", "symmetric"], index=0)
        else:
            # Semantic v2.4 (experimental) knobs
            n_samples = st.slider("n_samples per skeleton", 1, 12, 6)
            m = st.slider("m (skeleton variants)", 2, 12, 6)
            M_full = st.slider("M_full (full-prompt samples)", 4, 20, 10)
            entailment_timeout_s = st.slider("Entailment timeout (s)", 5.0, 60.0, 15.0, 1.0)
            max_entail_calls = st.slider("Max entailment calls", 30, 300, 120, 10)
            max_workers_skeletons = st.slider("Parallel skeleton calls", 1, 16, 6)
            use_llr = st.checkbox("Use LLR top-meaning budget", value=True)
            kl_eps = st.number_input("KL smoothing epsilon", min_value=0.0, max_value=0.1, value=0.01, step=0.005, format="%.3f")
            max_workers_items = st.slider("Parallel items", 1, 16, 1)

    with st.sidebar.expander("Answer generation", expanded=False):
        want_answer = st.checkbox("Generate answer if allowed", value=False)

        cfg = {
        "api_key": api_key or env_key,
        "model": model,
        "engine": engine,
        "n_samples": int(n_samples),
        "m": int(m),
        "skeleton_policy": (locals().get("skeleton_policy") or "auto"),
        "temperature": float(temperature),
        "h_star": float(h_star),
        "isr_threshold": float(isr_threshold),
        "margin_extra_bits": float(margin_extra_bits),
        "B_clip": float(locals().get("B_clip", 12.0)),
        "clip_mode": locals().get("clip_mode", "one-sided"),
        "M_full": int(locals().get("M_full", 10)),
        "entailment_timeout_s": float(locals().get("entailment_timeout_s", 15.0)),
        "max_entail_calls": int(locals().get("max_entail_calls", 120)),
        "want_answer": bool(want_answer),
        # no v2.3 concurrency knob anymore
        "max_workers_skeletons": int(locals().get("max_workers_skeletons", 6)),
        "use_llr": bool(locals().get("use_llr", True)),
        "kl_eps": float(locals().get("kl_eps", 0.01)),
        "max_workers_items": int(locals().get("max_workers_items", 1)),
        }
    return cfg


def main() -> None:
    st.set_page_config(page_title="Hallucination Risk Checker", page_icon="🧪", layout="centered")
    st.title("Closed‑Book Hallucination Risk Checker")
    st.caption("OpenAI‑only; Classic and Semantic (experimental) engines available.")

    cfg = sidebar_controls()

    prompt = st.text_area(
        "Enter your prompt",
        height=180,
        placeholder="Ask a question (no external evidence).",
    ).strip()

    col_run, col_reset = st.columns([1, 1])
    run_clicked = col_run.button("Run evaluation", type="primary")
    reset_clicked = col_reset.button("Reset")
    if reset_clicked:
        st.experimental_rerun()

    if run_clicked:
        if not prompt:
            st.warning("Please enter a prompt.")
            return
        if not cfg["api_key"]:
            st.error("OPENAI_API_KEY is missing. Provide it in the sidebar or set the environment variable.")
            return

        os.environ["OPENAI_API_KEY"] = cfg["api_key"]

        if cfg["engine"] == "Classic":
            item = ClassicItem(
                prompt=prompt,
                n_samples=cfg["n_samples"],
                m=cfg["m"],
                skeleton_policy=cfg["skeleton_policy"],
            )

            try:
                backend = OpenAIBackend(model=cfg["model"])
            except Exception as e:
                st.error(f"Failed to initialize OpenAI backend: {e}")
                st.info("Install `openai>=1.0.0` and ensure the API key is valid.")
                return

            planner = ClassicPlanner(
                backend=backend,
                temperature=cfg["temperature"],
            )

            with st.spinner("Evaluating (Classic)…"):
                metrics = planner.run(
                    [item],
                    h_star=cfg["h_star"],
                    isr_threshold=cfg["isr_threshold"],
                    margin_extra_bits=cfg["margin_extra_bits"],
                    B_clip=cfg["B_clip"],
                    clip_mode=cfg["clip_mode"],
                )
            m = metrics[0]
        else:
            if semx is None:
                st.error("Semantic (experimental) engine is unavailable. Ensure 'scripts/hallucination_toolkit_semantic.py' is present.")
                return

            try:
                backend = semx.OpenAIChatBackend(model=cfg["model"], debug=False, strict=True)
            except Exception as e:
                st.error(f"Failed to initialize OpenAI backend (v2.4): {e}")
                st.info("Install `openai>=1.0.0` and ensure the API key is valid.")
                return

            item = semx.OpenAIItem(
                prompt=prompt,
                n_samples_per_skeleton=cfg["n_samples"],
                M_full=cfg["M_full"],
                m_skeletons=cfg["m"],
                skeleton_policy=cfg["skeleton_policy"],
            )
            planner = semx.SemanticPlanner(
                backend,
                temperature=cfg["temperature"],
                top_p=0.9,
                max_tokens_answer=48,
                max_entail_calls=cfg["max_entail_calls"],
                entailment_timeout_s=cfg["entailment_timeout_s"],
                use_llr_top_meaning_budget=cfg["use_llr"],
                max_workers_skeletons=cfg["max_workers_skeletons"],
                kl_smoothing_eps=cfg["kl_eps"],
            )
            with st.spinner("Evaluating (Semantic experimental)…"):
                metrics = planner.run(
                    [item],
                    h_star=cfg["h_star"],
                    isr_threshold=cfg["isr_threshold"],
                    margin_extra_bits=cfg["margin_extra_bits"],
                    max_workers_items=cfg["max_workers_items"],
                )
            m = metrics[0]

        decision_str = "Answer" if m.decision_answer else "Abstain"
        if m.decision_answer:
            st.success(f"Decision: {decision_str}")
        else:
            st.warning(f"Decision: {decision_str}")

        st.code(m.rationale, language="text")

        st.subheader("Next steps")
        for tip in advice_for_metric(m.decision_answer, m.roh_bound, m.isr, m.b2t):
            st.markdown(f"- {tip}")

        if cfg["want_answer"] and m.decision_answer:
            if cfg["engine"] == "Classic":
                with st.spinner("Generating answer…"):
                    ans = classic_generate_answer(backend, item, m, max_tokens_answer=256)
                if ans:
                    st.subheader("Model answer")
                    st.write(ans)
                else:
                    st.info("No answer generated (model abstained or error).")
            else:
                # v2.4: generate a short answer directly
                with st.spinner("Generating answer…"):
                    try:
                        msgs = semx.short_answer_messages(prompt)
                        out = backend.sample_with_logprobs(
                            msgs,
                            n=1,
                            temperature=0.4,
                            max_tokens=256,
                            top_p=1.0,
                            logprobs=False,
                            top_logprobs=None,
                        )
                        ans = out[0][0] if out else ""
                    except Exception as e:
                        ans = ""
                        st.info(f"Answer generation failed: {e}")
                if ans:
                    st.subheader("Model answer")
                    st.write(ans)
                else:
                    st.info("No answer generated (model abstained or error).")

        # SLA export available only for Classic planner (where aggregation is defined)
        if cfg["engine"] == "Classic":
            report = planner.aggregate([item], metrics, h_star=cfg["h_star"], isr_threshold=cfg["isr_threshold"], margin_extra_bits=cfg["margin_extra_bits"])  # type: ignore
            cert = make_sla_certificate(report, model_name=cfg["model"], confidence_1_minus_alpha=0.95)
            cert_json = json.dumps(asdict(cert), indent=2)
            st.download_button(
                "Download SLA certificate (JSON)",
                data=cert_json,
                file_name="sla_certificate.json",
                mime="application/json",
            )

        st.caption("All information measures are in nats. Closed‑book uses semantic masking.")
        st.caption("Built by Hassana Labs — https://hassana.io")


if __name__ == "__main__":
    main()
# Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License - see LICENSE file for details
