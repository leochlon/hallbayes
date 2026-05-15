# Running Berry against local models

Berry's verifier (`detect_hallucination`, `audit_trace_budget`) is backend-agnostic:
as long as the endpoint speaks OpenAI Chat Completions **and** returns `top_logprobs`
on the first generated token, Berry can use it. That means you can run the entire
verification loop on your own hardware.

This guide covers why you'd want to, what you actually need from the runtime, how
to wire up the three runtimes most people pick (LM Studio, vLLM, llama.cpp), and
how to debug the small handful of things that go wrong.

---

## 1. Why local

- **Privacy** — Berry sees your spans verbatim. With a hosted API, those spans
  (file contents, prompts, code, internal docs) leave your machine. A local
  endpoint keeps them inside your loopback interface.
- **Cost** — verification on every claim adds up fast on metered APIs. A local
  3B–20B model handles `detect_hallucination` calls indefinitely for the cost of
  your electricity.
- **Latency** — once a model is warm, a small verifier on Apple Silicon or a
  single GPU answers in well under a second. Hosted endpoints rarely beat that
  for first-token latency.
- **Offline** — works on planes, in air-gapped environments, in CI runners
  with no outbound network, and during third-party API outages.

Trade-off: small local models hallucinate too. Berry compensates by reading
`top_logprobs` to score *calibrated* confidence rather than trusting the model's
top pick. That only works if the runtime actually returns logprobs — which is
the next section.

---

## 2. Hard requirement: `top_logprobs`

Berry's verifier asks the model a closed question and inspects the probability
distribution over the **first** generated token. Concretely, every request looks
like:

```json
{
  "model": "your-model",
  "messages": [...],
  "max_tokens": 1,
  "logprobs": true,
  "top_logprobs": 5,
  "temperature": 0.0
}
```

And every response must include something like:

```json
{
  "choices": [{
    "logprobs": {
      "content": [{
        "token": "Yes",
        "logprob": -0.12,
        "top_logprobs": [
          {"token": "Yes", "logprob": -0.12},
          {"token": "No",  "logprob": -2.30}
        ]
      }]
    }
  }]
}
```

If `logprobs` is `null`, missing, or the runtime silently drops `top_logprobs`,
Berry's calibration math has nothing to work with. The verifier will either
fall back to a degenerate score or refuse the call outright. **No top_logprobs,
no Berry.**

`berry doctor` probes for this and reports `logprobs_populated` / `top_logprobs_nonempty`.

---

## 3. Runtime compatibility matrix

| Runtime              | `top_logprobs` | OpenAI-compat | Notes                                                                                       |
| -------------------- | -------------- | ------------- | ------------------------------------------------------------------------------------------- |
| **LM Studio**        | Yes (>=0.3)    | Yes           | Easiest path on Mac/Windows. GUI + headless server (`lms server start`).                    |
| **vLLM**             | Yes            | Yes           | Production choice. Highest throughput. Requires CUDA or ROCm.                               |
| **llama.cpp server** | Yes (>=b3000)  | Yes           | Lightest footprint. Pure CPU works. Pre-b3000 builds dropped logprobs on the chat endpoint. |
| **MLX-LM**           | Yes (>=0.18)   | Yes (`mlx_lm.server`) | Apple Silicon native. Lower memory than llama.cpp for the same model.               |
| **Ollama**           | **No**         | Partial       | `top_logprobs` is **not** exposed via the OpenAI-compat endpoint as of Ollama 0.5.x. Workaround: run llama.cpp server in front of the same GGUF. |
| **sglang**           | Yes            | Yes           | Good middle ground between vLLM throughput and llama.cpp simplicity.                        |
| **text-generation-webui** | Yes (with `--api`) | Yes  | OpenAI extension lags upstream; verify with `berry doctor` first.                           |

---

## 4. Setup — LM Studio (most common)

### 4.1 Download a model

1. Open LM Studio → **Discover** tab.
2. Pick a small instruct model (verification doesn't need a frontier model):
   - `lmstudio-community/Qwen2.5-7B-Instruct-GGUF` (Q4_K_M is fine)
   - `mlx-community/gpt-oss-20b-MXFP4-Q4` (MoE, fast on Apple Silicon)
3. Download.

### 4.2 Start the server

1. **Developer** tab (sometimes **Local Server**) → load model → toggle Server to Running.
2. Default port `1234`. Endpoint: `http://localhost:1234/v1`.

Headless:

```bash
lms server start
lms load <model-name>
```

### 4.3 Point Berry at it

Edit `~/.berry/mcp_env.json`:

```json
{
  "BERRY_VERIFIER_BACKEND": "local",
  "BERRY_VERIFIER_MODEL": "gpt-oss-20b",
  "BERRY_LOCAL_BASE_URL": "http://127.0.0.1:1234/v1",
  "OPENAI_API_KEY": "not-needed"
}
```

LM Studio doesn't check `OPENAI_API_KEY`, but the SDK requires *some* value.

---

## 5. Setup — vLLM (production)

```bash
pip install vllm
vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype mxfp4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

`~/.berry/mcp_env.json`:

```json
{
  "BERRY_VERIFIER_BACKEND": "local",
  "BERRY_VERIFIER_MODEL": "openai/gpt-oss-20b",
  "BERRY_LOCAL_BASE_URL": "http://127.0.0.1:8000/v1",
  "OPENAI_API_KEY": "not-needed"
}
```

vLLM exposes `top_logprobs` by default. No extra flag needed.

---

## 6. Setup — llama.cpp server (lightweight)

```bash
brew install llama.cpp   # or build from source
llama-server \
  --model /path/to/model.Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 8192 \
  --n-gpu-layers 999 \
  --chat-template llama3
```

`--chat-template` matters — if the GGUF has no embedded template, omitting this
flag will silently produce broken prompts. Common values: `llama3`, `chatml`,
`qwen2`, `phi3`.

`~/.berry/mcp_env.json`:

```json
{
  "BERRY_VERIFIER_BACKEND": "local",
  "BERRY_VERIFIER_MODEL": "local",
  "BERRY_LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
  "OPENAI_API_KEY": "not-needed"
}
```

llama.cpp ignores `model`; pick any short label.

---

## 7. Verification

### 7.1 `berry doctor`

```bash
berry doctor
```

Reports backend, model, base_url, HTTP status, latency, `logprobs_populated`,
`top_logprobs_nonempty`. Exits non-zero if any of those are unhealthy.

### 7.2 Manual curl

```bash
curl -s http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-needed" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role":"user","content":"Reply with only the word Yes."}],
    "max_tokens": 1,
    "logprobs": true,
    "top_logprobs": 5,
    "temperature": 0.0
  }' | python -m json.tool
```

Look for non-null `choices[0].logprobs.content[0].top_logprobs` with >=2 entries.

---

## 8. Calibration

Different local models calibrate differently than `gpt-4.1-nano`. Berry's
default `verification_*_default_target = 0.95` may be unreachable for a given
local model on supported claims, producing false-positive flags.

```bash
python scripts/calibrate_local.py \
  --backend local \
  --model gpt-oss-20b \
  --base-url http://127.0.0.1:1234/v1
```

Script runs 20 known-true + 20 known-false claim/span pairs and prints the
P(YES) histogram for each class plus a suggested threshold. Set it in
`~/.berry/config.json`:

```json
{
  "verification_write_default_target": 0.85,
  "verification_output_default_target": 0.85
}
```

Re-run calibration when you change model or quant level.

---

## 9. Troubleshooting

### `top_logprobs` is `null` or missing

- **Ollama**: not fixable on the OpenAI-compat endpoint. Switch to llama.cpp
  pointing at the same GGUF.
- **llama.cpp**: upgrade past b3000.
- **LM Studio < 0.3**: update.
- **text-generation-webui**: load the `openai` extension; verify with curl.

### "Model not loaded" / 404 on every call

- **LM Studio**: load a model in Developer tab or enable JIT model loading.
- **llama.cpp**: one model per server process; restart with `--model <new>`.
- **vLLM**: one model per server; run multiple instances on different ports.

### Port already in use

```bash
lsof -iTCP:1234 -sTCP:LISTEN
kill $(lsof -tiTCP:1234 -sTCP:LISTEN)
```

### First call takes 30+ seconds

Cold-start cost. Pre-warm before benchmarking. Berry bumps `timeout_s` to 60s
floor automatically when `base_url` is loopback.

### Verifier returns plausible but wrong scores

1. **Calibration mismatch** — re-run `scripts/calibrate_local.py`.
2. **Wrong chat template** — llama.cpp / MLX-LM with no `--chat-template` will
   produce structurally broken prompts; logprobs come back on the wrong tokens.
3. **Temperature drift** — some runtimes ignore `temperature: 0.0`. Force it
   server-side.

### Slow throughput under parallel load

- **vLLM**: `--max-num-seqs 32`.
- **llama.cpp**: `--parallel 4` for continuous batching.
- **LM Studio**: single-request at a time. Switch to vLLM for concurrent load.

---

See also `docs/CONFIGURATION.md` for the full env-var reference and
`docs/CLI.md` for `berry doctor` / `berry setup` flags.
