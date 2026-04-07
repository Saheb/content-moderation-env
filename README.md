---
title: Content Moderation OpenEnv
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# content-moderation-env

**Submission for Meta PyTorch OpenEnv Hackathon Round 1**  
**Scaler School of Technology — March 2026**  
**Author:** Saheb Motiani (@sahebmotiani)

A small, reproducible OpenEnv for user-generated content (UGC) moderation. The environment simulates a content-moderation workflow where an AI agent reviews posts and threads and issues decisions like: keep, warn, remove, or escalate.

## Real-world task
The environment models moderation for a fictional social platform ("VibeNet"). It mirrors workflows used in production moderation systems (e.g., Reddit, Discord, TikTok) and is designed for research and RL experimentation.

## Features
- Four deterministic task packs (easy → medium → hard → very_hard) with adversarially designed post content
- Deterministic graders with scores in [0.0, 1.0]
- Dense reward shaping with partial progress signals and policy-violation penalties
- Thread-context awareness scoring — skipping `view_thread` on context-critical posts incurs grader-level penalties (hard: 3 posts; very_hard: 14 posts)
- Typed Pydantic models (see `models.py`)
- Reproducible inference script (`inference.py`)
- Server ready for local hosting or containerized deployment

## Benchmark Results

> Scores are grader accuracy (0.0–1.0). Rewards are shaped for RL — on the hard task, reward ranges from **−0.93 to +6.39**, providing meaningful separation even when scores converge. The **hard** task is intentionally designed so that no model can achieve a perfect score without reading thread context — ambiguous post framing, adversarial replies, and coded language require genuine policy reasoning.
>
> **Interactive visualization:** [`benchmark.html`](./benchmark.html) — bar charts for scores and rewards across all tiers.

| Model | Provider | Easy (Score / Rwd) | Medium (Score / Rwd) | Hard (Score / Rwd) | Total Time |
|---|---|---|---|---|---|
| Mistral-Nemo 12B (zero-shot) | Local (Ollama) | 0.40 / — | 0.30 / — | 0.09 / — | — |
| DeepSeek-R1-Distill-Qwen 14B | Local (mistral.rs) | **1.00** / +1.37 | 0.00 / +0.31 (timeout) | 0.00 / +0.00 (timeout) | 60m 26s |
| gpt-oss-20b | Cloud (Groq) | **1.00** / +3.15 | **1.00** / +2.57 | **0.82** / +4.99 | 5m 8s |
| gemma-4-31b-it | Cloud (Groq/Google) | **1.00** / +2.25 | **1.00** / +3.46 | 0.66 / -0.93 | 7m 34s |
| Llama 3.3 70B Versatile | Cloud (Groq) | **1.00** / +0.65 | **1.00** / +1.70 | **0.82** / +1.99 | 6m 6s |
| gpt-oss-120b | Cloud (Groq) | **1.00** / +2.25 | **1.00** / +2.95 | **0.82** / +5.86 | 5m 32s |
| Gemini 3.1 Flash Lite | Cloud (Google) | **1.00** / +1.90 | **1.00** / +3.30 | **0.82** / **+6.39** | 8m 19s |
| Qwen 3 32B | Cloud (Groq) | **1.00** / +2.25 | **1.00** / +4.72 | **0.82** / **+6.39** | 6m 23s |

### Why reward matters more than score on the hard task

Easy and medium show a ceiling effect — every cloud model hits 1.00. Hard shows score convergence at 0.82 for most models. But reward on hard ranges from **−0.93** (gemma) to **+6.39** (Gemini Flash Lite, Qwen 3 32B), which reveals genuine quality differences in how efficiently and confidently models reach correct decisions. Score alone is insufficient for evaluating frontier moderation agents; reward shaping is what separates them.

---

## Requirements
- Python: >= 3.10 (see `pyproject.toml`)
- Key Python dependencies: `openenv-core`, `pydantic`, `fastapi`, `uvicorn`

## Quick Setup
Two options are supported: using `uv` (recommended) or a traditional `venv`.

### Option A — using `uv` (recommended)
```bash
# Install `uv` if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/YOUR-USERNAME/content-moderation-env.git
cd content-moderation-env

# Create venv and install
uv venv
source .venv/bin/activate
uv pip install -e .

# Optional: validate OpenEnv metadata (if you have `openenv` installed)
openenv validate || true
```

### Option B — traditional venv + pip
```bash
git clone https://github.com/YOUR-USERNAME/content-moderation-env.git
cd content-moderation-env

python -m venv venv
source venv/bin/activate
pip install -e .

# Optional validation
openenv validate || true
```

## Run locally
Start the server (development):

```bash
# with `uv` wrapper
uv run uvicorn server.app:app --reload --port 8000

# or directly (when venv active)
uvicorn server.app:app --reload --port 8000
```

Run the baseline inference script. It supports any OpenAI-compatible API via environment variables (complies with Hackathon Pre-Submission schema):

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | falls back to `OPENAI_API_KEY` | Hugging Face API key or OpenAI-compatible key |
| `OPENAI_API_KEY` | *(empty)* | Optional OpenAI API key (used directly if set) |
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM provider endpoint |
| `MODEL_NAME` | `gpt-4o` | Model name for generation |
| `ENV_CLIENT_BASE_URL` | `http://localhost:8000` | Environment server endpoint (for deployed HF/OpenEnv instances) |

### Quick test against deployed HF Space (no local server needed)
```bash
cp .env.example .env
# Edit .env and set your HF_TOKEN
uv run python inference.py
```

### Run with different providers
```bash
# OpenAI (default)
export HF_TOKEN=sk-...
uv run python inference.py

# Google Gemini
export HF_TOKEN=AIza...
export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_NAME=gemini-2.0-flash
uv run python inference.py

# Groq
export HF_TOKEN=gsk_...
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
uv run python inference.py

# Local Ollama
export API_BASE_URL=http://localhost:11434/v1
export HF_TOKEN=ollama
export MODEL_NAME=llama3
uv run python inference.py
```

## Deploying (Hugging Face Spaces)
You can deploy using a Docker-based Space:
- Create a new Space on https://huggingface.co/new-space and choose the Docker template
- Link this GitHub repository
- The root `Dockerfile` will be used to build the container

## Project layout
```
content-moderation-env/
├── data/                  # Deterministic task packs (easy/medium/hard/very_hard)
│   ├── easy.json
│   ├── medium.json
│   ├── hard.json
│   └── very_hard.json
├── models.py              # Typed Action, Observation, State
├── server/                # FastAPI OpenEnv server
│   ├── __init__.py
│   ├── environment.py     # Core logic + rewards + graders
│   └── app.py
├── benchmark.html         # Interactive benchmark visualization
├── Dockerfile             # Container configuration for HF Spaces
├── inference.py           # Reproducible baseline inference script
├── openenv.yaml           # OpenEnv metadata
├── pyproject.toml
└── README.md
```

## Notes & missing/optional files
- `.dockerignore` is optional; add one if you plan to build Docker images locally.

## Action & Observation spaces
Full type definitions live in `models.py`.

Actions (`ModerationAction`) include: `view_post`, `view_thread`, `categorize`, `moderate`, `lookup_policy`, `escalate`.

Observations (`ModerationObservation`) include: queue summary, current post, thread context, and last action result.

Rewards are dense and shaped for RL (typical per-step range ≈ -0.55 to +0.63).

## Contributing
- Open an issue or a pull request with a description of the change.
- For code changes, ensure a reproducible environment and include minimal repro steps.

## License
This project is licensed under the [MIT License](LICENSE).