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
- Three deterministic task packs (easy → medium → hard)
- Deterministic graders with scores in [0.0, 1.0]
- Dense reward shaping with partial progress signals and policy-violation penalties
- Typed Pydantic models (see `models.py`)
- Reproducible baseline script (`baseline.py`)
- Server ready for local hosting or containerized deployment

## Baseline scores (Mistral-Nemo 12B, zero-shot via Ollama)
- easy: **0.40**
- medium: **0.30**
- hard: **0.09**

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
uv run uvicorn server.app:create_app --factory --reload --port 8000

# or directly (when venv active)
uvicorn server.app:create_app --factory --reload --port 8000
```

Run the baseline inference script. It supports any OpenAI-compatible API via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | falls back to `OPENAI_API_KEY` | API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Provider endpoint |
| `LLM_MODEL` | `gpt-4o` | Model name |

```bash
# OpenAI (default)
export LLM_API_KEY=sk-...
uv run python baseline.py

# Google Gemini
export LLM_API_KEY=AIza...
export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export LLM_MODEL=gemini-2.0-flash
uv run python baseline.py

# Groq
export LLM_API_KEY=gsk_...
export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_MODEL=llama-3.3-70b-versatile
uv run python baseline.py

# Local Ollama
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_API_KEY=ollama
export LLM_MODEL=llama3
uv run python baseline.py
```

## Deploying (Hugging Face Spaces)
You can deploy using a Docker-based Space:
- Create a new Space on https://huggingface.co/new-space and choose the Docker template
- Link this GitHub repository
- The `server/Dockerfile` will be used to build the container

## Project layout
```
content-moderation-env/
├── data/                  # Deterministic task packs (easy/medium/hard)
│   ├── easy.json
│   ├── medium.json
│   └── hard.json
├── models.py              # Typed Action, Observation, State
├── server/                # FastAPI OpenEnv server
│   ├── __init__.py
│   ├── environment.py     # Core logic + rewards + graders
│   ├── app.py
│   └── Dockerfile
├── baseline.py            # Reproducible inference script
├── openenv.yaml           # OpenEnv metadata
├── pyproject.toml
├── README.md
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