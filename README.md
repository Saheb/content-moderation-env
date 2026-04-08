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
**Author:** Saheb Motiani ([@saheb](https://github.com/saheb))
**License:** MIT

A small, deterministic OpenEnv for evaluating AI agents on user-generated content (UGC) moderation. The environment simulates the sequential review workflow a human trust & safety analyst follows: read a post, check the thread for context, apply a policy, and issue a decision — keep, warn, remove, or escalate.

---

## Why content moderation?

Content moderation is one of the few domains where:

1. **Ambiguity is the point.** A post saying "we'll take matters into our own hands" is either protected political speech or a call to violence depending entirely on the thread replies. Getting that right requires multi-step context gathering, not surface-level pattern matching.
2. **Over- and under-enforcement both have costs.** Removing safe content has a real signal (false-positive penalty) distinct from failing to remove harmful content. Most benchmarks only penalize one direction.
3. **Policy reasoning is testable.** There are five concrete, enumerated policies. A correct decision that also cites the right policy earns a bonus — this separates models that reason from models that guess correctly.
4. **Frontier models converge on score but diverge on reward.** On the hard task, most large cloud models score 0.82 — but their cumulative rewards range from −0.93 to +6.39. This means reward shaping is doing work that accuracy cannot.

This makes it a useful benchmark environment for evaluating **policy-grounded reasoning** in RL agents and LLMs.

---

## Features

- Four deterministic task packs (easy → medium → hard → very_hard) with adversarially designed post content
- Deterministic graders with scores in [0.0, 1.0]
- Dense reward shaping: partial progress, policy-linking bonus, over-censorship penalty, loop detection
- Thread-context awareness: skipping `view_thread` on context-critical posts incurs grader-level deductions (hard: 3 posts; very_hard: 14 posts)
- Typed Pydantic models for all Action, Observation, and State types
- Reproducible inference script supporting any OpenAI-compatible provider
- Server deployable locally or as a containerized HF Space

---

## Benchmark Results

> **Interactive visualization:** [Live Benchmark UI](https://saheb-content-moderation-env.hf.space/benchmark.html) — bar charts for scores and rewards across all tiers.

| Model | Provider | Easy (Score / Rwd) | Medium (Score / Rwd) | Hard (Score / Rwd) | Very Hard (Score / Rwd) | Total Time |
|---|---|---|---|---|---|---|
| Mistral-Nemo 12B (zero-shot) | Local (Ollama) | 0.40 / — | 0.30 / — | 0.09 / — | — | — |
| DeepSeek-R1-Distill-Qwen 14B | Local (mistral.rs) | **1.00** / +1.37 | 0.00 / +0.31 (timeout) | 0.00 / +0.00 (timeout) | — | 60m 26s |
| gpt-oss-20b | Cloud (Groq) | **1.00** / +3.15 | **1.00** / +2.57 | **0.82** / +4.99 | 0.44 / +1.38 | 5m 8s |
| gemma-4-31b-it | Cloud (Groq/Google) | **1.00** / +2.25 | **1.00** / +3.46 | 0.66 / -0.93 | — | 7m 34s |
| Llama 3.3 70B Versatile | Cloud (Groq) | **1.00** / +1.35 | **1.00** / +1.17 | **0.82** / +1.64 | 0.44 / +1.37 | 11m 31s |
| gpt-oss-120b | Cloud (Groq) | **1.00** / +2.25 | **1.00** / +2.95 | **0.82** / +5.86 | 0.44 / +2.96 | 5m 32s |
| Gemini 3.1 Flash Lite | Cloud (Google) | **1.00** / +1.90 | **1.00** / +3.30 | **0.82** / +6.39 | — | 8m 19s |
| Qwen 3 32B | Cloud (Groq) | **1.00** / +3.15 | **1.00** / +1.89 | **0.82** / **+8.51** | 0.44 / **+4.36** | 12m 47s |
| NVIDIA Nemotron 3 Super 120B | Cloud (OpenRouter) | **1.00** / +3.15 | **1.00** / +2.95 | 0.00 / +2.73 | 0.00 / -0.42 | 16m 5s |

**Key findings:**
- Easy/medium: ceiling effect — all capable cloud models score 1.00
- Hard: score converges at 0.82; reward ranges **−0.93 → +8.51**, separating models that reason efficiently from those that don't
- Very hard: capable cloud models score exactly **0.44** — the theoretical maximum without reading thread context (14 thread-critical posts × 0.04 penalty = 0.56 deduction from a perfect 1.00). No model reads threads proactively. Reward ranges **−0.42 → +4.36**, showing some differentiation in how models handle the non-thread posts.

**Score alone is insufficient for evaluating frontier moderation agents.** The environment's reward shaping — and the very_hard tier's thread-context penalty — are what provide meaningful signal beyond binary accuracy.

---

## Tasks

Each task is a deterministic pack of posts the agent must moderate sequentially. The grader computes accuracy against ground truth and applies thread-context penalties on harder tiers.

### Easy — `data/easy.json`
**5 posts · max 20 steps · no thread context required**

Straightforward, unambiguous cases. The primary challenge is correctly distinguishing angry-but-legitimate user venting (keep) from explicit spam (remove). No thread reading is needed; post text alone is sufficient.

| Post | Content | Label | Policy |
|---|---|---|---|
| p1 | Frustrated product review — strong language, no harassment | keep | P4 |
| p2 | Community Discord invite — genuinely free, no solicitation | keep | P4 |
| p3 | Positive community message | keep | P4 |
| p4 | Explicit follower-selling spam | remove | P2 |
| p5 | Angry feedback about a platform update | keep | P4 |

Expected score range: **0.80–1.00** for capable models. A model that over-censors frustrated users will score poorly despite appearing "safe."

---

### Medium — `data/medium.json`
**10 posts · max 50 steps · no mandatory thread reading**

Introduces all five policies and genuinely ambiguous cases across them. Highlights include:
- Dogwhistle hate speech using coded emoji (looks like political commentary)
- A doxxing post framed as a safety request ("help me block my harasser")
- A journalist documenting extremist recruitment verbatim — escalate, not remove
- A whistleblower post with a buried regulatory report — escalate for review
- A "just kidding" violence post with thread replies calling for reports

Models must discriminate between P1/P3 (remove), P4 (keep), and P5 (escalate) across ambiguous framings.

Expected score range: **0.70–1.00** for capable models. Zero-shot baselines and weaker models begin to struggle here.

---

### Hard — `data/hard.json`
**18 posts · max 100 steps · 3 thread-critical posts**

Adversarially designed for genuine policy reasoning. Includes:
- Coded language that is ambiguous without thread context (h5, h12, h15)
- Satire-claimed content that was clearly harmful before reporting
- Posts that are safe despite alarming surface text
- Content spanning all five policies, with ground truth designed to foil surface-level heuristics

**Grader penalty:** Posts h5, h12, and h15 each deduct 0.06 from the final score if the agent issued a decision without first calling `view_thread`. Maximum achievable score without thread reading: **0.82**.

Expected score range: **0.66–0.82** for cloud models. No model achieves 1.00 by design.

---

### Very Hard — `data/very_hard.json`
**20 posts · max 150 steps · 14 thread-critical posts**

Extended adversarial tier. The post text alone is systematically misleading — the thread replies are the signal. Includes:

- **Context inversion:** Safety reports and block lists that are actually harassment tools (vh2, vh3)
- **Coordinated coding:** Posts that are go-signals for off-platform campaigns, undetectable without cross-post pattern awareness surfaced in replies (vh12)
- **Retroactive satire claims:** Content framed as satire only after moderator reporting began (vh9)
- **Dual-interpretation posts:** Posts where replies provide genuinely contradictory signals requiring judgment (vh8, vh14)

**Grader penalty:** 0.04 deducted per thread-critical post decided without `view_thread`, across 14 qualifying posts (max −0.56). A model that ignores thread context entirely scores ≤ 0.44.

---

## Action Space

All actions are defined in `models.py` as `ModerationAction`.

| Action type | Required fields | Optional fields | Reward |
|---|---|---|---|
| `view_post` | `post_id` | — | +0.08 |
| `view_thread` | `post_id` | — | +0.12 (also tracks thread-read for grader) |
| `categorize` | `post_id` | `category` | +0.15 |
| `lookup_policy` | `policy_id` | — | +0.10 |
| `moderate` | `post_id`, `decision` | `rationale` | see below |
| `escalate` | `post_id` | — | +0.30 correct / −0.30 incorrect |

**`moderate` reward breakdown:**

| Outcome | Reward |
|---|---|
| Correct decision | +0.45 |
| + Correct `rationale` policy ID | +0.18 bonus |
| Incorrect decision | −0.35 |
| Incorrect + removing safe content (`keep` post) | −0.35 − 0.55 = **−0.90** |

**Loop detection:** If the same `(post_id, decision, rationale)` tuple is repeated ≥ N times (N=3 easy, 4 medium, 5 hard/very_hard), a −5.0 penalty is applied and the episode terminates.

---

## Observation Space

All observations are defined in `models.py` as `ModerationObservation`.

| Field | Type | Description |
|---|---|---|
| `active_post_summary` | `{id, preview}` | ID and first 60 characters of the next unmoderated post |
| `failed_attempts` | `list[str]` | Decisions already tried on the active post and marked incorrect |
| `current_post` | `dict` | Full post content (populated after `view_post`) |
| `thread_context` | `list[dict]` | Thread replies (populated after `view_thread`) |
| `last_action_result` | `str` | Human-readable result of the last action |
| `reward` | `float` | Per-step reward from the last action |
| `done` | `bool` | Whether the episode has ended |
| `grader_score` | `float \| null` | Final accuracy score — only set when `done=True` |
| `metadata` | `dict` | `{moderated: int, step_count: int}` |

---

## State

`ModerationState` (returned by the `state` property):

| Field | Type |
|---|---|
| `episode_id` | `str` (UUID) |
| `task_name` | `str` |
| `step_count` | `int` |
| `moderated_count` | `int` |

---

## Requirements
- Python >= 3.10
- Key dependencies: `openenv-core`, `pydantic`, `fastapi`, `uvicorn`, `python-dotenv`
- Inference only: `openai>=1.0`

## Quick Setup

### Option A — `uv` (recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/saheb/content-moderation-env.git
cd content-moderation-env

uv venv && source .venv/bin/activate
uv pip install -e .

# Validate OpenEnv metadata
openenv validate
```

### Option B — traditional venv
```bash
git clone https://github.com/saheb/content-moderation-env.git
cd content-moderation-env

python -m venv venv && source venv/bin/activate
pip install -e .
openenv validate
```

## Run locally

```bash
# Start the environment server
uvicorn server.app:app --reload --port 8000
```

```bash
# Run the baseline inference script
export API_KEY=your-key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
python inference.py
```

### Supported providers

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | preferred | API key, required by the hackathon validator |
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM provider endpoint, required by the hackathon validator |
| `HF_TOKEN` | local fallback | Optional API key fallback for local development |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `ENV_CLIENT_BASE_URL` | `http://localhost:8000` | Environment server URL |

`inference.py` reads configuration from process environment variables by default. To opt into loading a local `.env` for local development, set `LOAD_DOTENV=1` before running it.

```bash
# Groq
export API_KEY=gsk_... && export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile && python inference.py

# Google Gemini
export API_KEY=AIza... && export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_NAME=gemini-2.0-flash && python inference.py

# Local Ollama
export API_KEY=ollama && export API_BASE_URL=http://localhost:11434/v1
export MODEL_NAME=llama3 && python inference.py
```

## Deploying to Hugging Face Spaces

**Live Space:** [huggingface.co/spaces/saheb/content-moderation-env](https://huggingface.co/spaces/saheb/content-moderation-env)
**Interactive demo:** [huggingface.co/spaces/saheb/content-moderation-env/demo](https://huggingface.co/spaces/saheb/content-moderation-env/demo) — play through the environment as a human agent.

The root `Dockerfile` targets port 7860 (HF Spaces default):

```bash
docker build -t content-moderation-env .
docker run -p 7860:7860 content-moderation-env
```

To deploy your own: create a Docker-based Space on [huggingface.co/new-space](https://huggingface.co/new-space) and link this repository.

## Project layout

```
content-moderation-env/
├── data/                  # Deterministic task packs
│   ├── easy.json          #  5 posts, max 20 steps
│   ├── medium.json        # 10 posts, max 50 steps
│   ├── hard.json          # 18 posts, max 100 steps, 3 thread-critical
│   └── very_hard.json     # 20 posts, max 150 steps, 14 thread-critical
├── server/
│   ├── environment.py     # ContentModerationEnvironment — step/reset/grader/rewards
│   ├── app.py             # FastAPI app + health endpoints
│   └── __init__.py
├── models.py              # ModerationAction, ModerationObservation, ModerationState
├── inference.py           # Baseline inference script (OpenAI-compatible)
├── benchmark.html         # Interactive benchmark visualization
├── openenv.yaml           # OpenEnv metadata
├── Dockerfile             # HF Spaces container
├── pyproject.toml
└── README.md
```

## Novel mechanics

**Thread-context inversion as a grader penalty** — Unlike most environments where reading context only unlocks reward signals, here skipping context on ambiguous posts actively reduces the final grader score. This forces agents to develop a genuine strategy for when to read context vs. when post text alone is sufficient.

**Over-censorship is distinctly penalized** — Removing a post that should be kept carries a −0.90 net reward (−0.35 incorrect + −0.55 over-censorship), which is worse than any other single error. This mirrors real-world trust & safety constraints where false positives damage user trust as much as false negatives damage platform safety.

**Reward/score divergence as a research signal** — On the hard task, score converges to 0.82 for most frontier models, but reward ranges from −0.93 to +6.39. This divergence is itself a finding: binary accuracy metrics are insufficient for evaluating agents on nuanced judgment tasks. The environment is designed to demonstrate this property explicitly.

## License

MIT — see [LICENSE](LICENSE).
