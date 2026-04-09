---
title: Content Moderation OpenEnv
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: mit
tags:
  - openenv
---

# content-moderation-env

**Submission for Meta PyTorch OpenEnv Hackathon Round 1**
**Scaler School of Technology — March 2026**
**Author:** Saheb Motiani ([@saheb](https://github.com/saheb))
**License:** MIT

A small, deterministic OpenEnv for evaluating AI agents on user-generated content (UGC) moderation. The environment simulates the sequential review workflow a human trust & safety analyst follows: read a post, check the thread for context, apply a policy, and issue a decision — keep, warn, remove, or escalate.

**[▶ Try the interactive demo](https://saheb-content-moderation-env.hf.space)** — play through the environment as a human agent.

**[📊 Live benchmark visualization](https://saheb-content-moderation-env.hf.space/benchmark.html)** — scores and rewards across all tiers.

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


### One-shot baseline (no thread reading)

All actions forced to `moderate` — models never call `view_post` or `view_thread`.

| Model | Provider | Easy (Score / Rwd) | Medium (Score / Rwd) | Hard (Score / Rwd) | Very Hard (Score / Rwd) | Total Time |
|---|---|---|---|---|---|---|
| Mistral-Nemo 12B | Local (Ollama) | 0.40 / — | 0.30 / — | 0.09 / — | — | — |
| DeepSeek-R1-Distill-Qwen 14B | Local (mistral.rs) | **1.00**† / +1.37 | 0.00† / +0.31 (timeout) | 0.00† / +0.00 (timeout) | — | 60m 26s |
| gpt-oss-20b | Cloud (Groq) | **1.00**† / +1.90 | **1.00**† / +3.14 | 0.82 / -0.20 | 0.44 / +2.08 | 10m 36s |
| gemma-4-31b-it | Cloud (Google) | **1.00**† / +2.25 | **1.00**† / +3.46 | 0.66 / -0.93 | — | 7m 34s (one-shot only) |
| Llama 3.3 70B Versatile | Cloud (Groq) | **1.00**† / +1.35 | **1.00**† / +1.17 | 0.82 / +1.64 | 0.44 / +1.37 | 11m 31s |
| gpt-oss-120b | Cloud (Groq) | **1.00**† / +2.25 | **1.00**† / +2.95 | 0.82 / +5.86 | 0.44 / +2.96 | 5m 32s |
| Gemini 3.1 Flash Lite | Cloud (Google) | **1.00**† / +1.90 | **1.00**† / +3.30 | 0.82 / +6.39 | — | 8m 19s |
| Qwen 3 32B | Cloud (Groq) | **1.00**† / +3.15 | **1.00**† / +1.89 | 0.82 / **+8.51** | 0.44 / **+4.36** | 12m 47s |
| NVIDIA Nemotron 3 Super 120B | Cloud (OpenRouter) | **1.00**† / +3.15 | **1.00**† / +2.95 | 0.00† / +2.73 | 0.00† / -0.42 | 16m 5s |

> Hard ceiling: 0.82 (3 thread-critical posts × −0.06). Very hard ceiling: 0.44 (14 thread-critical posts × −0.04).

### Multi-step agent (thread reading enabled)

Models can call `view_post`, `view_thread`, and `lookup_policy` before deciding.

| Model | Provider | Easy (Score / Rwd) | Medium (Score / Rwd) | Hard (Score / Rwd) | Very Hard (Score / Rwd) | Total Time |
|---|---|---|---|---|---|---|
| gemma-4-31b-it | Cloud (Google) | **1.00**† / +3.47 | **1.00**† / +5.71 | **0.99** / +11.33 | **0.88** / +10.37 | 5m 14s |
| gpt-oss-120b | Cloud (Groq) | **1.00**† / +1.68 | **1.00**† / +4.31 | **0.94** / +7.76 | **0.84** / +5.96 | 9m 47s |
| Gemini 3.1 Flash Lite | Cloud (Google) | **1.00**† / +2.49 | 0.90 / -1.29 | **0.94** / +1.70 | **0.84** / +8.06 | 12m 29s |
| Qwen 3 32B | Cloud (Groq) | **1.00**† / +3.35 | **1.00**† / +5.58 | **0.99** / +10.03 | **0.88** / +10.86 | 11m 8s |

> † Raw grader accuracy shown; the validator requires scores strictly within (0, 1), so 1.00 is emitted as **0.99** and 0.00 as **0.01**.

**Key findings:**
- **One-shot baseline:** Hard score converges at 0.82, very hard at 0.44 — both are hard ceilings imposed by skipping thread context, not model capability limits.
- **Multi-step agent:** gemma-4-31b-it scores **0.99 on hard and 0.88 on very hard**, the strongest result in the table. gpt-oss-120b scores 0.94 / 0.84. Both vastly exceed the one-shot baseline by proactively calling `view_thread` on ambiguous posts.
- **The context gap is the story:** The jump from 0.44 → 0.88 on very_hard is entirely explained by thread reading. Models that reason about *when* to gather context vastly outperform those that moderate on surface text alone.

**Score alone is insufficient for evaluating frontier moderation agents.** The environment's reward shaping and thread-context penalty are designed to surface exactly this gap.

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

**Reward deduplication:** Context actions (`view_post`, `view_thread`, `lookup_policy`) only award their reward on the **first call per target** within an episode. Repeat calls return the same data but yield +0.00 and trigger a warning in `last_action_result`.

**Loop detection:** Repeating the same action on the same target 3+ times triggers a **−5.0 penalty**. Behaviour differs by action type:

| Looping action | Consequence |
|---|---|
| `view_post` / `view_thread` / `categorize` on same `post_id` | Post forfeited (`"skipped"` — scores 0), stale content cleared, episode continues with next post |
| `lookup_policy` on same `policy_id` | Penalty applied, model told to stop and moderate — no post forfeited |
| `moderate` with same `(post_id, decision, rationale)` (N=3/4/5 by difficulty) | Post forfeited, episode continues |

The episode never terminates early from a loop — the model is always given a chance to complete remaining posts.

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
export HF_TOKEN=your-key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
python inference.py
```

### Supported providers

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | *(required)* | API key — injected automatically by the hackathon validator |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM provider endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` / `ENV_CLIENT_BASE_URL` | `http://localhost:8000` | Environment server URL |

```bash
# Groq
export HF_TOKEN=gsk_... && export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile && python inference.py

# Google Gemini
export HF_TOKEN=AIza... && export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_NAME=gemini-2.0-flash && python inference.py

# Local Ollama
export HF_TOKEN=ollama && export API_BASE_URL=http://localhost:11434/v1
export MODEL_NAME=llama3 && python inference.py
```

## Deploying to Hugging Face Spaces

**Live Space:** [saheb-content-moderation-env.hf.space](https://saheb-content-moderation-env.hf.space) — interactive demo loads at the root URL.

The server listens on port 8000 (`app_port: 8000` in the README YAML tells HF Spaces to route traffic there):

```bash
docker build -t content-moderation-env .
docker run -p 8000:8000 content-moderation-env
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

**Thread-context inversion as a grader penalty** — Unlike most environments where reading context only unlocks reward signals, here skipping context on ambiguous posts actively reduces the final grader score. This forces agents to develop a genuine strategy for when to read context vs. when post text alone is sufficient. The set of context-critical posts and the per-post penalty are defined in each task's JSON file (`context_critical_posts`), keeping grading logic fully data-driven and reproducible without touching server code.

**Over-censorship is distinctly penalized** — Removing a post that should be kept carries a −0.90 net reward (−0.35 incorrect + −0.55 over-censorship), which is worse than any other single error. This mirrors real-world trust & safety constraints where false positives damage user trust as much as false negatives damage platform safety.

**Reward/score divergence as a research signal** — On the hard task, score converges to 0.82 for most frontier models, but reward ranges from −0.93 to +6.39. This divergence is itself a finding: binary accuracy metrics are insufficient for evaluating agents on nuanced judgment tasks. The environment is designed to demonstrate this property explicitly.

## License

MIT — see [LICENSE](LICENSE).
