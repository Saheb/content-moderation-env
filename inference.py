"""Baseline inference script for content-moderation-env.

Required environment variables (injected by the hackathon validator):
    HF_TOKEN     — API key for the LLM proxy (mandatory, no default)
    API_BASE_URL — LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME   — Model identifier (default: Qwen/Qwen2.5-72B-Instruct)

Optional:
    ENV_BASE_URL / ENV_CLIENT_BASE_URL — environment server (default: http://localhost:8000)

Examples:
    # Groq
    export HF_TOKEN=gsk_... API_BASE_URL=https://api.groq.com/openai/v1 MODEL_NAME=llama-3.3-70b-versatile
    python inference.py

    # Local Ollama
    export HF_TOKEN=ollama API_BASE_URL=http://localhost:11434/v1 MODEL_NAME=llama3
    python inference.py
"""

import os
import sys
import json
import time
from typing import List, Optional

import openai
from openai import OpenAI, RateLimitError

from openenv import GenericEnvClient
from env_loader import load_environment
from models import ModerationAction

load_environment()

# ── Constants ────────────────────────────────────────────────────────────────

TASKS = ["easy", "medium", "hard", "very_hard"]

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_ENV_BASE_URL = "http://localhost:8000"

SCORE_MIN = 0.01   # validator rejects exactly 0.0
SCORE_MAX = 0.99   # validator rejects exactly 1.0

MAX_RATE_LIMIT_DELAY_S = 300   # bail if retry-after exceeds 5 minutes
GLOBAL_TIMEOUT_S       = 3600  # 60-minute hard cap across all tasks

# Optional per-step delay — off by default, set STEP_DELAY_S=5 to throttle for rate-limited providers
STEP_DELAY_S = float(os.getenv("STEP_DELAY_S", "0"))

MODERATION_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "moderation_action",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "type":      {"type": "string", "enum": ["view_post", "view_thread", "categorize", "lookup_policy", "moderate", "escalate"]},
                "post_id":   {"type": "string"},
                "policy_id": {"type": "string"},
                "category":  {"type": "string"},
                "decision":  {"type": "string", "enum": ["keep", "warn", "remove", "escalate"]},
                "rationale": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5"]},
            },
            "required": ["type"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """You are an AI moderator for VibeNet working through a queue of posts one at a time.

For each post you may take ONE action per turn. Use multiple turns to gather context before deciding.

Available actions:
  view_post      — load the full text of the active post          {"type":"view_post","post_id":"<id>"}
  view_thread    — load the reply thread for context             {"type":"view_thread","post_id":"<id>"}
  lookup_policy  — look up a specific policy                     {"type":"lookup_policy","policy_id":"<P1-P5>"}
  moderate       — issue your final decision (use when ready)    {"type":"moderate","post_id":"<id>","decision":"<keep|warn|remove>","rationale":"<P1-P5>"}
  escalate       — escalate a post for human review             {"type":"escalate","post_id":"<id>"}

Policies:
  P1: Hate speech or targeted harassment → remove
  P2: Spam, scams, or commercial solicitation → remove
  P3: Doxxing or sharing private personal info → remove
  P4: Edgy humor, sarcasm, or strong opinions → keep (optional warn)
  P5: Inciting violence or real-world harm → escalate

Strategy:
- For ambiguous posts, always call view_thread before moderating — thread replies often reveal intent.
- For clear-cut posts (obvious spam, obvious safe venting), you may moderate directly.
- Never repeat a decision already marked as failed."""

# ── Logging ──────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def clamp_score(score: float) -> float:
    """Clamp to (0, 1) exclusive — validator rejects exactly 0.0 or 1.0."""
    return max(SCORE_MIN, min(score, SCORE_MAX))


def _connect_with_retry(client, max_attempts: int = 6) -> None:
    """Connect to the env server with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            client.connect()
            return
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(f"Env server not reachable (attempt {attempt + 1}/{max_attempts}): {e}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)


_RETRIABLE_KEYWORDS = (
    # format issues
    "response_format", "json_object", "json_schema", "invalid schema",
    "json_validate_failed", "validate json", "unknown variant", "enum",
    # tool-call issues
    "tool_use_failed", "tool_choice", "tool called", "failed_generation",
    # generic server-side issues
    "internal error", "unsupported", "not supported",
)

_FORMAT_LEVELS = [
    ("json_schema", MODERATION_JSON_SCHEMA),
    ("json_object", {"type": "json_object"}),
    ("plain_text",  None),
]

# Cache the first (format, tool_choice) combo that succeeds so every
# subsequent call skips straight to it.  Resets naturally on process restart.
_working_combo: Optional[tuple] = None


def _build_kwargs(model: str, messages: list, response_fmt, use_tool_choice: bool) -> dict:
    kwargs = {"model": model, "messages": messages}
    if use_tool_choice:
        kwargs["tool_choice"] = "none"
    if response_fmt is not None:
        kwargs["response_format"] = response_fmt
    return kwargs


def _call_llm(client: OpenAI, model: str, messages: list) -> object:
    """Call the LLM, gracefully degrading when a provider rejects a feature.

    Fallback order (tried once, then the winning combo is cached):
      1. json_schema  + tool_choice="none"   (strictest)
      2. json_schema  (no tool_choice)
      3. json_object  + tool_choice="none"
      4. json_object  (no tool_choice)
      5. plain text   + tool_choice="none"
      6. plain text   (no tool_choice)        (most permissive)

    Each step is only tried when the previous one raised a recognisable
    "not supported" style error.  Unrecognised errors are re-raised immediately.
    """
    global _working_combo

    # Fast path: reuse the combo that already worked
    if _working_combo is not None:
        label, response_fmt, use_tool_choice = _working_combo
        kwargs = _build_kwargs(model, messages, response_fmt, use_tool_choice)
        return client.chat.completions.create(**kwargs)

    # Discovery path: try each combo until one succeeds
    last_exc = None
    for label, response_fmt in _FORMAT_LEVELS:
        for use_tool_choice in (True, False):
            try:
                kwargs = _build_kwargs(model, messages, response_fmt, use_tool_choice)
                result = client.chat.completions.create(**kwargs)
                _working_combo = (label, response_fmt, use_tool_choice)
                tc_tag = "+tool_choice" if use_tool_choice else ""
                print(f"LLM format locked: {label}{tc_tag}", file=sys.stderr)
                return result

            except (openai.UnprocessableEntityError, openai.BadRequestError, openai.InternalServerError) as e:
                last_exc = e
                err_lower = str(e).lower()
                if any(kw in err_lower for kw in _RETRIABLE_KEYWORDS):
                    tc_tag = "+tool_choice" if use_tool_choice else ""
                    print(f"LLM fallback: {label}{tc_tag} failed, trying next… ({e!s:.120})", file=sys.stderr)
                    continue
                raise

    raise last_exc


def _call_llm_with_retry(client: OpenAI, model: str, messages: list) -> object:
    """Wrap _call_llm with exponential backoff for rate limits."""
    for attempt in range(6):
        try:
            return _call_llm(client, model, messages)
        except RateLimitError as e:
            if attempt == 5:
                raise
            headers    = getattr(getattr(e, "response", None), "headers", {})
            retry_ms   = headers.get("retry-after-ms")
            retry_s    = headers.get("retry-after")
            sleep_time = (
                int(retry_ms) / 1000.0 if retry_ms is not None
                else int(retry_s) if retry_s and str(retry_s).isdigit()
                else 2 ** attempt
            )
            if sleep_time > MAX_RATE_LIMIT_DELAY_S:
                raise SystemExit(f"Rate limit retry delay ({sleep_time:.0f}s) exceeds {MAX_RATE_LIMIT_DELAY_S}s — likely quota exhausted.")
            print(f"Rate limited (429). Retrying in {sleep_time:.1f}s...", file=sys.stderr)
            time.sleep(sleep_time)


def _parse_action(raw: str) -> dict:
    """Extract and normalise the first JSON object from a model response."""
    # Handle tool-call wrapped responses (model emitted a function call instead of plain JSON)
    try:
        maybe = json.loads(raw)
        if isinstance(maybe, dict) and "arguments" in maybe and "name" in maybe:
            raw = maybe["arguments"] if isinstance(maybe["arguments"], str) else json.dumps(maybe["arguments"])
    except (json.JSONDecodeError, ValueError):
        pass

    # Find first complete JSON object using brace counting
    start = raw.find("{")
    if start != -1:
        depth, end = 0, -1
        for i in range(start, len(raw)):
            depth += (raw[i] == "{") - (raw[i] == "}")
            if depth == 0:
                end = i
                break
        if end != -1:
            raw = raw[start:end + 1]

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: no parseable JSON in model output: {raw!r} ({e})", file=sys.stderr)
        return {}

    # Unwrap single-key root wrappers (e.g. {"action": {...}} or {"actions": [{...}]})
    if isinstance(data, dict) and len(data) == 1:
        val = next(iter(data.values()))
        if isinstance(val, dict):
            data = val
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            data = val[0]

    # Strip unknown keys and whitespace
    valid_keys = ModerationAction.model_fields.keys()
    clean = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in data.items() if k.strip() in valid_keys}

    if "type" not in clean and "decision" in clean:
        clean["type"] = "moderate"

    return clean

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Per hackathon spec: HF_TOKEN mandatory, API_BASE_URL and MODEL_NAME have defaults.
    HF_TOKEN     = os.getenv("HF_TOKEN")
    API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    MODEL_NAME   = os.getenv("MODEL_NAME",   DEFAULT_MODEL_NAME)

    if HF_TOKEN is None:
        raise SystemExit("HF_TOKEN environment variable is required")

    llm    = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    print(f"Using model={MODEL_NAME} via {API_BASE_URL}", file=sys.stderr)

    env_url = os.getenv("ENV_BASE_URL") or os.getenv("ENV_CLIENT_BASE_URL", DEFAULT_ENV_BASE_URL)
    client  = GenericEnvClient(base_url=env_url).sync()

    global_start  = time.time()
    total_sleep_s = 0.0
    final_summary: dict = {}

    try:
        for task in TASKS:
            rewards:      List[float] = []
            step          = 0
            score         = clamp_score(0.0)  # safe default for failed tasks
            success       = False
            total_reward  = 0.0

            log_start(task=task, env="content-moderation-env", model=MODEL_NAME)
            try:
                _connect_with_retry(client)
                result = client.reset(task_name=task)
                done   = getattr(result, "done", False)

                while not done:
                    if time.time() - global_start > GLOBAL_TIMEOUT_S:
                        print(f"Global timeout reached. Stopping {task} early.", file=sys.stderr)
                        break

                    step += 1
                    obs            = result.observation
                    active_post    = obs.get("active_post_summary")
                    failed_attempts = obs.get("failed_attempts", [])
                    last_result    = obs.get("last_action_result", "")

                    current_post   = obs.get("current_post")
                    thread_context = obs.get("thread_context")

                    prompt = f"Active post: {json.dumps(active_post, indent=2)}"
                    if current_post:
                        prompt += f"\n\nFull post text:\n{json.dumps(current_post, indent=2)}"
                    if thread_context:
                        prompt += f"\n\nThread replies:\n{json.dumps(thread_context, indent=2)}"
                    if last_result:
                        prompt += f"\n\nLast action result: {last_result}"
                    if failed_attempts:
                        all_decisions = ["keep", "warn", "remove", "escalate"]
                        remaining = [d for d in all_decisions if d not in failed_attempts]
                        prompt += f"\n\n⚠️ Failed decisions already tried: {failed_attempts} — do NOT repeat these."
                        if remaining:
                            prompt += f" Remaining options: {remaining}."
                    prompt += "\n\nRespond with a single JSON action object."

                    print(prompt, file=sys.stderr)

                    resp = _call_llm_with_retry(llm, MODEL_NAME, [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ])
                    print(resp, file=sys.stderr)

                    content  = (resp.choices[0].message.content or "")
                    reasoning = getattr(resp.choices[0].message, "reasoning_content", "") or ""
                    clean_data = _parse_action((content + "\n" + reasoning).strip())

                    if not clean_data:
                        print(f"Warning: invalid action structure, using fallback view_post", file=sys.stderr)
                        clean_data = {"type": "view_post", "post_id": "unknown"}

                    try:
                        action = ModerationAction.model_validate(clean_data)
                    except Exception as e:
                        print(f"Warning: action validation failed ({e}), using fallback view_post", file=sys.stderr)
                        action = ModerationAction.model_validate({"type": "view_post", "post_id": "unknown"})

                    try:
                        result    = client.step(action)
                        error_msg = None
                        reward    = result.reward or 0.0
                        done      = result.done
                    except Exception as e:
                        error_msg = str(e)
                        reward    = 0.0
                        done      = True

                    total_reward += reward
                    rewards.append(reward)
                    log_step(step=step, action=json.dumps(clean_data).replace('"', "'"), reward=reward, done=done, error=error_msg)

                    if STEP_DELAY_S > 0 and not done:
                        time.sleep(STEP_DELAY_S)
                        total_sleep_s += STEP_DELAY_S

                score = clamp_score(obs.get("grader_score") or 0.0 if (obs := result.observation) else 0.0)
                print(f"Task {task}: grader_score={score:.2f} total_reward={total_reward:.2f}", file=sys.stderr)
                success = score >= 0.8
                final_summary[task] = {"score": score, "reward": total_reward, "success": success}

            except Exception as e:
                print(f"Task '{task}' failed: {e}", file=sys.stderr)
                final_summary[task] = {"score": score, "reward": total_reward, "success": False}
            finally:
                log_end(success=success, steps=step, score=score, rewards=rewards)

    finally:
        try:
            client.close()
        except Exception:
            pass

    elapsed    = time.time() - global_start
    net_elapsed = elapsed - total_sleep_s
    mins, secs = divmod(net_elapsed, 60)
    print(f"\n{'='*45}\nFINAL RUN SUMMARY\n{'='*45}", file=sys.stderr)
    for t, m in final_summary.items():
        print(f"Task: {t:<8} | Score: {m['score']:.2f} | Reward: {m['reward']:>6.2f} | Success: {m['success']}", file=sys.stderr)
    time_line = f"Time: {int(mins)}m {int(secs)}s"
    if total_sleep_s > 0:
        sm, ss = divmod(total_sleep_s, 60)
        time_line += f" (excl. {int(sm)}m {int(ss)}s throttle delay)"
    print(f"{'-'*45}\n{time_line} | Model: {MODEL_NAME}\n{'='*45}\n", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
