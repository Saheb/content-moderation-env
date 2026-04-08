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
from models import ModerationAction

# ── Constants ────────────────────────────────────────────────────────────────

TASKS = ["easy", "medium", "hard", "very_hard"]

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_ENV_BASE_URL = "http://localhost:8000"

SCORE_MIN = 0.01   # validator rejects exactly 0.0
SCORE_MAX = 0.99   # validator rejects exactly 1.0

MAX_RATE_LIMIT_DELAY_S = 300   # bail if retry-after exceeds 5 minutes
GLOBAL_TIMEOUT_S       = 3600  # 60-minute hard cap across all tasks

MODERATION_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "moderation_action",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "type":      {"type": "string", "enum": ["moderate"]},
                "post_id":   {"type": "string"},
                "decision":  {"type": "string", "enum": ["keep", "warn", "remove", "escalate"]},
                "rationale": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5"]},
            },
            "required": ["type", "post_id", "decision", "rationale"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """You are an AI moderator for VibeNet.
Your task is to review the active post and issue a moderation action.
Available Policies:
  P1: Hate speech or targeted harassment → remove
  P2: Spam, scams, or commercial solicitation → remove
  P3: Doxxing or sharing private personal info → remove
  P4: Edgy humor, sarcasm, or strong opinions → keep (optional warn)
  P5: Inciting violence or real-world harm → escalate

Output an action with:
  "type": "moderate"
  "post_id": the id of the active post
  "decision": "keep", "warn", "remove", or "escalate"
  "rationale": the policy ID applied (e.g. "P4")"""

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


def _call_llm(client: OpenAI, model: str, messages: list) -> object:
    """Call the LLM with structured-output fallbacks: json_schema → json_object → plain text."""
    format_fallbacks = [
        MODERATION_JSON_SCHEMA,
        {"type": "json_object"},
        None,  # plain text
    ]
    for fmt in format_fallbacks:
        try:
            kwargs = {"model": model, "messages": messages}
            if fmt:
                kwargs["response_format"] = fmt
            return client.chat.completions.create(**kwargs)
        except (openai.UnprocessableEntityError, openai.BadRequestError, openai.InternalServerError) as e:
            if fmt is None:
                raise  # already on plain text, nothing left to try
            err = str(e).lower()
            if any(kw in err for kw in ("response_format", "json_object", "json_schema", "unknown variant", "internal error")):
                continue  # try next format
            raise


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

    global_start = time.time()
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

                    prompt = f"Observation:\n- Active Post to Moderate: {json.dumps(active_post, indent=2)}"
                    if last_result:
                        prompt += f"\n- Environment Feedback: {last_result}"
                    if failed_attempts:
                        prompt += f"\n  ⚠️ PAST FAILURES: {failed_attempts} — do NOT repeat these decisions."
                    prompt += '\n\nRespond ONLY with a JSON object:\n{"type":"moderate","post_id":"<id>","decision":"<keep|warn|remove|escalate>","rationale":"<P1-P5>"}'

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

    elapsed = time.time() - global_start
    mins, secs = divmod(elapsed, 60)
    print(f"\n{'='*45}\nFINAL RUN SUMMARY\n{'='*45}", file=sys.stderr)
    for t, m in final_summary.items():
        print(f"Task: {t:<8} | Score: {m['score']:.2f} | Reward: {m['reward']:>6.2f} | Success: {m['success']}", file=sys.stderr)
    print(f"{'-'*45}\nTime: {int(mins)}m {int(secs)}s | Model: {MODEL_NAME}\n{'='*45}\n", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
