"""Reproducible baseline inference script for content-moderation-env.

Supports any OpenAI-compatible API (OpenAI, Gemini, Groq, Together, Ollama, etc.)
via environment variables:

    HF_TOKEN         — Hugging Face / API key
    API_BASE_URL     — Base URL (default: https://api.openai.com/v1)
    MODEL_NAME       — Model name (default: gpt-4o)

Examples:
    # OpenAI (default)
    export HF_TOKEN=sk-...
    python inference.py

    # Google Gemini
    export HF_TOKEN=AIza...
    export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    export MODEL_NAME=gemini-2.0-flash
    python inference.py

    # Groq
    export HF_TOKEN=gsk_...
    export API_BASE_URL=https://api.groq.com/openai/v1
    export MODEL_NAME=llama-3.3-70b-versatile
    python inference.py

    # Local Ollama
    export API_BASE_URL=http://localhost:11434/v1
    export HF_TOKEN=ollama
    export MODEL_NAME=llama3
    python inference.py
"""

import os
import re
import sys
import json
import time
from typing import List, Optional

from openenv import GenericEnvClient

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)
import openai
from openai import OpenAI, RateLimitError
from models import ModerationAction


def main() -> None:
    """
    Main entry point for running the content moderation agent.
    Iterates through easy, medium, and hard task packs and logs performance.
    """
    # Optional: load environment variables from a .env file if available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("MODEL_NAME", "gpt-4o")

    if not api_key:
        raise SystemExit("Set HF_TOKEN environment variable")

    openai_client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Using model={model} via {base_url}", file=sys.stderr)

    # --- Environment client ---
    env_client_base = os.getenv("ENV_CLIENT_BASE_URL", "http://localhost:8000")
    async_client = GenericEnvClient(base_url=env_client_base)
    client = async_client.sync()
    
    global_start_time = time.time()
    final_summary = {}

    for task in ["easy", "medium", "hard"]:
        client.connect()
        try:
            result = client.reset(task_name=task)
            total_reward = 0.0
            rewards = []
            history = []
            step = 0
            log_start(task=task, env="content-moderation-env", model=model)
            # result is a StepResult where `.observation` is a dict (Generic client)
            done = result.done if hasattr(result, "done") else False
            while not done:
                if time.time() - global_start_time > 3600:
                    print(f"Global 60-minute timeout reached. Terminating {task} task early.", file=sys.stderr)
                    break
                    
                step += 1
                obs = result.observation
                active_post = obs.get("active_post_summary")
                failed_attempts = obs.get("failed_attempts", [])
                
                # Weaker models get extremely confused by Pydantic JSON schemas because they start with "type": "object"
                # Instead of schema draft syntax, give them a simple JSON template to fill out
                schema_str = """{
  "type": "moderate",
  "post_id": "<ID of the active post>",
  "decision": "<keep | warn | remove | escalate>",
  "rationale": "<P1 | P2 | P3 | P4 | P5>"
}"""

                system_prompt = """You are an AI moderator for VibeNet.
Your task is to review the active post and issue a moderation action.
Available Policies:
  P1: Hate speech or targeted harassment → remove
  P2: Spam, scams, or commercial solicitation → remove
  P3: Doxxing or sharing private personal info → remove
  P4: Edgy humor, sarcasm, or strong opinions → keep (optional warn)
  P5: Inciting violence or real-world harm → escalate

To moderate the active pending post, output an action with:
  "type": "moderate"
  "post_id": the id of the active post
  "decision": "keep", "warn", "remove", or "escalate"
  "rationale": the policy ID applied (e.g. "P4")"""

                last_result = obs.get("last_action_result", "")
                
                prompt = f"Observation:\n- Active Post to Moderate: {json.dumps(active_post, indent=2)}"
                if last_result:
                    prompt += f"\n- Environment Feedback: {last_result}"
                
                if failed_attempts:
                    prompt += f"\n  ⚠️ PAST FAILURES: You have already tried these decisions for this post: {failed_attempts}. They were INCORRECT. DO NOT use these decisions again."
                    
                prompt += f"\n\nDecide next moderation action.\nRespond ONLY with a JSON object matching this schema:\n{schema_str}"

                print(prompt, file=sys.stderr)
                
                msg_list = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                resp = None
                max_retry_delay = (
                    300  # 5 minutes — beyond this likely means daily quota exhausted
                )
                # Attempt generation with exponential backoff for rate limits
                for attempt in range(6):
                    try:
                        try:
                            # 1. Primary Attempt: Strict JSON Schema (Best for GPT-4 / Groq)
                            resp = openai_client.chat.completions.create(
                                model=model,
                                messages=msg_list,
                                response_format={
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "moderation_action",
                                        "strict": True,
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "string", "enum": ["moderate"]},
                                                "post_id": {"type": "string"},
                                                "decision": {"type": "string", "enum": ["keep", "warn", "remove", "escalate"]},
                                                "rationale": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5"]}
                                            },
                                            "required": ["type", "post_id", "decision", "rationale"],
                                            "additionalProperties": False
                                        }
                                    }
                                }
                            )
                        except (openai.UnprocessableEntityError, openai.BadRequestError, openai.InternalServerError) as e:
                            err_str = str(e).lower()
                            if "internal error" in err_str or "response_format" in err_str or "json_object" in err_str or "json_schema" in err_str or "unknown variant" in err_str:
                                try:
                                    # Fallback 1: generic json_object
                                    resp = openai_client.chat.completions.create(
                                        model=model,
                                        messages=msg_list,
                                        response_format={"type": "json_object"},
                                    )
                                except (openai.UnprocessableEntityError, openai.BadRequestError, openai.InternalServerError):
                                    # Fallback 2: Plain Text (Worst case)
                                    resp = openai_client.chat.completions.create(
                                        model=model,
                                        messages=msg_list,
                                    )
                            else:
                                raise e
                                
                        break  # Break retry loop on successful generation
                    except RateLimitError as e:
                        if attempt == 5:
                            raise e
                        # Prefer server-supplied retry delay
                        headers = getattr(getattr(e, "response", None), "headers", {})
                        retry_ms = headers.get("retry-after-ms")
                        if retry_ms is not None:
                            sleep_time = int(retry_ms) / 1000.0
                        else:
                            retry_after = headers.get("retry-after")
                            if retry_after is not None and retry_after.isdigit():
                                sleep_time = int(retry_after)
                            else:
                                sleep_time = 2**attempt
                        if sleep_time > max_retry_delay:
                            raise SystemExit(
                                f"Rate limit retry delay ({sleep_time:.0f}s) exceeds {max_retry_delay}s threshold. "
                                f"Likely daily quota exhausted — try again later or switch providers."
                            )
                        print(f"Rate limited (429). Retrying in {sleep_time:.1f}s...", file=sys.stderr)
                        time.sleep(sleep_time)

                print(resp, file=sys.stderr)
                assert resp is not None
                msg = resp.choices[0].message
                content_str = msg.content or ""
                reasoning_str = getattr(msg, "reasoning_content", "") or ""
                raw = (content_str + "\n" + reasoning_str).strip()
                # Extract JSON block to ignore conversational text from chatty models
                # Use a brace-counting parser to grab ONLY the first complete JSON object
                start = raw.find("{")
                if start != -1:
                    brace_count = 0
                    end = -1
                    for i in range(start, len(raw)):
                        if raw[i] == '{':
                            brace_count += 1
                        elif raw[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i
                                break
                    if end != -1:
                        raw = raw[start : end + 1]

                # Pre-clean the JSON strings and filter out hallucinated extra keys
                data = json.loads(raw)

                # Sub-LLMs often hallucinate root wrappers (e.g. {"action": {...}} or {"actions": [{...}]})
                if isinstance(data, dict) and len(data) == 1:
                    val = list(data.values())[0]
                    if isinstance(val, dict):
                        data = val
                    elif (
                        isinstance(val, list)
                        and len(val) > 0
                        and isinstance(val[0], dict)
                    ):
                        data = val[0]

                valid_keys = ModerationAction.model_fields.keys()
                clean_data = {
                    k.strip(): (v.strip() if isinstance(v, str) else v)
                    for k, v in data.items()
                    if k.strip() in valid_keys
                }

                if "type" not in clean_data and "decision" in clean_data:
                    clean_data["type"] = "moderate"

                if not clean_data:
                    print(f"Warning: Model output invalid JSON structure: {raw}", file=sys.stderr)
                    # Fallback to prevent crash, agent will receive negative reward
                    clean_data = {"type": "view_post", "post_id": "unknown"}

                action = ModerationAction.model_validate(clean_data)
                try:
                    new_result = client.step(action)
                    error_msg = None
                except Exception as e:
                    error_msg = str(e)
                    new_result = None
                    
                if new_result:
                    result = new_result
                    reward = result.reward or 0.0
                    done = result.done
                else:
                    reward = 0.0
                    done = True
                    
                total_reward += reward
                rewards.append(reward)
                
                action_str = json.dumps(clean_data).replace('"', "'")
                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

                # Preemptive throttle for free APIs like Groq (stays under 12K TPM / 30 RPM limits)
                # Bypass artificial delay for local endpoints to maximize inference speed
                if not any(lh in base_url for lh in ("localhost", "127.0.0.1", "0.0.0.0")):
                    time.sleep(5)

            # Read grader score from the observation (surfaced as a top-level field)
            final_obs = result.observation
            score = 0.0
            if isinstance(final_obs, dict):
                score = final_obs.get("grader_score") or 0.0

            print(
                f"Task {task}: grader_score = {score:.2f} | total_reward = {total_reward:.2f}", file=sys.stderr
            )
            success = score >= 0.8
            final_summary[task] = {"score": score, "reward": total_reward, "success": success}
            log_end(success=success, steps=step, score=score, rewards=rewards)
        finally:
            client.close()
            
    total_time = time.time() - global_start_time
    mins, secs = divmod(total_time, 60)
    
    print("\n" + "="*45, file=sys.stderr)
    print("FINAL RUN SUMMARY", file=sys.stderr)
    print("="*45, file=sys.stderr)
    for t, m in final_summary.items():
        print(f"Task: {t: <8} | Score: {m['score']:.2f} | Reward: {m['reward']:>6.2f} | Success: {m['success']}", file=sys.stderr)
    print("-" * 45, file=sys.stderr)
    print(f"Total Time Taken: {int(mins)}m {int(secs)}s", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print("="*45 + "\n", file=sys.stderr)


if __name__ == "__main__":
    main()
