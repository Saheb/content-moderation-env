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
import json
import time

from openenv import GenericEnvClient
from openai import OpenAI, RateLimitError
from models import ModerationAction

def main() -> None:
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("MODEL_NAME", "gpt-4o")

    if not api_key:
        raise SystemExit("Set HF_TOKEN environment variable")

    openai_client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Using model={model} via {base_url}")

    # --- Environment client ---
    async_client = GenericEnvClient(base_url="https://saheb-content-moderation-env.hf.space")
    client = async_client.sync()

    for task in ["easy", "medium", "hard"]:
        client.connect()
        try:
            result = client.reset(task_name=task)
            total_reward = 0.0
            # result is a StepResult where `.observation` is a dict (Generic client)
            while not result.done:
                obs = result.observation
                queue = obs.get("queue_summary")
                # Weaker models get extremely confused by Pydantic JSON schemas because they start with "type": "object"
                # Instead of schema draft syntax, give them a simple JSON template to fill out
                schema_str = """{
  "type": "moderate",
  "post_id": "<ID of the post from the queue>",
  "decision": "<keep | warn | remove | escalate>",
  "rationale": "<P1 | P2 | P3 | P4 | P5>"
}"""
                
                system_prompt = """You are an AI moderator for VibeNet.
Your task is to review the queue and issue moderation actions.
Available Policies:
  P1: Hate speech or targeted harassment → remove
  P2: Spam, scams, or commercial solicitation → remove
  P3: Doxxing or sharing private personal info → remove
  P4: Edgy humor, sarcasm, or strong opinions → keep (optional warn)
  P5: Inciting violence or real-world harm → escalate

To moderate a pending post from the queue, output an action with:
  "type": "moderate"
  "post_id": the id of the post
  "decision": "keep", "warn", "remove", or "escalate"
  "rationale": the policy ID applied (e.g. "P4")"""
                
                prompt = f"Observation:\n- Queue: {json.dumps(queue, indent=2)}\n\nDecide next moderation action.\nRespond ONLY with a JSON object matching this schema:\n{schema_str}"
                
                for attempt in range(6):
                    try:
                        resp = openai_client.chat.completions.create(
                            model=model,
                            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                            response_format={"type": "json_object"}
                        )
                        break  # Success
                    except RateLimitError as e:
                        if attempt == 5:
                            raise e
                        sleep_time = 2 ** attempt
                        print(f"Rate limited (429). Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                
                raw = resp.choices[0].message.content.strip()
                # Extract JSON block to ignore conversational text from chatty models
                start = raw.find('{')
                end = raw.rfind('}')
                if start != -1 and end != -1 and end > start:
                    raw = raw[start:end+1]
                
                # Pre-clean the JSON strings and filter out hallucinated extra keys
                data = json.loads(raw)
                
                # Sub-LLMs often hallucinate root wrappers (e.g. {"action": {...}} or {"actions": [{...}]})
                if isinstance(data, dict) and len(data) == 1:
                    val = list(data.values())[0]
                    if isinstance(val, dict):
                        data = val
                    elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
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
                    print(f"Warning: Model output invalid JSON structure: {raw}")
                    # Fallback to prevent crash, agent will receive negative reward
                    clean_data = {"type": "view_post", "post_id": "unknown"}
                    
                action = ModerationAction.model_validate(clean_data)
                result = client.step(action)
                total_reward += result.reward or 0.0
                
                # Preemptive throttle for free APIs like Groq (limits hitting the 30 RPM ceiling)
                time.sleep(1.5)

            # Read grader score from the observation (surfaced as a top-level field)
            final_obs = result.observation
            score = 0.0
            if isinstance(final_obs, dict):
                score = final_obs.get("grader_score") or 0.0

            print(f"Task {task}: grader_score = {score:.2f} | total_reward = {total_reward:.2f}")
        finally:
            client.close()

if __name__ == "__main__":
    main()