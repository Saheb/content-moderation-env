import sys

with open("inference.py", "r") as f:
    text = f.read()

# 1. Imports and log functions
text = text.replace(
    "import os\nimport re\nimport json\nimport time\n\nfrom openenv import GenericEnvClient",
    """import os
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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)"""
)

# 2. Redirect stdout
text = text.replace('print(f"Using model={model} via {base_url}")', 'print(f"Using model={model} via {base_url}", file=sys.stderr)')
text = text.replace('print(prompt)', 'print(prompt, file=sys.stderr)')
text = text.replace('print(resp)', 'print(resp, file=sys.stderr)')
text = text.replace('print(f"Rate limited (429). Retrying in {sleep_time:.1f}s...")', 'print(f"Rate limited (429). Retrying in {sleep_time:.1f}s...", file=sys.stderr)')
text = text.replace('print(f"Warning: Model output invalid JSON structure: {raw}")', 'print(f"Warning: Model output invalid JSON structure: {raw}", file=sys.stderr)')
text = text.replace('print(\n                f"Task {task}: grader_score = {score:.2f} | total_reward = {total_reward:.2f}"\n            )', 'print(\n                f"Task {task}: grader_score = {score:.2f} | total_reward = {total_reward:.2f}", file=sys.stderr\n            )')

# 3. Add Step logic
text = text.replace(
    """    for task in ["easy", "medium", "hard"]:
        client.connect()
        try:
            result = client.reset(task_name=task)
            total_reward = 0.0
            # result is a StepResult where `.observation` is a dict (Generic client)
            while not result.done:
                obs = result.observation""",
    """    for task in ["easy", "medium", "hard"]:
        client.connect()
        try:
            result = client.reset(task_name=task)
            total_reward = 0.0
            rewards = []
            step = 0
            log_start(task=task, env="content-moderation-env", model=model)
            # result is a StepResult where `.observation` is a dict (Generic client)
            while not result.done:
                step += 1
                obs = result.observation"""
)

# 4. Action logic
text = text.replace(
    """                action = ModerationAction.model_validate(clean_data)
                result = client.step(action)
                total_reward += result.reward or 0.0

                # Preemptive throttle for free APIs like Groq (stays under 12K TPM / 30 RPM limits)
                time.sleep(5)""",
    """                action = ModerationAction.model_validate(clean_data)
                
                try:
                    result = client.step(action)
                    error_msg = None
                except Exception as e:
                    error_msg = str(e)
                    result = None
                    
                if result:
                    reward = result.reward or 0.0
                    done = result.done
                else:
                    reward = 0.0
                    done = True
                    
                total_reward += reward
                rewards.append(reward)
                
                action_str = json.dumps(clean_data).replace('"', "'") # Avoid string problems
                
                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

                # Preemptive throttle for free APIs like Groq (stays under 12K TPM / 30 RPM limits)
                time.sleep(5)"""
)

# 5. End logic
text = text.replace(
    """            if isinstance(final_obs, dict):
                score = final_obs.get("grader_score") or 0.0

            print(
                f"Task {task}: grader_score = {score:.2f} | total_reward = {total_reward:.2f}"
            )
        finally:""",
    """            if isinstance(final_obs, dict):
                score = final_obs.get("grader_score") or 0.0

            print(
                f"Task {task}: grader_score = {score:.2f} | total_reward = {total_reward:.2f}", file=sys.stderr
            )
            success = score >= 0.8
            log_end(success=success, steps=step, score=score, rewards=rewards)
        finally:"""
)

with open("inference.py", "w") as f:
    f.write(text)

print("Patching done!")

