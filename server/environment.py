import uuid
import json
import os
import time
from typing import Dict, Any, Optional

from openenv.core.env_server import Environment
from models import ModerationAction, ModerationObservation, ModerationState


class ContentModerationEnvironment(Environment):
    def __init__(self):
        self.episode_id: str | None = None
        self.task_name: str | None = None
        self.task_data: Dict[str, Any] | None = None
        self.moderated: set[str] = set()
        self.decisions: Dict[str, str] = {}  # post_id -> agent's decision (for grader)
        self.viewed_threads: set[str] = set()  # post_ids where view_thread was called
        self.step_count: int = 0

        # Realistic VibeNet policy playbook
        self.policies: Dict[str, str] = {
            "P1": "Hate speech or targeted harassment → remove",
            "P2": "Spam, scams, or commercial solicitation → remove",
            "P3": "Doxxing or sharing private personal info → remove",
            "P4": "Edgy humor, sarcasm, or strong opinions → keep (optional warn)",
            "P5": "Inciting violence or real-world harm → escalate",
        }

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> ModerationObservation:
        task_name = kwargs.get("task_name", "easy")
        self.episode_id = str(uuid.uuid4())
        self.task_name = task_name
        self.moderated.clear()
        self.decisions.clear()
        self.viewed_threads.clear()
        self.step_count = 0
        self.action_history_counts: Dict[str, int] = {}

        # Load deterministic task pack
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{task_name}.json")
        with open(data_path, encoding="utf-8") as f:
            self.task_data = json.load(f)

        return ModerationObservation(
            queue_summary=self._get_queue(),
            last_action_result="Episode started — VibeNet moderation queue loaded",
            metadata={"task": task_name, "total_posts": len(self.task_data["items"]) }
        )

    def step(self, action: ModerationAction, timeout_s: Optional[float] = None, **kwargs) -> ModerationObservation:
        self.step_count += 1
        reward: float = 0.0
        result: str = ""
        
        # Track loop count by post_id
        if action.type == "moderate" and action.post_id and action.decision:
            action_key = f"{action.post_id}_{action.decision}_{action.rationale}"
            self.action_history_counts[action_key] = getattr(self, "action_history_counts", {}).get(action_key, 0) + 1

        if action.type == "view_post":
            reward += 0.08
            result = f"Post {action.post_id} loaded"

        elif action.type == "view_thread":
            reward += 0.12
            if action.post_id:
                self.viewed_threads.add(action.post_id)
            result = f"Thread context for {action.post_id} loaded"

        elif action.type == "categorize":
            reward += 0.15
            result = f"Preliminary category noted for {action.post_id}"

        elif action.type == "moderate":
            if not action.post_id or not action.decision:
                reward -= 0.20
                result = "Invalid moderate action"
            else:
                gt = self.task_data["ground_truth"].get(action.post_id, {})
                correct_label = gt.get("label")

                if action.decision == correct_label:
                    reward += 0.45
                    self.moderated.add(action.post_id)
                    self.decisions[action.post_id] = action.decision
                    result = "Correct moderation decision"
                    # Bonus for policy-linked rationale
                    if action.rationale and gt.get("policy_id") in (action.rationale or ""):
                        reward += 0.18
                else:
                    reward -= 0.35
                    result = "Incorrect decision"

                # Heavy penalty for over-censorship (removing safe content)
                if action.decision == "remove" and correct_label == "keep":
                    reward -= 0.55

        elif action.type == "lookup_policy":
            reward += 0.10
            result = f"Policy {action.policy_id} looked up: {self.policies.get(action.policy_id, 'Unknown')}"

        elif action.type == "escalate":
            if action.post_id and self.task_data["ground_truth"].get(action.post_id, {}).get("label") == "escalate":
                reward += 0.30
                self.moderated.add(action.post_id)
                self.decisions[action.post_id] = "escalate"
                result = "Correctly escalated"
            else:
                reward -= 0.30
                result = "Incorrect escalation"

        else:
            reward -= 0.10
            result = "Unknown action type"

        # Apply loop penalty based on task difficulty level
        limit = {"easy": 3, "medium": 4, "hard": 5}.get(self.task_name, 3)
        loop_exceeded = False
        
        if action.type == "moderate" and action.post_id and action.decision:
            action_key = f"{action.post_id}_{action.decision}_{action.rationale}"
            count = getattr(self, "action_history_counts", {}).get(action_key, 0)
            if count >= limit:
                reward -= 5.0
                result = f"Infinite loop detected: Model repeated the exact same decision for post {action.post_id} {count} times. Heavy penalty applied."
                loop_exceeded = True

        # Episode termination
        done = len(self.moderated) == len(self.task_data["items"]) or self.step_count >= self.task_data["max_steps"] or loop_exceeded

        grader = self._compute_grader_score() if done else None

        obs = ModerationObservation(
            queue_summary=self._get_queue(),
            last_action_result=result,
            grader_score=grader,
            reward=reward,
            done=done,
            metadata={
                "moderated": len(self.moderated),
                "step_count": self.step_count,
            }
        )

        return obs

    def _compute_grader_score(self) -> float:
        """Deterministic 0.0–1.0 grader (25% rubric requirement)."""
        if not self.task_data:
            return 0.0

        gt = self.task_data["ground_truth"]
        total_posts = len(gt)
        correct = 0

        for pid in self.moderated:
            if pid in gt and self.decisions.get(pid) == gt[pid]["label"]:
                correct += 1

        score = correct / total_posts if total_posts > 0 else 0.0

        # Hard task: deduct for context-critical posts where agent skipped thread review.
        # These posts have ambiguous text that requires reading the thread to decide correctly.
        # Skipping view_thread on them is a grader-level penalty (not just missing a reward).
        if self.task_name == "hard":
            THREAD_REQUIRED_POSTS = {"h5", "h12", "h15"}
            for pid in THREAD_REQUIRED_POSTS:
                if pid in self.decisions and pid not in self.viewed_threads:
                    score -= 0.06  # 3 posts × 0.06 = up to 0.18 deduction

        return round(max(0.0, min(1.0, score)), 2)

    def _get_queue(self) -> list[Dict[str, Any]]:
        return [
            {
                "id": pid,
                "status": "moderated" if pid in self.moderated else "pending",
                "preview": self.task_data["items"][pid]["text"][:60] + "..." if len(self.task_data["items"][pid]["text"]) > 60 else self.task_data["items"][pid]["text"]
            }
            for pid in self.task_data["items"]
        ]

    @property
    def state(self) -> ModerationState:
        return ModerationState(
            episode_id=self.episode_id or "",
            step_count=self.step_count,
            moderated_count=len(self.moderated),
            task_name=self.task_name or "easy"
        )
