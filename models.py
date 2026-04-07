
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional

# Import core typed bases from the OpenEnv runtime package
from openenv.core.env_server import Action as BaseAction, Observation as BaseObservation, State

class ModerationAction(BaseAction):
    """Action schema for moderation decisions and context lookups."""
    type: Literal["view_post", "view_thread", "categorize", "moderate", "lookup_policy", "escalate"] = Field(description="The type of action to perform")
    post_id: Optional[str] = Field(None, description="The ID of the post being acted upon")
    decision: Optional[Literal["keep", "warn", "remove", "escalate"]] = Field(None, description="The moderation decision (for type='moderate')")
    rationale: Optional[str] = Field(None, description="The policy ID (e.g. P1) supporting the decision")
    policy_id: Optional[str] = Field(None, description="The policy ID to look up (for type='lookup_policy')")
    category: Optional[str] = Field(None, description="The preliminary category (for type='categorize')")

class ModerationObservation(BaseObservation):
    """Observation schema returned by the ContentModerationEnvironment."""
    active_post_summary: Optional[Dict[str, str]] = Field(None, description="A preview of the current active post")
    failed_attempts: list[str] = Field(default_factory=list, description="List of decisions previously tried on this post and marked incorrect")
    current_post: Optional[Dict[str, Any]] = Field(None, description="Full details of the active post if loaded")
    thread_context: Optional[list[Dict[str, Any]]] = Field(None, description="Thread messages if loaded")
    last_action_result: str = Field(..., description="A status message describing the result of the last action")
    grader_score: Optional[float] = Field(None, description="Final score, only provided when the episode is done")
    reward: Optional[float] = Field(None, description="Per-step reward from the last action")
    done: Optional[bool] = Field(None, description="Whether the episode has ended")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional environment metadata")

class ModerationState(State):
    episode_id: str
    step_count: int
    moderated_count: int
    task_name: str