
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional

# Import core typed bases from the OpenEnv runtime package
from openenv.core.env_server import Action as BaseAction, Observation as BaseObservation, State

class ModerationAction(BaseAction):
    type: Literal["view_post", "view_thread", "categorize", "moderate", "lookup_policy", "escalate"]
    post_id: Optional[str] = None
    decision: Optional[Literal["keep", "warn", "remove", "escalate"]] = None
    rationale: Optional[str] = None
    policy_id: Optional[str] = None
    category: Optional[str] = None

class ModerationObservation(BaseObservation):
    queue_summary: list[Dict[str, Any]]
    current_post: Optional[Dict[str, Any]] = None
    thread_context: Optional[list[Dict[str, Any]]] = None
    last_action_result: str
    grader_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ModerationState(State):
    episode_id: str
    step_count: int
    moderated_count: int
    task_name: str