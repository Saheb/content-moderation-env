# Copyright (c) Saheb Motiani
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Content Moderation Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import ModerationAction, ModerationObservation
except ImportError:
    from .models import ModerationAction, ModerationObservation


class ContentModerationEnv(EnvClient[ModerationAction, ModerationObservation, State]):
    """
    Client for the Content Moderation Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> from content_moderation_env import ContentModerationEnv, ModerationAction
        >>>
        >>> env = ContentModerationEnv(base_url="http://localhost:8000")
        >>> # Start with easy task
        >>> result = env.reset(task_name="easy")
        >>> print(f"Active post: {result.observation.active_post_summary}")
        >>>
        >>> # View the full post
        >>> result = env.step(ModerationAction(type="view_post", post_id="p1"))
        >>> print(f"Post content: {result.observation.current_post}")
        >>>
        >>> # View thread context if needed
        >>> result = env.step(ModerationAction(type="view_thread", post_id="p1"))
        >>> print(f"Thread: {result.observation.thread_context}")
        >>>
        >>> # Make moderation decision
        >>> result = env.step(ModerationAction(
        ...     type="moderate", 
        ...     post_id="p1", 
        ...     decision="keep",
        ...     rationale="P4"
        ... ))
        >>> print(f"Score: {result.observation.grader_score}")
        >>> print(f"Done: {result.observation.done}")
        >>> env.close()

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ContentModerationEnv.from_docker_image("content-moderation-env:latest")
        >>> try:
        ...     result = client.reset(task_name="medium")
        ...     # Process posts...
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ModerationAction) -> Dict:
        """
        Convert ModerationAction to JSON payload for step message.

        Args:
            action: ModerationAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {"type": action.type}
        
        if action.post_id:
            payload["post_id"] = action.post_id
        if action.decision:
            payload["decision"] = action.decision
        if action.rationale:
            payload["rationale"] = action.rationale
        if action.policy_id:
            payload["policy_id"] = action.policy_id
        if action.category:
            payload["category"] = action.category
            
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[ModerationObservation]:
        """
        Parse server response into StepResult[ModerationObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ModerationObservation
        """
        obs_data = payload.get("observation", {})
        observation = ModerationObservation(
            active_post_summary=obs_data.get("active_post_summary"),
            failed_attempts=obs_data.get("failed_attempts", []),
            current_post=obs_data.get("current_post"),
            thread_context=obs_data.get("thread_context"),
            last_action_result=obs_data.get("last_action_result", ""),
            grader_score=obs_data.get("grader_score"),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
