import os
import uuid
import logging

import uvicorn
from fastapi import HTTPException
from fastapi.responses import HTMLResponse

from openenv.core.env_server import create_fastapi_app
from env_loader import load_environment
from .environment import ContentModerationEnvironment
from models import ModerationAction, ModerationObservation

logger = logging.getLogger(__name__)

_loaded_env_file = load_environment()
if _loaded_env_file:
    logger.info("Loaded environment variables from %s", _loaded_env_file)

# Demo session store — keyed by session UUID, capped at 50 concurrent sessions
_demo_sessions: dict = {}
_MAX_DEMO_SESSIONS = 50


def create_app():
    """Factory function for creating the FastAPI app instance."""
    logger.info("Creating FastAPI app...")
    app = create_fastapi_app(
        lambda: ContentModerationEnvironment(),
        action_cls=ModerationAction,
        observation_cls=ModerationObservation,
    )

    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    def root():
        """Root endpoint serving the interactive demo UI."""
        demo_html_path = os.path.join(os.path.dirname(__file__), "..", "demo.html")
        with open(demo_html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    @app.get("/health", tags=["Health"])
    def health():
        """Standard health check endpoint for monitoring."""
        return {"status": "healthy"}

    logger.info("FastAPI app created successfully")
    return app


def main():
    """Create and run the FastAPI app for local/dev execution."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()

# Module-level app for runners that expect `server.app:app` (no --factory)
app = create_app()


@app.get("/ready", tags=["Health"])
def ready():
    """Readiness endpoint often used by cloud platforms for startup probes."""
    return {"status": "ready"}


# ── Demo endpoints ────────────────────────────────────────────────────────────

@app.get("/demo", response_class=HTMLResponse, tags=["Demo"])
def demo_page():
    """Serve the interactive human-playable demo UI."""
    demo_html_path = os.path.join(os.path.dirname(__file__), "..", "demo.html")
    with open(demo_html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/benchmark.html", response_class=HTMLResponse, tags=["UI"])
def benchmark_page():
    """Serve the interactive benchmark visualization UI."""
    benchmark_html_path = os.path.join(os.path.dirname(__file__), "..", "benchmark.html")
    with open(benchmark_html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/demo/reset", tags=["Demo"])
def demo_reset(task: str = "easy"):
    """Start a new demo session. Returns session_id and initial observation."""
    if task not in ("easy", "medium", "hard", "very_hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    # Evict oldest session if at capacity
    if len(_demo_sessions) >= _MAX_DEMO_SESSIONS:
        oldest_key = next(iter(_demo_sessions))
        del _demo_sessions[oldest_key]

    session_id = str(uuid.uuid4())
    env = ContentModerationEnvironment()
    obs = env.reset(task_name=task)
    _demo_sessions[session_id] = env

    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/demo/step/{session_id}", tags=["Demo"])
def demo_step(session_id: str, action: ModerationAction):
    """Submit one action in an existing demo session."""
    env = _demo_sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found or expired. Start a new session via /demo/reset.")

    obs = env.step(action)

    if obs.done:
        del _demo_sessions[session_id]

    return {"observation": obs.model_dump()}
