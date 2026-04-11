import os
import uuid
import logging
import sys
from typing import Optional

import uvicorn
from fastapi import HTTPException
from fastapi.responses import HTMLResponse

from openenv.core.env_server import create_fastapi_app
from env_loader import load_environment
from .environment import ContentModerationEnvironment
from models import ModerationAction, ModerationObservation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables safely
_loaded_env_file: Optional[str] = None
try:
    _loaded_env_file = load_environment()
    if _loaded_env_file:
        logger.info("Loaded environment variables from %s", _loaded_env_file)
except Exception as e:
    logger.warning("Failed to load environment variables: %s", e)

# Demo session store — keyed by session UUID, capped at 50 concurrent sessions
_demo_sessions: dict = {}
_MAX_DEMO_SESSIONS = 50


def create_app():
    """Factory function for creating the FastAPI app instance."""
    logger.info("Creating FastAPI app...")
    
    try:
        app = create_fastapi_app(
            lambda: ContentModerationEnvironment(),
            action_cls=ModerationAction,
            observation_cls=ModerationObservation,
        )
        logger.info("OpenEnv app created successfully")
    except Exception as e:
        logger.error("Failed to create OpenEnv app: %s", e)
        raise

    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    def root():
        """Root endpoint serving the interactive demo UI."""
        try:
            demo_html_path = os.path.join(os.path.dirname(__file__), "..", "demo.html")
            with open(demo_html_path, encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        except FileNotFoundError:
            logger.warning("demo.html not found, serving simple response")
            return HTMLResponse(content="<h1>Content Moderation Environment</h1><p>Interactive demo not available</p>")
        except Exception as e:
            logger.error("Error serving demo: %s", e)
            return HTMLResponse(content="<h1>Content Moderation Environment</h1><p>Error loading demo</p>", status_code=500)

    @app.get("/health", tags=["Health"])
    def health():
        """Standard health check endpoint for monitoring."""
        try:
            # Test environment creation to ensure it's working
            test_env = ContentModerationEnvironment()
            test_obs = test_env.reset(task_name="easy")
            return {
                "status": "healthy",
                "service": "content-moderation-env",
                "environment_test": "passed"
            }
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {
                "status": "unhealthy",
                "service": "content-moderation-env", 
                "error": str(e)
            }, 500

    @app.get("/ready", tags=["Health"])
    def ready():
        """Readiness endpoint often used by cloud platforms for startup probes."""
        return {"status": "ready"}

    logger.info("FastAPI app created successfully")
    return app


def main():
    """Create and run the FastAPI app for local/dev execution."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()

# Module-level app for runners that expect `server.app:app` (no --factory)
# Create lazily to avoid import-time errors
_app = None

def get_app():
    global _app
    if _app is None:
        _app = create_app()
    return _app

# For direct imports
app = get_app()

# Demo endpoints - add them to the app with error handling
try:
    current_app = get_app()
    
    @current_app.get("/demo", response_class=HTMLResponse, tags=["Demo"])
    def demo_page():
        """Serve the interactive human-playable demo UI."""
        try:
            demo_html_path = os.path.join(os.path.dirname(__file__), "..", "demo.html")
            with open(demo_html_path, encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        except FileNotFoundError:
            logger.warning("demo.html not found")
            return HTMLResponse(content="<h1>Demo Not Available</h1><p>demo.html file not found</p>")
        except Exception as e:
            logger.error("Error serving demo page: %s", e)
            return HTMLResponse(content="<h1>Demo Error</h1><p>Failed to load demo</p>", status_code=500)

    @current_app.get("/benchmark.html", response_class=HTMLResponse, tags=["UI"])
    def benchmark_page():
        """Serve the interactive benchmark visualization UI."""
        try:
            benchmark_html_path = os.path.join(os.path.dirname(__file__), "..", "benchmark.html")
            with open(benchmark_html_path, encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        except FileNotFoundError:
            logger.warning("benchmark.html not found")
            return HTMLResponse(content="<h1>Benchmark Not Available</h1><p>benchmark.html file not found</p>")
        except Exception as e:
            logger.error("Error serving benchmark page: %s", e)
            return HTMLResponse(content="<h1>Benchmark Error</h1><p>Failed to load benchmark</p>", status_code=500)

    @current_app.post("/demo/reset", tags=["Demo"])
    def demo_reset(task: str = "easy"):
        """Start a new demo session. Returns session_id and initial observation."""
        if task not in ("easy", "medium", "hard", "very_hard"):
            raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

        # Evict oldest session if at capacity
        if len(_demo_sessions) >= _MAX_DEMO_SESSIONS:
            oldest_key = next(iter(_demo_sessions))
            del _demo_sessions[oldest_key]

        session_id = str(uuid.uuid4())
        try:
            env = ContentModerationEnvironment()
            obs = env.reset(task_name=task)
            _demo_sessions[session_id] = env
            return {"session_id": session_id, "observation": obs.model_dump()}
        except Exception as e:
            logger.error("Error creating demo session: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to create demo session: {e}")

    @current_app.post("/demo/step/{session_id}", tags=["Demo"])
    def demo_step(session_id: str, action: ModerationAction):
        """Submit one action in an existing demo session."""
        env = _demo_sessions.get(session_id)
        if env is None:
            raise HTTPException(status_code=404, detail="Session not found or expired. Start a new session via /demo/reset.")

        try:
            obs = env.step(action)
            if obs.done:
                del _demo_sessions[session_id]
            return {"observation": obs.model_dump()}
        except Exception as e:
            logger.error("Error in demo step: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to process action: {e}")

except Exception as e:
    logger.error("Failed to register demo endpoints: %s", e)
