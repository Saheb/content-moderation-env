from openenv.core.env_server import create_fastapi_app
from .environment import ContentModerationEnvironment
from models import ModerationAction, ModerationObservation
import uvicorn
import logging

logger = logging.getLogger(__name__)


def create_app():
    """Factory function for creating the FastAPI app instance."""
    logger.info("Creating FastAPI app...")
    # create_fastapi_app expects a factory (callable) that creates new env instances
    app = create_fastapi_app(
        lambda: ContentModerationEnvironment(),
        action_cls=ModerationAction,
        observation_cls=ModerationObservation,
    )
    
    @app.get("/", tags=["Health"])
    def root():
        """Root endpoint returning basic environment status."""
        return {"status": "ok", "message": "Content moderation environment running"}
    
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