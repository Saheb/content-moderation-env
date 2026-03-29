from openenv.core.env_server import create_fastapi_app
from .environment import ContentModerationEnvironment
from models import ModerationAction, ModerationObservation
import uvicorn


def create_app():
    return create_fastapi_app(
        ContentModerationEnvironment,
        action_cls=ModerationAction,
        observation_cls=ModerationObservation,
    )


def main():
    """Create and run the FastAPI app for local/dev execution.

    This function is installable as a console script (`server`) and
    is also guarded by an ``if __name__ == '__main__'`` block below.
    """
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    return app


if __name__ == "__main__":
    main()