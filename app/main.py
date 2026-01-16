from fastapi import FastAPI

from app.api.v1 import analyze, health


def create_app() -> FastAPI:
    """Application factory to build the FastAPI app."""

    app = FastAPI(title="Dance Analysis API", version="1.0.0")
    app.include_router(health.router)
    app.include_router(analyze.router)

    return app


app = create_app()
