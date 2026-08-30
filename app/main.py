from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.interactions import router as interactions_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.personalization import router as personalization_router
from app.api.routes.resources import router as resources_router
from app.api.routes.speech import router as speech_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.init_db import create_tables
from app.db.session import engine


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        create_tables(engine)
        yield

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(interactions_router)
    app.include_router(personalization_router)
    app.include_router(resources_router)
    app.include_router(knowledge_router)
    app.include_router(speech_router)
    return app


app = create_app()
