from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings


settings = get_settings()
if settings.database_url is None and settings.database_path is not None:
    settings._resolve_path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(settings.resolved_database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
