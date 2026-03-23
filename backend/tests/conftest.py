"""
Pytest configuration and fixtures for the dance analysis API backend tests.

Provides:
- FastAPI test client
- Database session with transaction rollback for test isolation
- Dependency overrides for FastAPI routes
"""

import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, get_db
from app.main import create_app

# Test database setup
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")

# For SQLite in-memory, use a persistent connection
if "sqlite" in TEST_DATABASE_URL and ":memory:" in TEST_DATABASE_URL:
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=None,
    )
else:
    engine = create_engine(TEST_DATABASE_URL)

# Create all tables before tests run
Base.metadata.create_all(bind=engine)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db() -> Generator[Session, None, None]:
    """Override the get_db dependency to use test database with transaction rollback."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def db() -> Generator[Session, None, None]:
    """Provide a database session for each test with automatic rollback."""
    yield from override_get_db()


@pytest.fixture(scope="function")
def app():
    """Create a FastAPI app instance for testing."""
    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    return app


@pytest.fixture(scope="function")
def client(app) -> TestClient:
    """Provide a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture(scope="function")
def clean_db(db: Session):
    """Provide a clean database session (all tables truncated before test)."""
    # Optional: truncate all tables before each test if needed
    # For now, transaction rollback from override_get_db is sufficient
    yield db
