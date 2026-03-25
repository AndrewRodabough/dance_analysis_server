# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest

# Run a single test file or test
pytest tests/unit/test_authorization_nonleaky.py
pytest tests/unit/test_routes_smoke.py::TestRoutesSmoke::test_health

# Start the server locally (after setting env vars)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start all services via Docker
docker-compose up backend

# Run database migrations
alembic upgrade head

# Create a new migration
alembic revision --autogenerate -m "description"
```

## Environment Variables

Configured in `app/core/config.py`. Key variables:

| Variable | Default |
|----------|---------|
| `DATABASE_URL` | `postgresql://danceuser:dancepass@localhost:5432/dancedb` |
| `JWT_SECRET_KEY` | `dev-secret-key-change-in-production` |
| `S3_ENDPOINT` | `http://minio:9000` |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | `minioadmin` |
| `S3_BUCKET` | `dance-videos` |
| `USE_MOCK_ANALYSIS` | `false` |

For Docker, see `docker-compose.yml` for service-specific env anchors (`common-env`, `mac-env`, etc.).

## Architecture

**Stack:** FastAPI + SQLAlchemy (sync) + PostgreSQL + Alembic + MinIO (S3) + JWT auth

**Request flow:** Route handler → Service → ORM model → Database

**Layer responsibilities:**
- `app/core/` — cross-cutting concerns: config, JWT deps, auth helpers, logging
- `app/models/` — SQLAlchemy ORM; all PKs are UUIDs
- `app/schemas/` — Pydantic I/O; `Base`/`Create`/`Response` pattern
- `app/services/` — business logic with static methods; returns ORM objects, not schemas
- `app/api/v1/` — thin route handlers; delegate to services
- `app/middleware/` — `RequestLoggingMiddleware` injects request ID into log context

**Authorization:** Non-leaky access control — unauthorized access to another user's resource returns 404, not 403. Helpers live in `app/core/authorization.py` (e.g., `require_group_member`, `require_routine_owner`).

**Testing:** In-memory SQLite with per-test transaction rollback. `tests/conftest.py` provides `db`, `app`, `client`, and `clean_db` fixtures.

## Code Conventions

**Models** (`app/models/`)
- UUID primary keys using PostgreSQL `UUID` type
- All timestamps use `DateTime(timezone=True)` — `created_at` with `server_default=func.now()`, `updated_at` with `onupdate=func.now()`
- `ondelete="CASCADE"` on FK columns where appropriate
- Export all models in `app/models/__init__.py`

**Schemas** (`app/schemas/`)
- Response schemas must have `model_config = ConfigDict(from_attributes=True)` — never use the old `class Config`

**Services** (`app/services/`)
- All methods are `@staticmethod`; first param is `db: Session`
- Services own `db.add()`, `db.commit()`, `db.refresh()`

**Routes** (`app/api/v1/`)
- No prefix on `APIRouter()` — prefix is set in `app/main.py` during registration
- Auth: `current_user: User = Depends(get_current_active_user)`
- DB: `db: Session = Depends(get_db)`
- Use `status.HTTP_*` constants; return `None` for 204 responses

**Migrations** (`alembic/versions/`)
- Sequential numeric IDs: `001`, `002`, …
- Filename: `{revision}_{description}.py`
- Always include both `upgrade()` and `downgrade()`

## Checklist for New Resources

- [ ] `app/models/{resource}.py` — SQLAlchemy model
- [ ] Update `app/models/__init__.py` — export model
- [ ] `app/schemas/{resource}.py` — Pydantic schemas
- [ ] `app/services/{resource}_service.py` — business logic
- [ ] `app/api/v1/{resource}.py` — route handlers
- [ ] Update `app/main.py` — register router
- [ ] `alembic/versions/{next}_{description}.py` — migration
