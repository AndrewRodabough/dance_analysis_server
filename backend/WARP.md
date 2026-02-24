# LLM Code Generation Guide - Dance Analysis Backend

## Project Overview
FastAPI backend for video dance analysis with pose estimation. Uses PostgreSQL + Alembic for persistence, MinIO (S3-compatible) for object storage, and JWT-based authentication.

## Tech Stack
- **Framework**: FastAPI 0.104.1
- **Database**: PostgreSQL via SQLAlchemy 2.0 (async not used - sync sessions)
- **Migrations**: Alembic 1.12.1
- **Auth**: JWT with python-jose, bcrypt password hashing
- **Storage**: S3/MinIO via boto3
- **Python**: 3.10

## Directory Structure
```
backend/
├── app/
│   ├── main.py              # App factory, router registration
│   ├── database.py          # Engine, SessionLocal, Base, get_db
│   ├── core/
│   │   ├── config.py        # Settings class with env vars
│   │   ├── deps.py          # FastAPI dependencies (auth)
│   │   └── security.py      # Password hashing, JWT utils
│   ├── models/              # SQLAlchemy ORM models
│   │   ├── __init__.py      # Export all models + Base
│   │   ├── user.py
│   │   └── job.py
│   ├── schemas/             # Pydantic request/response schemas
│   │   ├── user.py
│   │   ├── job.py
│   │   └── token.py
│   ├── services/            # Business logic layer
│   │   └── job_service.py
│   └── api/v1/              # Route handlers
│       ├── auth.py
│       ├── jobs.py
│       ├── analyze.py
│       ├── videos.py
│       └── health.py
├── alembic/
│   └── versions/            # Migration files (###_description.py)
└── requirements.txt
```

## Code Patterns

### SQLAlchemy Models
```python
# app/models/example.py
"""Brief module docstring."""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Example(Base):
    __tablename__ = "examples"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", backref="examples")

    def __repr__(self):
        return f"<Example(id={self.id}, name={self.name})>"
```

**Model conventions:**
- Use `Column()` with explicit `nullable=` param
- Index frequently queried columns: `index=True`
- Foreign keys use `ondelete="CASCADE"` where appropriate
- Always include `__repr__` for debugging
- Timestamps: `created_at` with `server_default=func.now()`, `updated_at` with `onupdate=func.now()`
- String columns: specify max length `String(255)`
- Export models in `app/models/__init__.py`

### Pydantic Schemas
```python
# app/schemas/example.py
"""Pydantic schemas for example-related requests and responses."""

from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional


class ExampleBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class ExampleCreate(ExampleBase):
    """Request body for creating an example."""
    pass


class ExampleResponse(ExampleBase):
    """Response body for example data."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
```

**Schema conventions:**
- Base class for shared fields
- `Create` schema for POST bodies
- `Response` schema with `model_config = ConfigDict(from_attributes=True)` for ORM compatibility
- Use `Field(...)` for required fields with validation
- Use `Optional[T] = None` for nullable fields

### Service Layer
```python
# app/services/example_service.py
"""Service for managing example operations."""

from sqlalchemy.orm import Session
from typing import Optional, List

from app.models.example import Example
from app.schemas.example import ExampleCreate


class ExampleService:
    """Service for managing example operations."""

    @staticmethod
    def create(db: Session, user_id: int, data: ExampleCreate) -> Example:
        """Create a new example."""
        new_example = Example(
            name=data.name,
            user_id=user_id,
        )
        db.add(new_example)
        db.commit()
        db.refresh(new_example)
        return new_example

    @staticmethod
    def get_by_id(db: Session, example_id: int, user_id: Optional[int] = None) -> Optional[Example]:
        """Get example by ID, optionally filtered by user."""
        query = db.query(Example).filter(Example.id == example_id)
        if user_id is not None:
            query = query.filter(Example.user_id == user_id)
        return query.first()

    @staticmethod
    def get_user_examples(
        db: Session,
        user_id: int,
        limit: int = 50,
        offset: int = 0
    ) -> List[Example]:
        """Get all examples for a user with pagination."""
        return db.query(Example).filter(
            Example.user_id == user_id
        ).order_by(Example.created_at.desc()).limit(limit).offset(offset).all()

    @staticmethod
    def delete(db: Session, example_id: int, user_id: int) -> bool:
        """Delete an example. Returns True if deleted, False if not found."""
        example = db.query(Example).filter(
            Example.id == example_id,
            Example.user_id == user_id
        ).first()
        if not example:
            return False
        db.delete(example)
        db.commit()
        return True
```

**Service conventions:**
- Static methods on a class (not instance methods)
- `db: Session` is first parameter
- Return ORM objects, not schemas
- Filter by `user_id` for authorization
- Return `Optional` or `bool` for operations that may fail

### API Routes
```python
# app/api/v1/examples.py
"""Example management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.core.deps import get_current_active_user
from app.models.user import User
from app.schemas.example import ExampleCreate, ExampleResponse
from app.services.example_service import ExampleService

router = APIRouter()


@router.post("", response_model=ExampleResponse, status_code=status.HTTP_201_CREATED)
def create_example(
    data: ExampleCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new example.

    - **name**: Example name (1-255 characters)
    """
    return ExampleService.create(db, current_user.id, data)


@router.get("", response_model=List[ExampleResponse])
def list_examples(
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all examples for current user."""
    return ExampleService.get_user_examples(db, current_user.id, limit, offset)


@router.get("/{example_id}", response_model=ExampleResponse)
def get_example(
    example_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific example."""
    example = ExampleService.get_by_id(db, example_id, current_user.id)
    if not example:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Example not found"
        )
    return example


@router.delete("/{example_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_example(
    example_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete an example."""
    deleted = ExampleService.delete(db, example_id, current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Example not found"
        )
    return None
```

**Route conventions:**
- Module docstring describes endpoint group
- `router = APIRouter()` (no prefix - added in main.py)
- Auth via `Depends(get_current_active_user)` 
- DB via `Depends(get_db)`
- Use `status.HTTP_*` constants
- Docstrings with param descriptions for Swagger
- Return `None` for 204 responses
- Service does the work, route handles HTTP concerns

### Router Registration (main.py)
```python
from app.api.v1 import examples

app.include_router(examples.router, prefix="/api/v1/examples", tags=["examples"])
```

### Alembic Migrations
```python
# alembic/versions/002_add_examples_table.py
"""Add examples table

Revision ID: 002
Revises: 001
Create Date: 2026-02-17
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('examples',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_examples_id'), 'examples', ['id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_examples_id'), table_name='examples')
    op.drop_table('examples')
```

**Migration conventions:**
- Sequential numeric revision IDs: `001`, `002`, etc.
- Filename: `{revision}_{description}.py`
- Include both `upgrade()` and `downgrade()`
- Create indexes for primary keys and frequently queried columns
- Match model definitions exactly

## Authentication Pattern
All protected endpoints use:
```python
current_user: User = Depends(get_current_active_user)
```

This:
1. Extracts JWT from `Authorization: Bearer <token>` header
2. Decodes and validates the token
3. Looks up user by ID from token's `sub` claim
4. Raises 401 if invalid, 403 if user inactive
5. Returns the User ORM object

## Configuration
Add new settings in `app/core/config.py`:
```python
class Settings:
    NEW_SETTING: str = os.getenv("NEW_SETTING", "default_value")
```

Access via: `from app.core.config import settings`

## Testing Philosophy
- Test behavior, not implementation
- Focus on meaningful coverage over raw percentage
- Test edge cases that reflect real failures
- Don't test Python language features

## Common Gotchas

1. **Sync SQLAlchemy** - This project uses synchronous sessions, not async
2. **Service methods are static** - Use `@staticmethod`, not instance methods
3. **Models in __init__** - Always export new models in `app/models/__init__.py`
4. **ConfigDict** - Use `model_config = ConfigDict(from_attributes=True)` not `class Config`
5. **UTC timestamps** - Use `DateTime(timezone=True)` for all timestamps
6. **Query params** - Use `Query()` with `ge`/`le` for pagination bounds
7. **HTTPException imports** - Use `from fastapi import HTTPException, status`

## File Checklist for New Features

When adding a new resource/entity:
- [ ] `app/models/{resource}.py` - SQLAlchemy model
- [ ] Update `app/models/__init__.py` - Export model
- [ ] `app/schemas/{resource}.py` - Pydantic schemas
- [ ] `app/services/{resource}_service.py` - Business logic
- [ ] `app/api/v1/{resource}.py` - Route handlers
- [ ] Update `app/main.py` - Register router
- [ ] `alembic/versions/{next}_add_{resource}.py` - Migration
