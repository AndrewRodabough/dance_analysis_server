# Migration Guidelines

This document explains how to write safe, predictable Alembic migrations for this project. It captures the lessons from a difficult debugging session so the same problems are not repeated.

---

## Running Migrations

Migrations must be run **inside the `dance-backend` container**, not from the host machine. The `DATABASE_URL` in `.env` uses the Docker-internal hostname `postgres`, which is only resolvable from within the Docker network.

```bash
docker exec dance-backend sh -c "cd /app && alembic upgrade head"
docker exec dance-backend sh -c "cd /app && alembic current"
docker exec dance-backend sh -c "cd /app && alembic downgrade -1"
```

---

## Creating a New Migration

```bash
docker exec dance-backend sh -c "cd /app && alembic revision -m 'your description here'"
```

Name files sequentially: `002_your_description.py`, `003_your_description.py`, etc.

---

## Enum Rules

This is where most migration bugs originate. Follow these rules exactly.

### Rule 1: Never call `.create()` before `op.create_table()`

`op.create_table()` creates enum types automatically for any `sa.Enum` column it contains. If you also call `.create()` explicitly beforehand, the type gets created twice within the same transaction, and PostgreSQL raises a `DuplicateObject` error.

```python
# WRONG — explicit create + op.create_table = DuplicateObject error
status_enum = sa.Enum("active", "inactive", name="status")

def upgrade():
    bind = op.get_bind()
    status_enum.create(bind, checkfirst=True)  # ← creates it once
    op.create_table("things", sa.Column("status", status_enum, ...))  # ← tries to create again
```

```python
# CORRECT — let op.create_table handle it
status_enum = sa.Enum("active", "inactive", name="status")

def upgrade():
    op.create_table("things", sa.Column("status", status_enum, ...))
```

### Rule 2: Enums used in multiple tables are safe — the memos system handles deduplication

Alembic's DDL runner tracks which enum types have been created within a single `upgrade()` call. The second `op.create_table()` referencing the same enum name will automatically skip creation.

```python
# CORRECT — group_role is used in two tables; Alembic deduplicates automatically
group_role_enum = sa.Enum("owner", "coach", "member", name="group_role")

def upgrade():
    op.create_table("group_memberships", sa.Column("role", group_role_enum, ...), ...)
    op.create_table("group_invites", sa.Column("role", group_role_enum, ...), ...)
```

> **Important:** This deduplication only works when both tables are created in the **same `upgrade()` call**. If they are in different migrations, see Rule 4.

### Rule 3: `create_type=False` on `sa.Enum` does nothing

`create_type=False` is only a valid parameter on `sqlalchemy.dialects.postgresql.ENUM`, not on the generic `sa.Enum`. Passing it to `sa.Enum` is silently ignored.

```python
# WRONG — create_type=False is ignored on sa.Enum
sa.Enum("a", "b", name="my_enum", create_type=False)  # ← has no effect

# CORRECT if you need explicit control over type creation, use postgresql.ENUM
from sqlalchemy.dialects.postgresql import ENUM
ENUM("a", "b", name="my_enum", create_type=False)  # ← works as intended
```

In practice, you should rarely need `create_type=False`. Rely on the memos system instead.

### Rule 4: `op.add_column()` does NOT auto-create enum types

Unlike `op.create_table()`, `op.add_column()` does **not** trigger automatic enum creation. If you are adding a column with a new enum type, you must create the type explicitly first.

```python
new_status = sa.Enum("draft", "published", name="post_status")

def upgrade():
    bind = op.get_bind()
    new_status.create(bind, checkfirst=True)  # ← required before op.add_column
    op.add_column("posts", sa.Column("status", new_status, nullable=False, server_default="draft"))
```

### Rule 5: Dropping enum types in downgrade

When you drop a table, SQLAlchemy automatically drops single-use enum types that were created with that table. For enums used across multiple tables, or for enums added via `op.add_column`, drop them explicitly in `downgrade()` after dropping the tables.

```python
def downgrade():
    op.drop_table("group_invites")
    op.drop_table("group_memberships")

    # group_role was used in both tables — drop it explicitly
    bind = op.get_bind()
    sa.Enum(name="group_role").drop(bind, checkfirst=True)
```

---

## Table Creation Order

Always create tables in foreign-key dependency order. A table must exist before any other table can reference it.

Current dependency order (safe creation sequence):

1. `users`
2. `dances`
3. `groups` → references `users`
4. `group_memberships` → references `groups`, `users`
5. `group_invites` → references `groups`, `users`
6. `routines` → references `users`, `groups`, `dances`
7. `figures`
8. `steps` → references `figures`
9. `videos` → references `users`, `routines`
10. `notes` → references `users`, `routines`, `videos`
11. `jobs` → references `users`, `videos`

Downgrade must drop in reverse order.

---

## Migration Template

```python
"""Brief description

Revision ID: 00X
Revises: 00Y
Create Date: YYYY-MM-DD
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "00X"
down_revision: Union[str, None] = "00Y"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # If adding a new enum column via op.add_column, create the enum type first:
    # new_enum = sa.Enum("val1", "val2", name="my_enum")
    # new_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "my_table",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_my_table_id"), "my_table", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_my_table_id"), table_name="my_table")
    op.drop_table("my_table")
```

---

## Keeping Migrations in Sync with Models

Migrations are **not auto-generated** in this project. When you change an ORM model in `app/models/`, you must manually write a corresponding migration.

Checklist when adding a new resource:

- `app/models/{resource}.py` — SQLAlchemy model
- `app/models/__init__.py` — export the model
- `app/schemas/{resource}.py` — Pydantic schemas
- `app/services/{resource}_service.py` — business logic
- `app/api/v1/{resource}.py` — route handlers
- `app/main.py` — register the router
- `alembic/versions/{next_id}_add_{resource}.py` — migration

---

## Resetting the Database (Development Only)

If the database needs to be wiped entirely (no data to preserve):

```bash
# Terminate active connections and recreate
docker exec dance-postgres psql -U danceuser -d postgres \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'dancedb';"
docker exec dance-postgres psql -U danceuser -d postgres \
  -c "DROP DATABASE IF EXISTS dancedb;" \
  -c "CREATE DATABASE dancedb OWNER danceuser;"

# Apply all migrations from scratch
docker exec dance-backend sh -c "cd /app && alembic upgrade head"
```
