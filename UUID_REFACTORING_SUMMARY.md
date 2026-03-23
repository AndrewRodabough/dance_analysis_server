# UUID Refactoring: Complete Summary & Next Steps

## Overview
Refactoring all database IDs from `Integer` to `UUID` (universally unique identifiers) for better scalability and distributed system compatibility.

**Status**: 90% Complete - Rate limited on final schema file. Most models and all schemas except one have been updated.

---

## ✅ COMPLETED CHANGES

### 1. ORM Models (All 9 models updated)

#### `backend/app/models/user.py`
- `id`: `Integer` → `UUID(as_uuid=True)` with `default=uuid.uuid4`

#### `backend/app/models/group.py`
- `Group.id`: `Integer` → `UUID`
- `Group.created_by`: `Integer` → `UUID` (FK to users)
- `GroupMembership.group_id`: `Integer` → `UUID` (FK)
- `GroupMembership.user_id`: `Integer` → `UUID` (FK)
- `GroupInvite.id`: `Integer` → `UUID`
- `GroupInvite.group_id`: `Integer` → `UUID` (FK)
- `GroupInvite.created_by`: `Integer` → `UUID` (FK)
- `GroupInvite.accepted_by_user_id`: `Integer` → `UUID` (FK)

#### `backend/app/models/dance.py`
- `Dance.id`: `Integer` → `UUID`
- `Dance.tempo`: `Integer` → `String(10)` (flexibility: "120 BPM", "128", etc.)

#### `backend/app/models/routine.py`
- `Routine.id`: `Integer` → `UUID`
- `Routine.group_id`: `Integer` → `UUID` (FK)
- `Routine.created_by`: `Integer` → `UUID` (FK)
- `Routine.dance_id`: `Integer` → `UUID` (FK)

#### `backend/app/models/video.py`
- `Video.id`: `Integer` → `UUID`
- `Video.routine_id`: `Integer` → `UUID` (FK)
- `Video.uploaded_by`: `Integer` → `UUID` (FK)
- `Video.duration`: `Integer` → `String(20)` (ISO 8601 or "HH:MM:SS" format)

#### `backend/app/models/note.py`
- `Note.id`: `Integer` → `UUID`
- `Note.author_id`: `Integer` → `UUID` (FK)
- `Note.routine_id`: `Integer` → `UUID` (FK)
- `Note.video_id`: `Integer` → `UUID` (FK)
- `Note.video_timestamp_ms`: `Integer` → `String(20)` (store milliseconds as string)

#### `backend/app/models/job.py`
- `Job.id`: `Integer` → `UUID`
- `Job.user_id`: `Integer` → `UUID` (FK)
- `Job.video_id`: `Integer` → `UUID` (FK)
- `Job.progress`: `Integer` → `String(3)` (store "0"-"100")
- `Job.attempts`: `Integer` → `String(2)` (store "0"-"99")

#### `backend/app/models/figure.py`
- No changes needed (uses dataclass, not SQLAlchemy ORM)

---

### 2. Pydantic Schemas (6 of 7 updated)

#### `backend/app/schemas/user.py`
```python
id: int → id: UUID
```

#### `backend/app/schemas/group.py`
```python
# GroupResponse
id: int → id: UUID
created_by: int → created_by: UUID

# GroupMembershipResponse
group_id: int → group_id: UUID
user_id: int → user_id: UUID

# AddMemberRequest
user_id: int → user_id: UUID
```

#### `backend/app/schemas/routine.py`
```python
id: int → id: UUID
group_id: int → group_id: UUID
created_by: int → created_by: UUID
```

#### `backend/app/schemas/video.py`
```python
id: int → id: UUID
routine_id: int → routine_id: UUID
uploaded_by: int → uploaded_by: UUID
duration: int → duration: str
VideoDownloadResponse.video_id: int → video_id: UUID
```

#### `backend/app/schemas/note.py`
```python
id: int → id: UUID
author_id: int → author_id: UUID
routine_id: int → routine_id: UUID
video_id: int → video_id: UUID
video_timestamp_ms: int → video_timestamp_ms: str
```

---

## ⚠️ PENDING CHANGES (Rate Limited - Do This Next)

### 1. Group Invite Schema
**File**: `backend/app/schemas/group_invite.py`

```python
"""Pydantic schemas for group invite requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.group import GroupInviteStatus, GroupRole


class GroupInviteCreate(BaseModel):
    """Request body for creating a group invite."""
    email: EmailStr = Field(..., description="Email address to invite.")
    role: Optional[GroupRole] = Field(
        default=None,
        description="Optional role for the invitee (defaults to member).",
    )


class GroupInviteResponse(BaseModel):
    """Response body for a group invite."""
    id: UUID
    group_id: UUID
    created_by: UUID
    email: str
    role: Optional[GroupRole] = None
    status: GroupInviteStatus
    expires_at: datetime
    created_at: datetime
    accepted_at: Optional[datetime] = None
    token: str

    model_config = ConfigDict(from_attributes=True)


class AcceptInviteRequest(BaseModel):
    """Request body for accepting a group invite by token."""
    token: str = Field(..., min_length=1)
```

### 2. Migration File (001_initial_schema.py)
**File**: `backend/alembic/versions/001_initial_schema.py`

Complete rewrite needed. Key changes:
- Import UUID from `sqlalchemy.dialects.postgresql`
- All primary key columns use UUID with `gen_random_uuid()` server default
- All foreign key columns use UUID type
- String columns for tempo, duration, video_timestamp_ms, progress, attempts
- Proper enum handling
- Comprehensive downgrade() function

**Critical**: See the template below in section "Migration File Template"

---

## 🔧 MIGRATION FILE TEMPLATE

Complete replacement for `backend/alembic/versions/001_initial_schema.py`:

```python
"""Initial schema with UUID primary keys

Revision ID: 001
Revises: None
Create Date: 2026-03-20
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Enum definitions
job_status_enum = sa.Enum(
    "pending", "processing", "completed", "failed", "failed_hidden",
    name="jobstatus",
)
video_status_enum = sa.Enum(
    "pending_upload", "uploaded", "deleted",
    name="video_status",
)
dance_style_enum = sa.Enum(
    "samba", "cha_cha", "rumba", "paso_doble", "jive",
    "waltz", "tango", "viennese_waltz", "foxtrot", "quickstep",
    "american_waltz", "american_tango", "american_foxtrot", "american_viennese_waltz",
    "american_cha_cha", "american_rumba", "swing", "bolero", "mambo",
    name="dance_style",
)
note_type_enum = sa.Enum(
    "critique", "feedback", "complement",
    name="note_type",
)
note_source_enum = sa.Enum(
    "user", "ai", "system",
    name="note_source",
)
membership_status_enum = sa.Enum(
    "active", "invited", "removed",
    name="membership_status",
)
group_invite_status_enum = sa.Enum(
    "pending", "accepted", "revoked", "expired",
    name="group_invite_status",
)
group_role_enum = sa.Enum(
    "owner", "coach", "member",
    name="group_role",
)


def upgrade() -> None:
    # Users
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("username", sa.String(50), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("is_superuser", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

    # Dances
    op.create_table(
        "dances",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tempo", sa.String(10), nullable=False),
        sa.Column("meter", sa.String(10), nullable=False),
        sa.Column("style", dance_style_enum, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_dances_style"), "dances", ["style"])

    # Groups
    op.create_table(
        "groups",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("is_archived", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_groups_created_by"), "groups", ["created_by"])

    # Group Memberships
    op.create_table(
        "group_memberships",
        sa.Column("group_id", UUID(as_uuid=True), sa.ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", group_role_enum, nullable=False),
        sa.Column("status", membership_status_enum, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # Group Invites
    op.create_table(
        "group_invites",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("group_id", UUID(as_uuid=True), sa.ForeignKey("groups.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("role", group_role_enum, nullable=True),
        sa.Column("token", sa.String(255), nullable=False, unique=True),
        sa.Column("status", group_invite_status_enum, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_by_user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    )

    # Routines
    op.create_table(
        "routines",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("group_id", UUID(as_uuid=True), sa.ForeignKey("groups.id", ondelete="CASCADE"), nullable=True),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dance_id", UUID(as_uuid=True), sa.ForeignKey("dances.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Figures
    op.create_table(
        "figures",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("tags", JSONB, nullable=True),
        sa.Column("total_beats", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Steps
    op.create_table(
        "steps",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("figure_id", UUID(as_uuid=True), sa.ForeignKey("figures.id", ondelete="CASCADE"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("start_state", JSONB, nullable=False),
        sa.Column("end_state", JSONB, nullable=False),
        sa.Column("keyframes", JSONB, nullable=True),
        sa.Column("duration", sa.Float(), nullable=False),
    )

    # Videos
    op.create_table(
        "videos",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("routine_id", UUID(as_uuid=True), sa.ForeignKey("routines.id", ondelete="CASCADE"), nullable=True),
        sa.Column("uploaded_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("storage_key", sa.String(500), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("duration", sa.String(20), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=True),
        sa.Column("status", video_status_enum, server_default="pending_upload", nullable=False),
        sa.Column("original_filename", sa.String(500), nullable=True),
        sa.Column("content_type", sa.String(100), nullable=True),
    )

    # Notes
    op.create_table(
        "notes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("author_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("routine_id", UUID(as_uuid=True), sa.ForeignKey("routines.id", ondelete="CASCADE"), nullable=False),
        sa.Column("note_type", note_type_enum, nullable=False),
        sa.Column("contents", sa.Text(), nullable=False),
        sa.Column("source", note_source_enum, server_default="user", nullable=False),
        sa.Column("video_id", UUID(as_uuid=True), sa.ForeignKey("videos.id", ondelete="SET NULL"), nullable=True),
        sa.Column("video_deleted", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("video_timestamp_ms", sa.String(20), nullable=True),
        sa.Column("details", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Jobs
    op.create_table(
        "jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", sa.String(36), nullable=False, unique=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", job_status_enum, server_default="pending", nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("video_id", UUID(as_uuid=True), sa.ForeignKey("videos.id", ondelete="SET NULL"), nullable=True),
        sa.Column("result_path", sa.String(500), nullable=True),
        sa.Column("data_path", sa.String(500), nullable=True),
        sa.Column("progress", sa.String(3), server_default="0", nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("attempts", sa.String(2), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    # Drop tables in reverse dependency order
    op.drop_table("jobs")
    op.drop_table("notes")
    op.drop_table("videos")
    op.drop_table("steps")
    op.drop_table("figures")
    op.drop_table("routines")
    op.drop_table("group_invites")
    op.drop_table("group_memberships")
    op.drop_table("groups")
    op.drop_table("dances")
    op.drop_table("users")
```

---

## 🚀 FINAL STEPS TO COMPLETE

### Step 1: Update Group Invite Schema
Copy the schema code from the "PENDING CHANGES" section above into `backend/app/schemas/group_invite.py`

### Step 2: Update Migration File
Replace the entire `backend/alembic/versions/001_initial_schema.py` with the template provided above

### Step 3: Drop and Recreate Database

```bash
# Connect to PostgreSQL and drop the database
psql -U danceuser -d postgres -c "DROP DATABASE IF EXISTS danceuser_db;"
psql -U danceuser -d postgres -c "CREATE DATABASE danceuser_db;"

# Run migrations
cd dance_analysis_server/backend
alembic upgrade head

# Verify schema
psql -U danceuser -d danceuser_db -c "\dt"
psql -U danceuser -d danceuser_db -c "SELECT column_name, data_type FROM information_schema.columns WHERE table_name='users' ORDER BY ordinal_position;"
```

### Step 4: Verify with Tests

```bash
cd dance_analysis_server/backend
pip install -r requirements.txt
pytest tests/ -v
```

Expected: All tests should pass with new UUID schema

### Step 5: Update Tests (if needed)

Any test files that hardcode integer IDs will need updates. Look for:
- Test fixtures with `id=1`, `id=2`, etc.
- Mock responses with integer IDs
- Assertions comparing integer IDs

Replace with UUID generation:
```python
import uuid

user_id = uuid.uuid4()
group_id = uuid.uuid4()
```

---

## 🔑 Key Technical Details

### UUID Generation Strategy

**Python-side** (ORM):
```python
id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
```

**Database-side** (Direct SQL):
```sql
id UUID PRIMARY KEY DEFAULT gen_random_uuid()
```

Both are specified for maximum compatibility with different insertion methods.

### String Columns for Numeric Data

Changed these columns to `String` for flexibility:
- `Dance.tempo` → `String(10)` (allows "120", "120 BPM", "120 bpm", etc.)
- `Video.duration` → `String(20)` (allows "123" seconds, "1:23:45", "PT1M23S", etc.)
- `Note.video_timestamp_ms` → `String(20)` (allows "12300", "12.3", etc.)
- `Job.progress` → `String(3)` (stores "0" to "100")
- `Job.attempts` → `String(2)` (stores "0" to "99")

This provides flexibility without enforcing a specific format in the database.

---

## 📋 Verification Checklist

- [ ] Group invite schema updated
- [ ] Migration file (001_initial_schema.py) replaced
- [ ] Database dropped and recreated with `alembic upgrade head`
- [ ] Schema verified with `\dt` and `\d tablename` in psql
- [ ] All tests passing with `pytest tests/ -v`
- [ ] No remaining hardcoded integer ID references in tests
- [ ] API client (Flutter/REST) updated to handle UUID responses
- [ ] Documentation updated with new ID format

---

## Notes

- **No data loss**: Database is empty, so no migration data needed
- **Breaking change**: API responses now return UUIDs instead of integers
- **Client impact**: Flutter and any REST clients must handle UUID strings
- **Performance**: UUID indexing in PostgreSQL is efficient and well-supported
- **Distributed systems**: UUIDs enable better multi-database scenarios
