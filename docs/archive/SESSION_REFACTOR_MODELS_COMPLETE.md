# Session Refactor Models - Complete Implementation

## Overview

The session refactor has been completed with four models that separate content, access control, participation, and per-user state.

## Models Created

### 1. RoutineSession (Updated)

**File:** `backend/app/models/routine_session.py`

**Changes:**
- Removed `group_id`
- Added `owner_id` (FK to users)
- Maintained: `id`, `routine_id`, `created_by`, `label`, timestamps
- Added relationships to `SessionAccess`, `SessionParticipant`, `SessionUserState`

**Key fields:**
```
id: UUID (PK)
routine_id: UUID (FK) - CASCADE
created_by: UUID (FK) - CASCADE
owner_id: UUID (FK) - CASCADE
label: String(255) nullable
created_at: DateTime
updated_at: DateTime
```

**Relationships:**
- `routine` ← Routine (back_populates="sessions")
- `creator` ← User (foreign_keys=[created_by])
- `owner` ← User (foreign_keys=[owner_id])
- `access_records` ← SessionAccess (cascade delete-orphan)
- `participants` ← SessionParticipant (cascade delete-orphan)
- `user_states` ← SessionUserState (cascade delete-orphan)
- `slot_assignments` ← SlotAssignment (cascade delete-orphan)
- `videos` ← Video (cascade delete-orphan)
- `notes` ← Note (cascade delete-orphan)

---

### 2. SessionAccess (New)

**File:** `backend/app/models/session_access.py`

**Purpose:** Controls who can access a session (permissions layer)

**Key fields:**
```
id: UUID (PK)
session_id: UUID (FK) - CASCADE
user_id: UUID (FK, nullable) - CASCADE
group_id: UUID (FK, nullable) - CASCADE
role: Enum (VIEWER, EDITOR, ADMIN)
created_at: DateTime
```

**Constraints:**
- CHECK: `(user_id IS NOT NULL AND group_id IS NULL) OR (user_id IS NULL AND group_id IS NOT NULL)`
  - Enforces exactly one of user_id or group_id is set
- Index on `session_id`
- Index on `user_id`
- Index on `group_id`

**Relationships:**
- `routine_session` ← RoutineSession (back_populates="access_records")
- `user` ← User (backref="session_access_records")
- `group` ← Group (backref="session_access_records")

**Rules:**
- This is the ONLY access control mechanism
- Owner always has implicit admin rights
- Group membership determines access dynamically (no duplication of group data)

---

### 3. SessionParticipant (New)

**File:** `backend/app/models/session_participant.py`

**Purpose:** Represents who is involved in the routine (participation layer)

**Key fields:**
```
id: UUID (PK)
session_id: UUID (FK) - CASCADE
user_id: UUID (FK) - CASCADE
role: Enum (DANCER, COACH)
dancer_slot_id: UUID (FK, nullable) - SET NULL
created_at: DateTime
```

**Constraints:**
- UNIQUE: `(session_id, user_id)` - one participation record per user per session
- Index on `session_id`
- Index on `user_id`
- Index on `dancer_slot_id`

**Relationships:**
- `routine_session` ← RoutineSession (back_populates="participants")
- `user` ← User (backref="session_participants")
- `dancer_slot` ← RoutineDancerSlot (backref="session_participants")

**Rules:**
- NOT used for permissions
- Represents actual involvement in the routine
- Can link to a dancer slot but does not require it

---

### 4. SessionUserState (New)

**File:** `backend/app/models/session_user_state.py`

**Purpose:** Per-user session visibility and state (archive/delete)

**Key fields:**
```
id: UUID (PK)
session_id: UUID (FK) - CASCADE
user_id: UUID (FK) - CASCADE
is_archived: Boolean (default False)
is_deleted: Boolean (default False)
created_at: DateTime
updated_at: DateTime
```

**Constraints:**
- UNIQUE: `(session_id, user_id)` - one state record per user per session
- Index on `session_id`
- Index on `user_id`

**Relationships:**
- `routine_session` ← RoutineSession (back_populates="user_states")
- `user` ← User (backref="session_user_states")

**Rules:**
- Enables per-user archive/delete without affecting other users
- Does not cascade delete session content
- Independent for each user

---

## Ownership Rules

1. Owner is set at session creation via `owner_id`
2. Creator may differ from owner (can be reassigned)
3. Owner always has implicit admin access via `SessionAccess`
4. When owner leaves, session content is preserved
5. Sessions must have an owner (owner_id NOT NULL)

---

## Access Pattern

### To check if user has access to a session:

1. Query `SessionAccess` where `session_id=X` and `user_id=Y` → get role
2. OR query `SessionAccess` where `session_id=X` and `group_id` in user's groups → get role
3. Owner always has admin rights (can be implicit or explicit)

### To share a session:

- Create `SessionAccess` row with target `user_id` or `group_id` and desired role
- Group membership changes do NOT require session updates

---

## Participation Pattern

### To add a user as a participant:

1. Create `SessionParticipant` row with `session_id`, `user_id`, `role`
2. Optionally link to `dancer_slot_id` if assigning to a choreography position
3. User may participate without having access (orchestrated separately)

### To remove a user from participation:

- Delete `SessionParticipant` row
- Does NOT affect access or per-user state

---

## Per-User State Pattern

### To archive for a user:

1. Create or update `SessionUserState` row with `is_archived=True`
2. Other users' views unaffected
3. Content not deleted, only hidden for that user

### To delete for a user:

1. Create or update `SessionUserState` row with `is_deleted=True`
2. May be reverted by setting back to False
3. Session content preserved for others

---

## Exports

Updated `backend/app/models/__init__.py` to include:

```python
from app.models.session_access import SessionAccess, SessionAccessRole
from app.models.session_participant import SessionParticipant, ParticipantRole
from app.models.session_user_state import SessionUserState
```

---

## Migration Requirements

The following Alembic migration is needed:

1. Remove `group_id` from `routine_sessions` table
2. Add `owner_id` to `routine_sessions` table (set to `created_by` for existing rows)
3. Create `session_access` table with CHECK constraint
4. Create `session_participants` table
5. Create `session_user_states` table
6. Update `Group.routine_sessions` relationship (if still needed for backward compat)

---

## No Changes To

- `RoutineDancerSlot` - unchanged
- `SlotAssignment` - unchanged
- `Video` and `Note` - already scoped to `routine_session_id`
- Auth and group membership logic - authorization flows through `SessionAccess`
