# Migration 005: Session Refactor - Access Control, Participation, and Per-User State

## Overview

This migration completes the session refactor by separating **content**, **access control**, **participation**, and **per-user state** into distinct layers.

**Revision ID:** 005  
**Previous:** 004  
**Postgres Required:** Yes (enums, CHECK constraints)

---

## What Changes

### 1. RoutineSession Model
- **Add:** `owner_id` (FK to users)
- **Remove:** `group_id`
- Sessions are now "content only" with explicit ownership
- Owner identity is independent of creator (both stored)

### 2. SessionAccess Table (New)
- **Purpose:** Permissions layer
- **Tracks:** Who can access what session
- **Mechanism:** User-level or group-level access with roles (viewer, editor, admin)
- **Constraint:** Exactly one of `user_id` or `group_id` (enforced via CHECK)

### 3. SessionParticipant Table (New)
- **Purpose:** Participation layer
- **Tracks:** Who is actually involved in the routine
- **Roles:** dancer, coach
- **Separate from:** Access control (a user can have access but not be a participant)

### 4. SessionUserState Table (New)
- **Purpose:** Per-user session visibility
- **Tracks:** Archive/delete status per user
- **Enabled:** Users can archive or delete sessions independently without affecting others

---

## Migration Steps

### Upgrade Path

1. **Create Enums** (PostgreSQL types)
   - `session_access_role`: viewer, editor, admin
   - `participant_role`: dancer, coach

2. **Add owner_id to routine_sessions**
   - Adds nullable column
   - Backfills all existing sessions: `owner_id = created_by`
   - Adds NOT NULL constraint
   - Adds FK to users
   - Adds index

3. **Create session_access Table**
   - Stores (session, user/group, role) tuples
   - CHECK constraint: exactly one of user_id or group_id
   - Indexes on session_id, user_id, group_id

4. **Create session_participants Table**
   - Stores (session, user, role, optional dancer_slot)
   - Unique constraint: (session_id, user_id)
   - Indexes on session_id, user_id, dancer_slot_id

5. **Create session_user_states Table**
   - Stores (session, user, is_archived, is_deleted)
   - Unique constraint: (session_id, user_id)
   - Indexes on session_id, user_id

6. **Seed SessionAccess for Owners**
   - For each session: insert (session_id, owner_id, role='admin')
   - Ensures owner always has admin access

7. **Remove group_id from routine_sessions**
   - Drop index on group_id
   - Drop FK constraint
   - Drop column

---

## Data Migration Strategy

### Sessions → Access

The migration grants **admin access to session owners** automatically:

```sql
INSERT INTO session_access (session_id, user_id, role, created_at)
SELECT id, owner_id, 'admin', now()
FROM routine_sessions
```

**Result:** Every session owner has an explicit admin SessionAccess record.

### Sessions → Participants

**No automatic seeding of participants.** Participants are added explicitly by the application when:
- Users are assigned to dance slots
- Users join as coaches
- Users are invited to participate

This prevents accidentally creating participation records and ensures intentional involvement.

### Sessions → UserState

**No automatic seeding of user states.** Per-user state is created on-demand when:
- A user archives the session
- A user deletes the session
- The application initializes user state for listing/filtering

---

## Rollback Considerations

### Downgrade Path

1. **Re-add group_id to routine_sessions**
   - Backfill from session_access: first group_id per session (where group_id IS NOT NULL)
   - If no group access exists, column remains NULL
   - Re-adds FK and index

2. **Drop session_user_states Table**

3. **Drop session_participants Table**

4. **Drop session_access Table**

5. **Remove owner_id from routine_sessions**
   - Drop index and FK
   - Drop column

6. **Drop Enums**
   - `participant_role`
   - `session_access_role`

### Data Loss on Downgrade

⚠️ **Important:** Downgrading loses:
- All participation records (session_participants)
- All per-user state (session_user_states, is_archived/is_deleted flags)
- Granular access records beyond the primary group per session

To preserve data, export before downgrade:
```sql
-- Export participants
SELECT * FROM session_participants;

-- Export user states
SELECT * FROM session_user_states;

-- Export access records
SELECT * FROM session_access;
```

---

## Validation After Migration

### 1. Check Constraints Applied

```sql
-- Verify CHECK constraint on session_access
SELECT constraint_name, constraint_definition
FROM information_schema.check_constraints
WHERE table_name = 'session_access';
```

Expected: `ck_session_access_one_subject` should exist.

### 2. Enums Created

```sql
-- List enum types
SELECT typname FROM pg_type WHERE typtype = 'e';
```

Expected: `session_access_role`, `participant_role` present.

### 3. Owner Access Seeded

```sql
-- Count owner access records
SELECT COUNT(*)
FROM session_access
WHERE role = 'admin';

-- Should equal number of sessions (or more if shared)
SELECT COUNT(*) FROM routine_sessions;
```

Expected: At least one admin per session.

### 4. group_id Removed

```sql
-- Verify column doesn't exist
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'routine_sessions'
AND column_name = 'group_id';
```

Expected: Empty result set (no rows).

### 5. owner_id Present and Populated

```sql
-- Check owner_id not null
SELECT COUNT(*) as null_count
FROM routine_sessions
WHERE owner_id IS NULL;
```

Expected: 0 (no NULLs).

---

## Application Updates Required

After running the migration, update the application to:

### 1. Session Creation
```python
# Create session with owner_id set
session = RoutineSession(
    routine_id=routine_id,
    created_by=user_id,
    owner_id=user_id,  # NEW: explicitly set owner
    label=label,
)
# SessionAccess row is auto-seeded by migration for existing sessions
# For new sessions, create SessionAccess:
access = SessionAccess(
    session_id=session.id,
    user_id=user_id,
    role="admin",
)
```

### 2. Session Access Queries
```python
# Check access: query SessionAccess instead of session.group_id
access_record = db.query(SessionAccess).filter(
    SessionAccess.session_id == session_id,
    (SessionAccess.user_id == user_id) |
    (SessionAccess.group_id.in_(user_groups))
).first()

if access_record:
    # User has access with role: access_record.role
    pass
```

### 3. Session Sharing
```python
# Share with user
SessionAccess.create(
    session_id=session_id,
    user_id=other_user_id,
    role="viewer",  # or "editor"
)

# Share with group
SessionAccess.create(
    session_id=session_id,
    group_id=group_id,
    role="editor",
)
```

### 4. Per-User Archive/Delete
```python
# Archive for user
user_state = SessionUserState.get_or_create(
    session_id=session_id,
    user_id=user_id,
)
user_state.is_archived = True
db.commit()

# Delete for user
user_state.is_deleted = True
db.commit()

# List visible sessions (exclude archived/deleted)
visible_sessions = (
    db.query(RoutineSession)
    .outerjoin(SessionUserState)
    .filter(
        (SessionUserState.is_deleted != True) |
        (SessionUserState.id == None)
    )
)
```

---

## Troubleshooting

### Error: "Duplicate key value violates unique constraint"

**Cause:** Multiple session_access records created during seeding.

**Fix:** Already handled by INSERT logic. If seeding fails, manually delete duplicates:
```sql
DELETE FROM session_access sa1
WHERE EXISTS (
    SELECT 1 FROM session_access sa2
    WHERE sa1.session_id = sa2.session_id
    AND sa1.user_id = sa2.user_id
    AND sa1.id > sa2.id
);
```

### Error: "CHECK constraint violation on session_access"

**Cause:** Attempted to insert both user_id and group_id.

**Fix:** Ensure exactly one is NULL:
```sql
INSERT INTO session_access (session_id, user_id, group_id, role)
VALUES (sid, uid, NULL, 'admin');  -- Good

INSERT INTO session_access (session_id, user_id, group_id, role)
VALUES (sid, uid, gid, 'admin');  -- Bad: violates constraint
```

### group_id Backfill on Downgrade Lost Data

**Cause:** Downgrade can't recover all groups if multiple groups shared a session.

**Prevention:** Export all session_access records before downgrading.

---

## Related Documentation

- `SESSION_REFACTOR_MODELS_COMPLETE.md` - Model definitions and relationships
- `ROUTINE_SESSION_REFACTOR_ASSESSMENT_PLAN.md` - High-level refactor goals
- Models: `session_access.py`, `session_participant.py`, `session_user_state.py`

---

## Testing Checklist

- [ ] Migration applies without errors
- [ ] owner_id set for all existing sessions
- [ ] SessionAccess rows created for all session owners
- [ ] group_id removed from routine_sessions
- [ ] Enums created and visible in psql
- [ ] CHECK constraint enforces one subject
- [ ] Downgrade completes successfully
- [ ] No data loss in core session records
- [ ] Indexes created on all foreign keys
- [ ] Unique constraints enforced (session_participants, session_user_states)
