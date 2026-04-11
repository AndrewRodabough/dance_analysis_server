# Backend Refactor Implementation Summary

## Overview

This document summarizes the implementation of Phase 1-5 of the backend refactor for routine-centric navigation, session access, and casting UI. The refactor separates session ownership, access control, and participant management into distinct concerns while maintaining atomicity and data consistency.

## Changes Made

### Phase 1-2: Database Schema and Models

#### New Tables Created

1. **SessionAccessOrigin** (`session_access_origins`)
   - Tracks the source of access (direct or group-based)
   - Columns: id, session_id, user_id, source_type, group_id, created_at, updated_at
   - Unique constraint on (session_id, user_id, source_type, group_id)
   - Indexes on (session_id, user_id), (group_id), (source_type)
   - Purpose: Audit trail for how users gained access

2. **SessionGroupLink** (`session_group_links`)
   - Maps groups to sessions for access control
   - Columns: id, session_id, group_id, created_at, updated_at
   - Unique constraint on (session_id, group_id)
   - Purpose: Track which groups have access to which sessions

#### Modified Tables

1. **SessionAccess** (`session_access`)
   - Removed group_id column (now exclusively for direct user access)
   - Made user_id NOT NULL
   - Added unique constraint on (session_id, user_id)
   - Removed check constraint requiring one-or-the-other pattern
   - Now represents "effective access" only

2. **RoutineSession** (`routine_sessions`)
   - Added owner_id column (FK to users)
   - Removed group_id column
   - Backfilled owner_id from created_by
   - Added relationships to SessionAccessOrigin and SessionGroupLink

#### Migration File

- **006_refactor_session_access_to_separate_concerns.py**
  - Creates new tables with proper constraints and indexes
  - Backfills SessionAccessOrigin from existing SessionAccess records
  - Migrates group linkages to SessionGroupLink
  - Fully idempotent upgrade/downgrade

### Phase 3: Service Layer

#### SessionAccessService (`app/services/session_access_service.py`)

**Direct Access Management:**
- `grant_direct_access()` - Grant direct access to a user (atomic with origin tracking)
- `get_access()` - Retrieve effective access for a user
- `list_session_access()` - List all users with access to a session
- `revoke_direct_access()` - Remove direct access (with origin cleanup)

**Group Linkage Management:**
- `link_group_to_session()` - Link group and grant access to all members (atomic)
- `unlink_group_from_session()` - Unlink group and cleanup all member access
- `revoke_group_member_access()` - Remove specific group member's access
- `list_session_groups()` - List groups linked to a session

**Access Origins (Audit Trail):**
- `list_access_origins()` - Get all origins for a session
- `list_user_origins()` - Get all ways a user gained access

**Sync & Maintenance:**
- `sync_group_member_access()` - Sync access after group membership changes

#### RoutineSessionService Updates (`app/services/routine_session_service.py`)

**Session Creation:**
- `create_default_session()` - Auto-create default session for routine (with owner access)
- `create()` - Create named session with owner access

**Session Filtering:**
- `list_for_routine_filtered()` - Filter by user's archived/deleted state
- `get_default_session()` - Get first session created for a routine

**User State Management:**
- `archive_for_user()` - Archive session for user
- `unarchive_for_user()` - Unarchive session for user
- `mark_deleted_for_user()` - Soft-delete session for user
- `restore_for_user()` - Restore deleted session
- `get_user_state()` - Query user state for session

#### RoutinesService Updates (`app/services/routine_service.py`)

- `create_routine()` - Auto-creates routine + default session (atomic)
- Default session owner is the routine creator with admin access

### Phase 4: API Endpoints

#### Session Access Management (`app/api/v1/session_access.py`)

**Direct Access (Session Owner Only):**
- `GET /sessions/{session_id}/access` - List all access
- `POST /sessions/{session_id}/access/users` - Grant access
- `DELETE /sessions/{session_id}/access/users/{user_id}` - Revoke access

**Access Origins (Owner or Participant):**
- `GET /sessions/{session_id}/access/origins` - View all access sources
- `GET /sessions/{session_id}/access/users/{user_id}/origins` - View specific user's origins

**Group Linkage (Session Owner Only):**
- `GET /sessions/{session_id}/groups` - List linked groups
- `POST /sessions/{session_id}/groups` - Link group (grants member access)
- `DELETE /sessions/{session_id}/groups/{group_id}` - Unlink group
- `DELETE /sessions/{session_id}/access/groups/{group_id}/users/{user_id}` - Revoke member access

#### Session User State (`app/api/v1/session_user_state.py`)

**Archiving:**
- `POST /sessions/{session_id}/archive` - Archive for current user
- `POST /sessions/{session_id}/unarchive` - Unarchive for current user

**Deletion/Restoration:**
- `POST /sessions/{session_id}/delete` - Soft-delete for current user
- `POST /sessions/{session_id}/restore` - Restore deleted session

**Query:**
- `GET /me/sessions/state` - List current user's session states

#### Updated Routine Sessions (`app/api/v1/routine_sessions.py`)

**Changes:**
- Removed group_id from create request
- Removed `list_group_sessions()` endpoint (group sessions now via session_group_links)
- Updated delete to require session ownership
- Simplified documentation

### Phase 5: Authorization & Core Logic

#### Authorization Updates (`app/core/authorization.py`)

**New Functions:**
- `require_session_owner()` - Verify user owns the session
- `require_session_not_deleted()` - Check session not soft-deleted for user
- `require_session_visible_to_user()` - Combined visibility check

**Updated Functions:**
- `require_session_access()` - Now checks SessionAccess table instead of created_by/group_id
- Uses new SessionAccess model for effective access

### Phase 6: Data Models & Schemas

#### New Models

**SessionAccessOrigin** (`app/models/session_access_origin.py`)
```python
- id: UUID PK
- session_id: UUID FK
- user_id: UUID FK
- source_type: Enum('direct' | 'group')
- group_id: UUID FK (nullable)
- created_at, updated_at: DateTime
```

**SessionGroupLink** (`app/models/session_group_link.py`)
```python
- id: UUID PK
- session_id: UUID FK
- group_id: UUID FK
- created_at, updated_at: DateTime
```

#### Updated Models

**SessionAccess** (`app/models/session_access.py`)
- Removed user_id/group_id check constraint
- user_id now NOT NULL
- group_id removed entirely
- role now restricted to: admin, editor, viewer

**RoutineSession** (`app/models/routine_session.py`)
- Added owner_id column
- Removed group_id column
- Added relationships to access_origins and group_links

#### Schemas

**SessionAccessResponse** - Response for access records
**SessionAccessCreate** - Request to grant access
**SessionAccessOriginResponse** - Response for origin records
**SessionGroupLinkResponse** - Response for group links
**SessionGroupLinkCreate** - Request to link group
**RoutineSessionResponse** - Updated to include owner_id
**SessionUserStateResponse** - Per-user visibility state

### Phase 7: Integration

#### Main App Updates (`app/main.py`)

Added new routers:
- `session_access.router` - Access management endpoints
- `session_user_state.router` - Archiving/deletion endpoints

#### Exports

Updated `app/api/v1/__init__.py` and `app/models/__init__.py` to export new modules

## Key Design Principles

### 1. Separation of Concerns

- **SessionAccess**: "Who can access what" (effective permissions)
- **SessionAccessOrigin**: "How did they get access" (audit trail)
- **SessionGroupLink**: "Which groups have sessions" (configuration)
- **SessionUserState**: "User preferences for visibility"

### 2. Atomicity

All operations are atomic:
- Access grants include both SessionAccess and SessionAccessOrigin creation
- Group linkage creates links and syncs all member access
- Session creation auto-creates owner access

### 3. Cascade Delete Safety

- Removing origins only deletes SessionAccess if NO other origins remain
- Unlinking groups removes only group-derived origins, preserves direct access
- Removing users from groups only affects group-derived access

### 4. Authorization Model

- Session owner has full control (admin)
- Access participants can view session details
- Group-based access automatically syncs with group membership changes
- Soft deletion (per-user) separate from hard deletion (data destruction)

## Testing Recommendations

### Unit Tests for Services
- Direct access grant/revoke with origin tracking
- Group linkage with member sync
- User state transitions (archive/unarchive/delete/restore)
- Origin cleanup when multiple origins exist

### Integration Tests for API
- Complete flow: create routine → create session → grant access → revoke access
- Group workflow: create group → link to session → add member → remove member
- Archive/delete workflows with proper visibility

### Database Tests
- Unique constraints enforced
- Cascade deletes work correctly
- Backfill migration preserves existing data
- Indexes are present and performant

## Migration Path

1. Run migration 006 on database
2. Regenerate Flutter client models (SessionAccess, SessionAccessOrigin, etc.)
3. Update frontend to use new endpoints
4. Deploy backend with new routers
5. Monitor for any data inconsistencies

## Future Enhancements

- Participant role constraints (only dancer/coach in v1)
- Advanced role permissions (viewer/editor/admin nuances)
- Audit logging for access changes
- Batch operations for group access management
- Access expiration/time-limited sharing
