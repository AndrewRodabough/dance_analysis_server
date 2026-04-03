# Routine Session Refactor Assessment Plan

## Objective

Assess the current backend implementation against the proposed refactor so you can identify:

- what already matches the proposal
- what must change
- what is still missing

## Current implementation status

### Already correct

- `RoutineSession` exists as a separate table from `routines`
- `RoutineSession` already stores:
  - `id`
  - `routine_id`
  - `created_by`
  - timestamps
- `routine_sessions` are created independently from routines
- `RoutineDancerSlot` exists as a separate table
- `SlotAssignment` exists as a separate table
- `Note` and `Video` have already been moved to `routine_session_id`
- `Routine` no longer carries `group_id` in the migration path
- The refactor already separates routine content from some session-specific data

### Needs to change

- `RoutineSession` still contains `group_id`
- `RoutineSession` still contains `label`, which is not in the proposed core model
- There is no `SessionAccess` model yet
- There is no `SessionUserState` model yet
- Access control is still not modeled as the only permission layer
- Participation is still represented by `SlotAssignment`, not the proposed `SessionParticipant`
- Group-scoped behavior still exists in the session model
- Ownership behavior when the creator leaves is not implemented as a distinct ownership/admin rule
- Per-user archive/delete is not implemented as separate user state rows
- The proposed constraints are not enforced:
  - `SessionAccess` must allow exactly one of `user_id` or `group_id`
  - `SessionParticipant` must be separate from permissions
  - `SessionUserState` must be unique per `(session_id, user_id)`
- Existing service logic still assumes `group_id` on `RoutineSession`

## Implementation gaps by proposed model

### 1. `RoutineSession` content-only model

Current backend still stores `group_id` on `routine_sessions`.

Required change:

- remove `group_id` from `RoutineSession`
- keep only content-related session fields
- keep ownership fields only if they align with the new ownership model

### 2. `SessionAccess`

Missing entirely.

Required change:

- add new model and table
- enforce one and only one of `user_id` or `group_id`
- define role enum: `viewer`, `editor`, `admin`
- make owner implied or explicitly seeded as admin access

### 3. `SessionParticipant`

Missing entirely.

Required change:

- add participant table separate from access control
- allow participation roles such as `dancer` and `coach`
- optionally attach `dancer_slot_id`
- prevent permission semantics from leaking into participation semantics

### 4. `SessionUserState`

Missing entirely.

Required change:

- add per-user session state table
- support `is_archived` and `is_deleted`
- enforce one row per session/user pair

## Code areas that will need updates

- ORM models for sessions and related tables
- Alembic migration for the session refactor
- services that create, list, or delete sessions
- query logic that filters sessions by group membership
- any authorization code that currently depends on `routine_sessions.group_id`
- any API schemas that expose group-scoped session data

## Recommended implementation order

1. Introduce `SessionAccess`
2. Introduce `SessionParticipant`
3. Introduce `SessionUserState`
4. Remove `group_id` from `RoutineSession`
5. Update service and API logic to use access rows instead of session group ownership
6. Add ownership/admin seeding rules
7. Add archive/delete behavior per user
8. Verify all existing session, video, and note flows still work

## Validation checklist

- `RoutineSession` has no `group_id`
- `SessionAccess` exists and enforces exactly one target subject
- group-based access works without duplicating sessions
- group membership changes do not require session updates
- creator leaving does not orphan session ownership
- user archive/delete only affects that user's visibility
- participation is independent from access

## Outcome

Use this plan to drive the next backend migration and model refactor so the implementation matches the proposed separation of content, access control, and participation.
