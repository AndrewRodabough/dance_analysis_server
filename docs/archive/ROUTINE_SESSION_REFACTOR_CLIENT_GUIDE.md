# Client Guide: Routine Session Refactor

This document describes the backend model changes for the **routine session refactor** and the corresponding client-side changes required. It is intended as an actionable reference for client implementation.

---

## 1) What Changed and Why

### Old model (remove from client assumptions)
- A `Routine` was directly tied to a single `group_id`.
- Videos and Notes belonged directly to a `Routine`.
- There was no concept of dancer slots, roles, or reuse of a routine across contexts.

### New model (client must adopt)
- **Routine** is now a pure choreography definition. It has no `group_id`. It can be reused across multiple groups and contexts.
- **RoutineSession** is the new instance layer. It binds a routine to a group (optionally) and owns all Videos and Notes.
- **RoutineDancerSlot** defines named dancer positions within a routine (e.g. "A", "B", "Lead", "Left 3"). Supports paired and formation dances.
- **SlotAssignment** binds a specific user to a dancer slot within a specific session.

### Mental model shift
```
Old:  Group → Routine → [Videos, Notes]
New:  Routine (definition, reusable)
        ├── RoutineDancerSlot[] (positions in the choreography)
        └── RoutineSession (instance, optionally group-scoped)
              ├── SlotAssignment[] (who dances where)
              ├── Video[]
              └── Note[]
```

---

## 2) Data Model Changes

### Routine (modified)
Fields **removed**:
- `group_id` — no longer present

Fields **kept** (unchanged):
- `id`, `title`, `dance_id`, `created_by`, `created_at`, `updated_at`

### RoutineSession (new)
| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | Primary key |
| `routine_id` | UUID | FK → routines.id (required) |
| `group_id` | UUID | FK → groups.id (nullable, SET NULL on group delete) |
| `created_by` | UUID | FK → users.id (required) |
| `label` | string(255) | Optional human-friendly label, e.g. "Practice with Alex" |
| `created_at` | datetime | |
| `updated_at` | datetime | |

### RoutineDancerSlot (new)
| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | Primary key |
| `routine_id` | UUID | FK → routines.id (required) |
| `label` | string(50) | Required, e.g. "A", "B", "1", "Left 3" |
| `order_index` | int | Optional ordering hint |
| `created_at` | datetime | |

Constraint: `(routine_id, label)` must be unique.

### SlotAssignment (new)
| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | Primary key |
| `routine_session_id` | UUID | FK → routine_sessions.id (required) |
| `dancer_slot_id` | UUID | FK → routine_dancer_slots.id (required) |
| `user_id` | UUID | FK → users.id (required) |
| `created_at` | datetime | |

Constraint: `(routine_session_id, dancer_slot_id)` must be unique (one user per slot per session).

### Video (modified)
- `routine_id` → **renamed to `routine_session_id`** (FK → routine_sessions.id)
- All other fields unchanged.

### Note (modified)
- `routine_id` → **renamed to `routine_session_id`** (FK → routine_sessions.id)
- All other fields unchanged.

---

## 3) API Contract Changes

All routes described in this section are **live**. Client work can begin.

### 3a) Routines (no longer group-scoped)

Routines are a top-level resource and are no longer owned by a group.

**Routes:**
- `POST /api/v1/routines` — Create a routine (creator becomes owner)
- `GET /api/v1/routines` — List routines created by the current user
- `GET /api/v1/routines/{routine_id}` — Get routine detail
- `PATCH /api/v1/routines/{routine_id}` — Update a routine
- `DELETE /api/v1/routines/{routine_id}` — Delete a routine

**Removed routes:**
- `POST /api/v1/groups/{group_id}/routines` — replaced by top-level create + session create
- `GET /api/v1/groups/{group_id}/routines` — replaced by listing sessions for a group

**Response schema change (`RoutineResponse`):**
- `group_id` field **removed**
- All other fields unchanged

### 3b) Routine Sessions (new resource)

Sessions are the group-scoped instance of a routine.

**Routes:**
- `POST /api/v1/routines/{routine_id}/sessions` — Create a session for a routine
  - Request body: `{ group_id?: UUID, label?: string }`
- `GET /api/v1/routines/{routine_id}/sessions` — List sessions for a routine
- `GET /api/v1/groups/{group_id}/sessions` — List sessions within a group (replaces old group routine list)
- `GET /api/v1/sessions/{session_id}` — Get session detail
- `DELETE /api/v1/sessions/{session_id}` — Delete a session

**New response schema (`RoutineSessionResponse`):**
```json
{
  "id": "uuid",
  "routine_id": "uuid",
  "group_id": "uuid | null",
  "created_by": "uuid",
  "label": "string | null",
  "created_at": "datetime",
  "updated_at": "datetime | null"
}
```

### 3c) Dancer Slots (new resource)

**Routes:**
- `POST /api/v1/routines/{routine_id}/slots` — Create a dancer slot
  - Request body: `{ label: string, order_index?: int }`
- `GET /api/v1/routines/{routine_id}/slots` — List slots for a routine
- `DELETE /api/v1/routines/{routine_id}/slots/{slot_id}` — Delete a slot

**New response schema (`RoutineDancerSlotResponse`):**
```json
{
  "id": "uuid",
  "routine_id": "uuid",
  "label": "string",
  "order_index": "int | null",
  "created_at": "datetime"
}
```

### 3d) Slot Assignments (new resource)

**Routes:**
- `POST /api/v1/sessions/{session_id}/assignments` — Assign a user to a slot
  - Request body: `{ dancer_slot_id: UUID, user_id: UUID }`
- `GET /api/v1/sessions/{session_id}/assignments` — List assignments for a session
- `DELETE /api/v1/sessions/{session_id}/assignments/{assignment_id}` — Remove an assignment

**New response schema (`SlotAssignmentResponse`):**
```json
{
  "id": "uuid",
  "routine_session_id": "uuid",
  "dancer_slot_id": "uuid",
  "user_id": "uuid",
  "created_at": "datetime"
}
```

### 3e) Videos and Notes (re-scoped to sessions)

Videos and Notes now belong to a session, not a routine directly.

**Old URL pattern:**
```
/api/v1/groups/{group_id}/routines/{routine_id}/videos
/api/v1/groups/{group_id}/routines/{routine_id}/notes
```

**New URL pattern:**
```
/api/v1/sessions/{session_id}/videos
/api/v1/sessions/{session_id}/videos/{video_id}
/api/v1/sessions/{session_id}/videos/{video_id}/finalize
/api/v1/sessions/{session_id}/videos/{video_id}/download
/api/v1/sessions/{session_id}/notes
/api/v1/sessions/{session_id}/notes/{note_id}
/api/v1/sessions/{session_id}/videos/{video_id}/notes
```

**Response schema changes:**
- `VideoResponse`: `routine_id` → `routine_session_id`
- `NoteResponse`: `routine_id` → `routine_session_id`

---

## 4) Client Implementation Impact

### 4a) Files that need changes

**Data sources (API calls):**
- `routines_data_source.dart` — Remove `groupId` from all method signatures. Routines are no longer group-scoped. Add new methods for routine CRUD against top-level routes.
- `videos_data_source.dart` — Replace `(groupId, routineId)` parameters with `sessionId`. All video operations scope to a session.
- `notes_data_source.dart` — Replace `(groupId, routineId)` parameters with `sessionId`. All note operations scope to a session.

**New data sources to create:**
- `routine_sessions_data_source.dart` — CRUD for sessions (create, list by routine, list by group, delete).
- `dancer_slots_data_source.dart` — CRUD for dancer slots on a routine.
- `slot_assignments_data_source.dart` — CRUD for slot assignments on a session.

**Controllers:**
- `routines_controller.dart` — Remove `groupId` dependency. Add session management.
- `videos_controller.dart` — Replace `(groupId, routineId)` with `sessionId`.
- `notes_controller.dart` — Replace `(groupId, routineId)` with `sessionId`.

**New controllers to create:**
- `routine_sessions_controller.dart` — Session state management.
- `dancer_slots_controller.dart` — Slot management for routine setup.
- `slot_assignments_controller.dart` — Assignment management per session.

**State classes:**
- `routines_state.dart` — `RoutineResponse` no longer has `group_id`.
- New state classes for sessions, slots, assignments.

**UI / Pages:**
- `routine_detail_page.dart` — Currently receives `groupId` + `routineId`. Must be refactored to work with a `sessionId`. The Videos and Notes tabs load from a session, not a routine directly.
- `create_routine_dialog.dart` — Routine creation no longer requires a group context. Session creation (binding to a group) is a separate step.
- `routine_card.dart` — May need updates if it displayed group info.
- Navigation routes that include `/groups/{groupId}/routines/{routineId}` need to change to reflect the session-based hierarchy.

### 4b) Generated API models

The client uses generated API models from `generated/api/`. After the backend OpenAPI spec is updated, regenerate:
- `RoutineResponse` — `group_id` field removed
- `VideoResponse` — `routine_id` → `routine_session_id`
- `NoteResponse` — `routine_id` → `routine_session_id`
- New models: `RoutineSessionResponse`, `RoutineDancerSlotResponse`, `SlotAssignmentResponse`

### 4c) Navigation changes

**Current route structure:**
```
/groups/{groupId}/routines                    → routine list
/groups/{groupId}/routines/{routineId}        → routine detail (videos + notes tabs)
/groups/{groupId}/routines/{routineId}/upload  → video upload
/groups/{groupId}/routines/{routineId}/videos/{videoId} → video detail
```

**Proposed new route structure:**
```
/routines                                      → user's routine list
/routines/{routineId}                          → routine detail (slots, sessions overview)
/groups/{groupId}/sessions                     → sessions in a group (replaces old routine list)
/sessions/{sessionId}                          → session detail (videos + notes tabs)
/sessions/{sessionId}/upload                   → video upload
/sessions/{sessionId}/videos/{videoId}         → video detail
```

---

## 5) Migration Path (Client)

This is a **breaking change** to the API contract. The backend is complete — client work can begin now.

1. **Regenerate API client** — Regenerate the Dart API client from the updated OpenAPI spec to pick up new models and endpoints.
2. **Refactor data sources** — Update existing data sources and create new ones for sessions/slots/assignments.
3. **Refactor controllers and state** — Update to use session-based operations.
4. **Refactor UI** — Update pages, widgets, and navigation to reflect the session-based hierarchy.
5. **Test** — Verify all existing flows still work through the new session layer.

---

## 6) What is NOT Changing Yet

- **No time-based partner switching** — Slot assignments are static per session for now.
- **No role system (Leader/Follower)** — Slots have labels, not typed roles.
- **No choreography JSON changes** — The routine definition structure is unchanged.
- **Auth and group membership** — Unchanged. Authorization still flows through group membership; sessions inherit group access.
- **Video upload flow** — Still register → PUT → finalize. Only the scoping changes (session instead of routine).
- **Note types and structure** — `NoteType`, `NoteSource`, `details` JSON, `video_timestamp_ms` all unchanged.
