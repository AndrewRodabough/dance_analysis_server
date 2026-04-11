# Refactor Plan: Routine-Centric Navigation, Session Access, and Casting UI

## Overview

The app is being refactored to make **routines** the primary user-facing concept
while treating **groups** as a supporting access and organization mechanism
rather than a primary navigation destination.

This plan also introduces a **casting UI** inside the routine detail page so
users with session access can be assigned to cast roles. In the backend model,
the session is split into separate concepts:

- **Session content** — the routine session itself
- **Session access** — who can see or manage the session
- **Session participants** — who is actually cast into roles
- **Per-user session state** — archive / delete visibility for each user

The frontend must align with that model.

---

## Backend Dependencies

These backend changes must be available before the dependent frontend work lands.

| Change                                                                         | Affects                              |
| ------------------------------------------------------------------------------ | ------------------------------------ |
| `RoutineSession.owner_id` added; `routine_sessions.group_id` removed           | Session ownership / access model     |
| New `SessionAccess` table with `user_id` and access roles                      | Session sharing and permissions      |
| New `SessionAccessOrigin` table for direct/group origin tracking               | Derived access sync                  |
| New `SessionParticipant` table with participant roles and optional dancer slot | Casting UI                           |
| New `SessionUserState` table for per-user archive/delete visibility            | Routines list and session visibility |
| `SessionAccess` seeded for session owner as admin                              | Session creation and management      |
| Group membership / inferred group sync for session access                      | Manage groups and shared access      |
| Regenerate Flutter models and clients                                          | All frontend work                    |

### Important model implications

- A session no longer “belongs to a group”.
- A session may grant access to:
  - individual users
  - groups
- A user must already have access before they can become a participant.
- A participant is a user with access who has been assigned a cast role.
- One user may have at most one participant record per session.
- Participant roles are intentionally expandable in the future.
- Groups still exist as a first-class destination for managing membership and
  promotion of inferred groups.
- Access must be tracked as:
  - a unique `SessionAccess` row per `(session_id, user_id)`
  - one or more `SessionAccessOrigin` rows describing why that user has access

---

## Workstream A — Navigation Refactor

### Goal

Remove Groups from primary navigation and present a unified routines experience.

The main routines list should show:

- routines owned by the current user
- sessions the current user can access
- archive-aware visibility state

Groups remain accessible, but they should no longer be a primary tab.

### A1 — Remove Groups from primary navigation

**File:** `lib/config/routes.dart`

- Remove the `Groups` entry from the bottom navigation items.
- Remove the `Groups` bottom bar item from the mobile scaffold.
- Update path-to-tab mapping so `/groups` no longer maps to a primary tab.
- Update tab switching so profile becomes the third bottom-nav item.
- Keep `/groups/:groupId` reachable for deep links and internal navigation.
- Keep `/groups` as a destination, but it should be accessed through in-app
  navigation rather than primary bottom navigation.

### A2 — Unified routines list

**File:** `lib/features/routines/presentation/pages/routines_list_page.dart`

The routines page should become a unified feed of:

- owned routines
- accessible sessions

#### Desired behavior

- The page title becomes **Routines**
- The feed is filtered and grouped by access source
- Archived sessions remain available but are visually nested or labeled as archived
- Deleted sessions are hidden from the user

#### Feed content

The feed should include:

- owned routines
- accessible sessions created by or shared with the user
- sessions shared via individual access
- sessions shared via group access, expanded into user-level visibility

#### Suggested item model

Use a typed feed model rather than string sentinels.

Suggested filter / grouping categories:

- `All`
- `Owned`
- `Shared`
- `Archived`
- `Group`

If the implementation needs a local filter representation, prefer an enum or
sealed class over magic strings.

#### Data loading

The routines feed should be owned by a dedicated controller or aggregation layer.

Recommended loading sequence:

1. Load owned routines
2. Load accessible sessions
3. Load groups used for labeling / grouping
4. Merge and sort into a single feed
5. Apply per-user visibility state

If one source fails, the page should degrade gracefully rather than fail
completely.

#### Navigation on tap

- Owned routine → routine detail page
- Accessible session → session detail page

#### Filter chips

Examples:

- All
- Owned
- Shared
- Archived
- Group A
- Group B

The exact chip set can evolve, but the state model should support:

- owned routines
- direct shared sessions
- group-shared sessions
- archived sessions

### A3 — Session creation from routine context

Creating a session should happen from the routine detail page.

**Behavior**

- A routine should always have a default session created with the routine
- The session owner is the creating user
- The owner is seeded into `SessionAccess` as admin
- Additional access can be added later via sharing

This means session creation is no longer tied to a group as the primary object.

### A4 — Access-sharing from groups

Groups can still be used to seed session access, but they should not be
treated as session ownership.

#### Current rule

When a session is shared with a group:

- each group member receives session access
- the access is represented as user-level `SessionAccess` rows
- access source must be tracked in `SessionAccessOrigin`
- the session should maintain a link to the group so access can be updated
  when the group changes

#### Access origin rules

`SessionAccess` is the unique effective access row:

- `(session_id, user_id)` is unique
- `SessionAccess.role` stores the effective access role

`SessionAccessOrigin` stores why access exists:

- `session_id`
- `user_id`
- `source_type` = `direct` or `group`
- `group_id` nullable

#### Group membership changes

When a user is removed from a group:

- remove only the matching `SessionAccessOrigin` row for that group
- if no origins remain for that `(session_id, user_id)`, delete `SessionAccess`
- never delete access directly based on group removal alone

When a user is added to a group that is linked to a session:

- their derived session access origin should be created automatically
- if the user already has direct access, add a group origin row but do not
  duplicate `SessionAccess`

#### Important note

This means group-based access is a source of derived access, not a replacement
for user-level access state.

---

## Workstream B — Casting UI in Routine Detail

### Goal

Replace the old group-member casting model with a session-access-driven casting
UI.

The casting sheet should be driven by:

- users who have access to the session
- participant records created when those users are assigned to a role

### Core behavior

- Only users with session access can be assigned to roles
- Dragging an accessible user into a role creates a participant record
- Removing a user from a role deletes the participant record
- A user can only occupy one participant role per session
- Participant creation and slot binding must be atomic
- A participant record is the source of truth for casting assignment

### B1 — Data required by the routine detail page

The routine detail page will need:

- routine details
- default session
- session access list
- participant list
- dancer slots
- videos
- notes

The data should be orchestrated so the page does not contain a pile of
unrelated fetch logic.

Recommended controller scope:

- routine detail state
- session state
- access / participant state

If a single controller is too large, split by concern but keep the page-level
orchestration clear.

### B2 — Replace `_DancersSection` with `_CastingSection`

#### Intended layout

```text
Access
  Alex (admin)
  Jordan
  Sam

Admins
  Alex

Roles
  ┌─ Dancer 1 ──────── [ Riley      ✕ ] ─── ⠿ ┐
  ├─ Dancer 2 ──────── [ drag here    ] ─── ⠿ ┤
  └─ Dancer 3 ──────── [ drag here    ] ─── ⠿ ┘

Available
  ⠿ Sam   ⠿ Taylor
```

#### Access section

This section shows all session-access users.

Rules:

- all accessible users appear here
- archived/deleted users are hidden from the source list
- admins are shown here and also surfaced in routine details
- admins are allowed to be assigned to casting roles

#### Admin summary

A dedicated session details field should show the admin user.

Admins are not excluded from casting roles.

#### Roles section

This section shows the available casting roles / slots.

Rules:

- a role row can accept a draggable accessible user
- a role row displays the assigned participant if present
- a role row displays a placeholder if empty
- unassigning deletes the participant record
- each user may appear in only one participant slot per session

#### Available section

This section shows accessible users who do not yet have a participant record for
the session.

Rules:

- if a user already has a participant record, they are excluded
- archived / deleted users are excluded
- admins remain eligible for assignment

### B3 — Participant creation on drag-and-drop

The drag-and-drop interaction should create the participant behind the scenes.

Expected flow:

1. user drags accessible user to a role
2. frontend calls backend to create participant
3. participant creation also binds the role / slot
4. UI refreshes from backend response

The user should not be shown participant objects directly.

### B4 — Participant and slot behavior

The backend should treat the participant record as the source of truth for role
placement.

Recommended participant shape:

- `session_id`
- `user_id`
- `role`
- `dancer_slot_id`

Rules:

- one participant per user per session
- one user per slot per session
- slot binding is stored on the participant record
- unassigning deletes the participant record
- no separate slot assignment table is needed if participant owns the slot binding

If participant creation succeeds:

- infer the slot assignment from the participant data
- render that user in the correct slot on reload

If the participant creation fails:

- no cast changes should be shown
- the UI should treat the drag as a no-op

There are no bulk update endpoints yet, so the frontend should assume a single
drag action maps to a single backend transaction.

### B5 — Reordering slots

The app still needs slot reordering support.

`DancerSlotsController` should gain a method for updating slot order.

Important implementation notes:

- slot IDs must remain stable across reorder operations
- reorder should only affect display order / `orderIndex`
- participant assignment should remain tied to the slot identity, not the
  visual position

### B6 — Session management sheet

The session detail page should expose a management sheet for session-level access
and sharing.

This sheet should support:

- viewing current session access
- inviting users
- assigning or revoking access
- showing access source when possible

It should remain loosely coupled from group structure.

#### Access behavior

- inviting users creates individual `SessionAccess`
- sharing with a group adds group-derived access for each group member
- adding a group to a session should keep a `SessionGroupLink` so access can be
  updated when group membership changes

#### Group linkage

Add a concrete link table:

`SessionGroupLink`

- `session_id`
- `group_id`

This is required for:

- syncing group-derived access
- UI display such as “shared with group X”
- recomputing access when group membership changes

#### Group behavior

Groups remain a separate domain:

- explicit groups are user-created and managed
- inferred groups are system-suggested collaborator clusters
- inferred groups may be promoted before users can manage them directly

Key rule:

- sessions do not create or mutate groups as a primary data operation
- sessions can reference groups for access synchronization only

#### File to create

`lib/features/groups/presentation/widgets/manage_group_sheet.dart`

This may evolve into a more session-centric management sheet, but the reusable
structure should be extracted now.

### B7 — Manage Groups destination

Groups still need a dedicated destination so users can manage membership and
inferred groups.

This destination should support:

- viewing explicit groups
- viewing inferred groups
- promoting inferred groups to explicit groups
- adding users to groups
- removing users from groups
- assigning or updating group roles
- using groups as a source for session access

The group management destination should not be treated as the primary navigation
tab, but it must remain reachable from:

- routine detail session management
- deep links
- group-specific navigation flows

---

## Workstream C — Routine and Session Removal

### Goal

Support owner and non-owner visibility actions in a session-aware way.

### C1 — Owner routine deletion

For the routine owner:

- show a Delete action in the routine detail page overflow menu
- confirm before deletion
- deleting a routine permanently removes the routine and its associated access
  context

Calls:

- `RoutinesController.deleteRoutine(routineId)`

### C2 — Session archiving and deletion for current user

Because per-user session state now exists:

- sessions can be archived for the current user without removing them for everyone
- deleted sessions should be hidden for that user

The UI should distinguish:

- active
- archived
- deleted

Archived sessions should remain accessible through an archived section or label.

### C3 — Non-owner access removal

For users who are not the routine owner but have session access:

- remove or revoke access as appropriate
- if access was group-derived, remove only the matching origin row
- if no origins remain, delete the effective access row
- if access was direct, revoke the direct access origin row
- never delete access directly based on group removal alone

The exact semantics should match the backend’s access model.

---

## Backend API and Schema Checklist

Use this as the backend implementation checklist for the session refactor.

### 1. Session access ownership and origin tracking

#### Schema

- `routine_sessions`
  - add `owner_id`
  - remove `group_id`
- `session_access`
  - columns:
    - `id`
    - `session_id`
    - `user_id`
    - `role`
    - timestamps
  - unique constraint:
    - `(session_id, user_id)`
- `session_access_origins`
  - columns:
    - `id`
    - `session_id`
    - `user_id`
    - `source_type` (`direct` | `group`)
    - `group_id` nullable
    - timestamps
  - at minimum, index:
    - `(session_id, user_id)`
    - `(group_id)`
    - `(source_type)`
- `session_group_links`
  - columns:
    - `id`
    - `session_id`
    - `group_id`
    - timestamps
  - unique constraint:
    - `(session_id, group_id)`

#### Endpoints

- `GET /sessions/{sessionId}/access`
  - returns effective access rows and origin metadata
- `POST /sessions/{sessionId}/access/users`
  - create direct user access
- `POST /sessions/{sessionId}/access/groups`
  - create group-derived access
- `DELETE /sessions/{sessionId}/access/users/{userId}`
  - remove direct access origin
- `DELETE /sessions/{sessionId}/access/groups/{groupId}/users/{userId}`
  - remove a specific group-derived access origin
- `GET /sessions/{sessionId}/groups`
  - list linked groups for the session
- `POST /sessions/{sessionId}/groups`
  - link a group to a session
- `DELETE /sessions/{sessionId}/groups/{groupId}`
  - unlink a group from a session

#### Required behavior

- `SessionAccess` must remain the unique effective access row for `(session_id, user_id)`
- `SessionAccessOrigin` rows determine why access exists
- removing a group member should delete only the matching group origin row
- if a user has no remaining origins, delete the effective `SessionAccess` row
- never delete effective access directly based on group removal alone

---

### 2. Session participants and casting state

#### Schema

- `session_participants`
  - columns:
    - `id`
    - `session_id`
    - `user_id`
    - `role`
    - `dancer_slot_id`
    - timestamps
  - unique constraint:
    - `(session_id, user_id)`
  - unique constraint:
    - `(session_id, dancer_slot_id)` if one slot can only have one participant
- no separate slot assignment table is needed if participant owns the slot binding

#### Endpoints

- `GET /sessions/{sessionId}/participants`
  - list participants for a session
- `POST /sessions/{sessionId}/participants`
  - create participant
  - assign role
  - optionally bind `dancer_slot_id`
- `DELETE /sessions/{sessionId}/participants/{participantId}`
  - unassign participant
- `PATCH /sessions/{sessionId}/participants/{participantId}`
  - update participant role and/or slot binding if needed later

#### Required behavior

- the create endpoint must be atomic
- the create endpoint must enforce:
  - one participant per user per session
  - one user per slot per session
- drag-and-drop casting should call a single backend operation that creates the participant and binds the slot
- if the create operation fails, no cast state should be partially applied

#### Role constraints

- v1 participant roles:
  - `dancer`
  - `coach`
- do not expose `other` in the frontend until its migration is ready
- the database enum or constraint should enforce the v1 role set

---

### 3. Session user state and visibility

#### Schema

- `session_user_states`
  - columns:
    - `id`
    - `session_id`
    - `user_id`
    - `is_archived`
    - `is_deleted`
    - timestamps
  - unique constraint:
    - `(session_id, user_id)`

#### Endpoints

- `GET /me/session-state`
  - list the current user’s session state
- `POST /sessions/{sessionId}/archive`
  - archive the session for the current user
- `POST /sessions/{sessionId}/unarchive`
  - restore the current user’s archived session
- `POST /sessions/{sessionId}/delete`
  - mark the session deleted for the current user
- `POST /sessions/{sessionId}/restore`
  - restore a deleted session if supported

#### Required behavior

- `is_deleted = true` sessions must be excluded from normal lists
- archived sessions must remain visible, but can be grouped or labeled separately
- the backend should consistently apply the visibility rule at query time

---

### 4. Routine creation and default session invariant

#### Schema / behavior

- creating a routine must also create its default session
- the default session owner must be the creating user
- the owner must receive an admin `SessionAccess` row
- the default session must exist before the routine detail page renders session-dependent UI

#### Endpoints

- no separate session bootstrap endpoint is required if routine creation always creates the default session
- if a bootstrap endpoint exists, it should be internal-only or admin-only and not part of the normal UI flow

#### Required behavior

- the frontend should never need to handle “routine exists but no session exists”
- videos, notes, casting, and access management should all assume a default session exists

---

### 5. Group membership, inferred groups, and promotion

#### Schema

- group membership tables should continue to support:
  - member role updates
  - inferred vs explicit group status
- if session sharing via group exists, the linkage must be preserved through `session_group_links`

#### Endpoints

- `GET /groups`
- `GET /groups/{groupId}`
- `GET /groups/{groupId}/members`
- `POST /groups/{groupId}/members`
- `DELETE /groups/{groupId}/members/{userId}`
- `PATCH /groups/{groupId}/members/{userId}`
- `POST /groups/{groupId}/promote`
  - promote inferred group to explicit group
- any additional group role endpoints needed to manage membership roles cleanly

#### Required behavior

- group management remains a distinct destination from session management
- group membership changes must trigger derived session access synchronization when the group is linked to a session

---

### 6. Query rules for routines and sessions

#### Required behavior

- session lists must exclude deleted sessions for the current user
- archived sessions may be included, but should be flagged so the client can nest or label them
- access queries should return both:
  - effective access
  - origin metadata
- available-users logic on the client should be based on:
  - users with access
  - minus users with a participant record
  - minus users with deleted state

#### Suggested response fields

- `SessionAccess`
  - `session_id`
  - `user_id`
  - `role`
  - timestamps
- `SessionAccessOrigin`
  - `session_id`
  - `user_id`
  - `source_type`
  - `group_id`
- `SessionParticipant`
  - `session_id`
  - `user_id`
  - `role`
  - `dancer_slot_id`
- `SessionUserState`
  - `session_id`
  - `user_id`
  - `is_archived`
  - `is_deleted`

---

### 7. Migration and backfill rules

#### Required behavior

- backfill `owner_id` from `created_by`
- seed owner `SessionAccess` records as admin
- create `SessionAccessOrigin` rows for any direct access that exists at migration time
- create `SessionGroupLink` rows if existing data implies group-based access
- do not auto-create participants during migration
- do not auto-create user state rows unless needed for backfill consistency

---

### 8. Validation checklist

#### Database

- `routine_sessions.owner_id` populated for all existing rows
- `routine_sessions.group_id` removed
- `session_access` unique on `(session_id, user_id)`
- `session_access_origins` exists and is queryable
- `session_group_links` exists and is queryable
- `session_participants` unique on `(session_id, user_id)`
- `session_user_states` unique on `(session_id, user_id)`
- role enum only allows v1 values

#### API

- group-derived access can be added and removed without touching direct access
- participant creation + slot binding is atomic
- session lists respect archived/deleted visibility
- routine creation always yields a default session
- group promotion endpoints work for inferred groups

---

## Build Order

```
1. Backend: session ownership, access, access origins, participants, and per-user state
2. Regenerate Flutter models and clients
3. A1  Remove Groups from primary navigation
4. A2  Unified routines list + feed controller
5. A3  Session creation from routine context
6. A4  Group-derived session access handling
7. B1  Scope session / access / participant state to routine detail
8. B2  Casting section UI
9. B3  Participant creation on drag-and-drop
10. B4 Participant and slot behavior
11. B5 Slot reordering
12. B6 Session management sheet
13. B7 Manage Groups destination
14. C   Delete / archive / revoke access flows
```

---

## Files Touched

| File                                                                      | Change                                                |
| ------------------------------------------------------------------------- | ----------------------------------------------------- |
| `lib/config/routes.dart`                                                  | Remove Groups tab from primary navigation             |
| `lib/features/routines/presentation/pages/routines_list_page.dart`        | Unified routines feed with filters and archived state |
| `lib/features/routines/presentation/controllers/routines_controller.dart` | Feed aggregation or new controller                    |
| `lib/features/routines/presentation/pages/routine_detail_page.dart`       | Session-aware detail orchestration and casting UI     |
| `lib/features/groups/presentation/pages/group_detail_page.dart`           | Access seeding / group-based sharing changes          |
| `lib/features/groups/presentation/widgets/manage_group_sheet.dart`        | New session/group management sheet                    |
| `lib/generated/api/models/routine_session_response.dart`                  | Regenerated session model                             |
| `lib/generated/api/models/session_access_response.dart`                   | Regenerated access model                              |
| `lib/generated/api/models/session_access_origin_response.dart`            | Regenerated access origin model                       |
| `lib/generated/api/models/session_participant_response.dart`              | Regenerated participant model                         |
| `lib/generated/api/models/session_user_state_response.dart`               | Regenerated per-user state model                      |
| `lib/generated/api/models/routine_session_create.dart`                    | Regenerated creation model                            |

---

## Open Questions / Assumptions

These items should be confirmed before implementation is finalized.

### 1. Participant creation semantics

The current assumption is:

- drag accessible user to role
- backend creates participant record
- participant record ties the user to the role / slot
- unassigning deletes the participant record

### 2. Available roles

Current role set:

- dancer
- coach

The plan assumes:

- only dancer and coach are used initially
- `other` exists in the broader model for future use
- the UI should not expose `other` until the backend migration for it is ready

### 3. Admin handling

Admins should:

- appear in the session access list
- be shown explicitly in routine/session details
- remain eligible for casting roles
- have management permissions for access and invites

### 4. Group access synchronization

If a group is linked to a session:

- adding or removing group members should sync derived session access
- direct access must not be removed accidentally
- group-derived access should be distinguishable from direct access

### 5. Archived sessions

Archived sessions should:

- remain visible in a nested or labeled archived view
- not disappear entirely from the user’s routines area
- be treated separately from deleted sessions

### 6. Default session creation

The routine owner should get a default session at routine creation time.
That session should:

- belong to the creating user as owner
- seed admin access for that user
- be the primary target for casting and notes/videos

### 7. Assignment and reorder failure behavior

There are no bulk update endpoints yet, so the frontend should assume:

- each drag-and-drop results in one backend operation
- failure should not partially mutate visible state
- the UI should refresh from backend state after each mutation
