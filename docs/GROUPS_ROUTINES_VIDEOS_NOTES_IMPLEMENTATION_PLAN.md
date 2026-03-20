# Groups/Routines/Videos/Notes Implementation Plan (v0.0.1 Refactor)

## Agent Implementation Guide (How to Execute This Plan)

This section restructures the plan into discrete, agent-friendly chunks (phases) with:

- clear scope boundaries
- ordered work items
- concrete deliverables
- verification steps (tests + checks)

Implementation invariants (do not violate):

- Privacy is a primary functional requirement (see below).
- Group membership governs routine access.
- Pending uploads are uploader-private by default.
- Jobs and job artifacts are private to the job owner.
- Accepting invites requires strict email match (token forwarding must not grant access).

---

## Phase 0 — Guardrails (Policy + Conventions)

### Deliverables

- A written, enforced policy for non-leaky error handling (401 vs 404 vs 403).
- A single authorization seam used by all new endpoints.
- A clear router boundary: routine video library vs job artifact downloads.

### Verification

- FastAPI app boots and routers mount without collisions.
- A minimal test asserts:
  - unauthenticated returns 401 for protected endpoints
  - authenticated but unauthorized returns 404 for group-scoped resources
  - authenticated job non-owner returns 404 for job resources/artifacts

---

## Privacy & Youth Safety (Top Priority)

- This product may be used by youth teams; assume some group members may be minors.
- Preventing PII and video content leakage is a top priority.
- Security posture for v0.0.1:
  - Default-deny access: all group/routine/video/note resources require authorization checks.
  - Avoid information leakage in errors/responses (prefer 404 for non-accessible resources).
  - Pending uploads and analysis jobs are private by default (uploader/job owner).
  - Invite acceptance is locked down to strict email match (token forwarding must not grant access).

### Non-leaky Error Handling Policy (404 vs 403) — Apply Consistently and Test It

Goal: do not leak whether a group/routine/video/note/job exists to callers who should not know.

General rules:

- **401 Unauthorized**: request is not authenticated (missing/invalid auth).
- **404 Not Found**: request is authenticated but caller lacks access OR resource doesn’t exist OR resource is soft-deleted.
  - This is the default for group-scoped and privacy-sensitive resources.
- **403 Forbidden**: use only when the caller is already authorized to know the resource exists (generally avoid in v0.0.1).
- **409 Conflict**: avoid for deleted/not-uploaded content if it leaks status; prefer 404. (Reserve 409 for true validation/state conflicts that do not reveal existence across security boundaries.)

Resource-specific policy (to implement in services/routers and to assert in tests):

1. Groups (+ membership)

- `GET /groups`: returns only caller’s groups (200).
- `GET /groups/{group_id}`: **404** if not a member or group doesn’t exist.
- `GET /groups/{group_id}/members`: **404** if not a member or group doesn’t exist.
- Membership mutations (`POST/DELETE members`):
  - **404** if not a member or group doesn’t exist.
  - If later you enforce owner-only: still return **404** for non-owner members to avoid leaking group internals (consistent non-leaky posture).

2. Group invites (email-first)

- `POST /groups/{group_id}/invites`: **404** if not a member (or lacks capability later) or group doesn’t exist.
- `POST /group-invites/accept` (or similar token-based accept endpoint):
  - Return a generic **404** for any failure that could leak information:
    - token not found
    - token expired/revoked/already accepted
    - accepting user email does not match invite email
  - Rationale: token forwarding must not help an attacker discover a valid invite exists for another email.

3. Routines

- All routine endpoints are group-scoped:
  - `GET/POST /groups/{group_id}/routines...`: **404** if not a group member or group doesn’t exist.
  - `GET/PATCH/DELETE /groups/{group_id}/routines/{routine_id}`: **404** if routine not in group OR group not accessible OR routine doesn’t exist.

4. Routine videos (upload library)

- Default list:
  - `GET /groups/{group_id}/routines/{routine_id}/videos`: **404** if not a group member or routine not in group.
  - Returns only `status=uploaded` by default (privacy).
- Pending uploads visibility:
  - `GET .../videos?status=pending_upload`: if caller is not group member -> **404**; if group member but not uploader -> return **404** (do not leak that pending uploads exist).
- Direct access endpoints:
  - `GET .../videos/{video_id}`
  - `POST .../videos/{video_id}/finalize`
  - `GET .../videos/{video_id}/download`
  - `DELETE .../videos/{video_id}`
    Must return **404** if ANY are true:
  - caller not in group
  - routine/video mismatch
  - video does not exist
  - `status=deleted`
  - (for finalize) caller is not uploader
  - (for download) `status != uploaded`

5. Notes

- `GET/POST /groups/{group_id}/routines/{routine_id}/notes...`: **404** if not group member or routine not in group.
- Video note endpoints:
  - `POST/GET /.../videos/{video_id}/notes`: **404** if not group member, or video not in routine, or video is deleted/not accessible.
- Delete note:
  - If later you restrict to author-only, still prefer **404** for non-author group members to avoid policy probing.

6. Jobs and job artifacts (private by owner)

- All job and artifact endpoints must be authorized by `jobs.user_id == current_user.id`.
- For non-owners:
  - return **404** (not 403) for `/jobs/{job_id}` and `/jobs/{job_id}/artifacts/...`.
- This ensures group membership does not leak job existence or analysis activity.

Testing requirement:

- Every endpoint family must have tests asserting the above status codes for:
  - unauthenticated (401)
  - authenticated but unauthorized (404)
  - authenticated and authorized (200/2xx)
  - soft-deleted resources (404)
  - pending upload privacy rules (404 for non-uploader)

---

## 0) Goals, Non-Goals, and Guiding Principles

### Goals

- Implement **group-based collaboration**:
  - A “partnership” is now a **group** (supports 2+ members; may include coaches).
  - **Routine membership follows group membership** (no per-routine participant list).
  - Group members can **upload videos** and **create notes** (comments/feedback).
- Provide **clear feature boundaries** and **testable units**.
- Build **robust but flexible** infra appropriate for a pre-alpha product.
- Establish **extensibility points** for future changes (especially permissions).

### Non-Goals (for now)

- Backwards compatibility with existing DB schema/data (no users exist yet).
- Complex permission matrices and role-based ACL tooling (we’ll create hooks, not a full system).
- Real-time collaboration features (typing indicators, websockets, etc.).
- External sharing links, public URLs, etc. (optional later).
- UI/frontend changes (this plan is backend-focused).

### Guiding Principles

- Treat privacy as a primary functional requirement, not a later hardening task.

- Prefer **simple, explicit data models** with clear ownership and authorization paths.
- Ensure every write/read path has a single, obvious **authorization gate**.
- Keep APIs **resource-oriented** and predictable.
- Make features **self-contained**:
  - Models + schemas + routes + service functions + tests live together conceptually.
- Use “soft extensibility”: add fields/enums/tables that enable future changes without forcing them today.

---

## 1) Proposed Feature Boundaries

Implement as 4 primary backend features plus a shared authorization layer:

1. **Groups**

- Create group (explicit user action)
- Manage membership (add/remove/list)
- Group invites (v0.0.1): the primary way to establish relationships and grow groups
  - Important separation of concerns:
    - `POST /groups` creates the group
    - `POST /groups/{group_id}/invites` creates invites for an existing group (does not create the group)
- Group-scoped authorization

2. **Routines**

- CRUD routines scoped to a group
- List routines in a group

3. **Videos**

- Register/upload videos scoped to a routine (and therefore group)
- List videos for a routine
- Retrieve presigned download URL(s)
- (Optional) kick off analysis jobs from a routine video

4. **Notes (Annotations & Feedback)**

- Routine notes (general discussion)
- Video notes with timestamps and optional structured details
- AI/system notes support (extensible; write path gated)

5. **Jobs & Job Artifacts (Private)**

- Analysis jobs and their artifacts/reports are **private to the job owner** (`jobs.user_id`) for v0.0.1.
- This is an intentional privacy constraint: group members should not necessarily know a video is being uploaded/analyzed.
- Jobs/artifacts routing must live under the jobs resource (see API section).

Shared layer:

- **Authorization/Permissions hooks**
  - Today: “any group member can upload/comment on uploaded content”
  - Privacy constraints (v0.0.1):
    - **pending uploads are uploader-private by default** (see Videos API)
    - **analysis jobs and job artifacts are private to the job owner** (see Jobs section)
    - unauthenticated/non-member requests should not leak existence (prefer 404 where feasible)
  - Future: role/capability checks can be inserted centrally without rewiring endpoints.

---

## 2) Data Models & Relationships (DB + ORM)

> Assumption: existing codebase uses SQLAlchemy + Alembic migrations, FastAPI, and Pydantic schemas.

### 2.1 Core Entities

#### `users`

- Already exists.
- No changes required here for 0.0.1 unless needed for roles or service accounts.

#### `groups`

- **New**
- Fields:
  - `id` (PK)
  - `name` (string, required)
  - `created_by` (FK -> users.id, required)
  - `created_at`, `updated_at`
  - (Optional) `description` (text)
  - (Optional) `is_archived` (bool)

#### `group_memberships`

- **New**
- Join table: users <-> groups
- Fields:
  - `group_id` (FK -> groups.id, PK component)
  - `user_id` (FK -> users.id, PK component)
  - `role` (enum) — **extensibility point**
    - Suggested minimal roles:
      - `member` (default)
      - `coach` (optional semantic)
      - `owner` (creator; optional)
  - `status` (enum) — supports group-invite flow (email-first, pre-account)
    - `active`
    - `invited`
    - `removed` (optional; may prefer hard delete)
  - `created_at`

**Relationship rules**

- A user can belong to many groups.
- A group has many memberships.
- Membership existence is the primary check for access to group routines/videos/notes.

#### `routines`

- **Refactor** existing table.
- Replace/adjust fields:
  - `group_id` (FK -> groups.id, required, indexed) **NEW**
  - `dance_id` (existing)
  - `title` (existing)
  - Remove or deprecate `owner_id` (routine owned by group; if you keep it, treat as creator_id)
  - Add `created_by` (FK -> users.id, required) to track who created the routine
  - `created_at`, `updated_at`

**Relationship rules**

- A routine belongs to exactly one group.
- Any group member can read/write routines (subject to future permission hooks).

#### `videos`

- **Refactor** existing table (normalized group scoping).
- Replace/adjust fields:
  - `routine_id` (FK -> routines.id, required, indexed) **NEW**
  - `uploaded_by` (FK -> users.id, required)
  - `storage_key` (existing)
  - `created_at`
  - `duration`, `file_size` (existing)
  - `status` (enum, required, indexed) — **robust upload lifecycle + soft delete**
    - `pending_upload` (default)
    - `uploaded`
    - `deleted` (soft delete)
  - (Optional) `original_filename` (string, nullable) — stored in DB only
  - (Optional) `content_type` (string, nullable)

- Replace `visibility`/`VideoPermission` model with group-scoped access:
  - Remove `visibility`
  - Remove `VideoPermission` join table (no per-video sharing in the new model).

**Relationship rules**

- Video belongs to a routine; group is derived via `videos.routine_id -> routines.group_id`.
- Group members can access **uploaded** videos; pending uploads are uploader-private by default (see Videos API).
- Soft-deleted videos are treated as non-existent by default (404 for direct access).

#### `notes`

- **Refactor** existing table (normalized group scoping).
- Make notes routine-scoped and optionally video-scoped (and resilient to video deletion).
- Fields:
  - `id` (PK)
  - `routine_id` (FK -> routines.id, required, indexed) (existing)
  - `video_id` (FK -> videos.id, nullable, indexed)
  - `video_deleted` (bool, required, default false) — indicates the note used to reference a video that is now deleted/unlinked
  - `author_id` (FK -> users.id, required, indexed) (existing)
  - `source` (enum) — **extensibility for AI/system**
    - `user` (default)
    - `ai`
    - `system`
  - `note_type` (existing enum; keep or simplify)
  - `contents` (text, required)
  - Timestamp fields for video notes:
    - `video_timestamp_ms` (int, nullable) (prefer ms over seconds for future precision)
    - (Optional later) `video_timestamp_end_ms` (int, nullable)
  - (Optional) `details` (JSONB) — **structured extensibility**
    - e.g., `{ "category": "timing", "body_part": "frame", "severity": 2 }`
  - `created_at`, `updated_at`
  - (Optional later) `parent_note_id` for threads

**Relationship rules**

- Note always belongs to a routine; group is derived via `notes.routine_id -> routines.group_id`.
- Note may also belong to a specific video.
- When a video is soft-deleted, its notes are converted to routine notes:
  - `video_id` set to NULL
  - `video_deleted` set to TRUE
  - `video_timestamp_ms` is retained for context
- Authorization: routine’s group membership required.

### 2.2 Removing/Deprecating Old Structures

#### `routine_participants`

- Remove entirely.
- Rationale: routine membership follows group membership.

#### `user_relations` / `invites` (pairwise)

- Not needed for group-based collaboration.
- Since no users exist yet and backwards compatibility is not required:
  - Option A (preferred for clarity): remove these endpoints/models/migrations and replace with group membership management.
  - Option B: keep temporarily, but do not build new features on them (adds confusion).
- This plan assumes Option A: **replace pairwise invites with group invites** (do not postpone invites).

#### `video_permissions`

- Remove entirely.
- Rationale: access derived from group membership. No per-video share list.

---

## 3) API Design (Routes + Schemas)

All endpoints should be under `/api/v1` to match current structure.

### 3.1 Group API

#### `POST /groups`

- Create a group (this is an explicit user action in the client)
- Request: `{ name, description? }`
- Response: group object
- Side effects:
  - create membership for creator (recommended: `role=owner`, `status=active`)
- Notes:
  - Invites are created by a separate endpoint and never implicitly create the group.

#### `GET /groups`

- List groups current user belongs to

#### `GET /groups/{group_id}`

- Get group details (members count, etc.)

#### `GET /groups/{group_id}/members`

- List memberships (user + role)

#### `POST /groups/{group_id}/members`

- Add member by email/username/user_id (pre-alpha; simplest approach)
- Extensibility: later replace with invite flow

#### `DELETE /groups/{group_id}/members/{user_id}`

- Remove member

**Authorization checks**

- Base: requester must be a group member.
- Membership management future: only owners/admins can add/remove.
- Invite creation future: capability `group:invite:create` (v0.0.1 may allow any active member).

> Extensibility point: implement `require_group_capability(group_id, capability)` but initially allow all members.

### 3.2 Routine API

#### `POST /groups/{group_id}/routines`

- Create routine in group
- Request: `{ title, dance_id }`
- Response: routine object

#### `GET /groups/{group_id}/routines`

- List group routines

#### `GET /groups/{group_id}/routines/{routine_id}`

- Get routine details

#### `PATCH /groups/{group_id}/routines/{routine_id}`

- Update title/dance_id (optional)

#### `DELETE /groups/{group_id}/routines/{routine_id}`

- Delete or archive routine

**Authorization checks**

- Must be group member.

### 3.3 Video API (Routine-scoped)

#### `POST /groups/{group_id}/routines/{routine_id}/videos`

- Register a new video upload (DB record + presigned PUT URL)
- Request: `{ filename, content_type, file_size? }` (duration may be unknown)
- Response: `{ video, upload_url, expires_at }`
- Behavior:
  - Creates a `Video` record with `status=pending_upload`
  - Pending uploads are **uploader-private** (see list endpoint behavior below)

#### `GET /groups/{group_id}/routines/{routine_id}/videos`

- List videos for routine
- Default behavior:
  - Returns only `status=uploaded` videos (visible to all group members)
- Optional behavior (privacy-preserving):
  - `?status=pending_upload` returns only the caller’s pending uploads (uploader-private)
  - (Future) allow elevated roles/capabilities to view pending uploads

#### `GET /groups/{group_id}/routines/{routine_id}/videos/{video_id}`

- Get video metadata
- Behavior:
  - Treat `status=deleted` as not found (404)

#### `POST /groups/{group_id}/routines/{routine_id}/videos/{video_id}/finalize`

- Mark upload complete
- Behavior:
  - Only the uploader can finalize their pending upload (v0.0.1 privacy)
  - Transition `pending_upload -> uploaded` (idempotent finalize recommended)
  - Treat `status=deleted` as not found (404)
- Optionally validate object exists in storage
- Populate `file_size`, `duration` if extractable later

#### `GET /groups/{group_id}/routines/{routine_id}/videos/{video_id}/download`

- Return presigned GET URL to original video
- Behavior:
  - Only works for `status=uploaded`
  - Treat `status=deleted` as not found (404)

#### `DELETE /groups/{group_id}/routines/{routine_id}/videos/{video_id}`

- Soft delete a routine video
- Behavior:
  - Set `status=deleted`
  - Move any associated video notes to the routine (see Notes model):
    - `note.video_id = NULL`
    - `note.video_deleted = TRUE`
  - Treat `status=deleted` as not found (404) for direct access endpoints

**Authorization checks**

- Must be group member for uploaded video access.
- Pending upload privacy:
  - uploader-only for viewing/finalizing `pending_upload` videos in v0.0.1
- Extensibility: capability `video:upload`, `video:download`, `video:delete`.

### 3.4 Notes API (Routine + Video notes)

#### `POST /groups/{group_id}/routines/{routine_id}/notes`

- Create routine-level note
- Request: `{ note_type, contents, details? }`

#### `POST /groups/{group_id}/routines/{routine_id}/videos/{video_id}/notes`

- Create video note with timestamp
- Request: `{ note_type, contents, video_timestamp_ms, details? }`

#### `GET /groups/{group_id}/routines/{routine_id}/notes`

- List routine notes (optionally include video notes)

#### `GET /groups/{group_id}/routines/{routine_id}/videos/{video_id}/notes`

- List video notes

#### `DELETE /groups/{group_id}/routines/{routine_id}/notes/{note_id}`

- Delete note (initially allow author or group member; decide policy)
- Extensibility: capability `note:delete:any` etc.

**Authorization checks**

- Must be group member.
- Extensibility: capability `note:create`.

### 3.5 AI-Generated Notes (Extensibility)

- Do not fully implement AI pipeline here, but support it cleanly:
  - Allow `source=ai` and `author_id` referencing a service user or nullable author with `source=ai`.
  - Prefer: create a dedicated “system” user row and use it as `author_id`.
  - Add internal-only endpoint later or write notes from the analysis worker.

---

## 4) Implementation Steps (Agent-Friendly Work Breakdown)

This replaces the long “phase list” with discrete chunks optimized for parallelizable implementation by a coding agent. Each chunk has:

- scope (what files/logical modules it touches)
- deliverables
- verification steps

### Chunk A — Database Migrations (Schema is the Contract)

Scope:

- Alembic migrations only. No routers/services yet.

Work items (ordered):

1. Migration `006_add_groups_and_memberships`
   - Create `groups`
   - Create `group_memberships`
   - Indexes:
     - `ix_group_memberships_user_id`
     - `ix_group_memberships_group_id`

2. Migration `006b_add_group_invites` (or merge into `006`)
   - Create `group_invites` for email-first invites (invitee may not have an account yet)
     - `id` (PK)
     - `group_id` (FK -> groups.id, required, indexed)
     - `created_by` (FK -> users.id, required, indexed)
     - `email` (string, required, indexed; store normalized lowercase)
     - `role` (optional enum; e.g., `member`, `coach`)
     - `token` (string, unique, required)
     - `status` (enum: `pending`, `accepted`, `revoked`, `expired`)
     - `expires_at` (timestamp, required)
     - `created_at`
     - `accepted_at` (nullable timestamp)
     - `accepted_by_user_id` (nullable FK -> users.id) — populated on accept
   - Security requirement (strict):
     - Invites must be accepted only by an authenticated user whose `user.email` strictly matches the invite `email` (case-insensitive normalized comparison).
     - Tokens must not grant access if forwarded to a different email account.
   - Add uniqueness to prevent accidental duplicates (example policy; confirm in implementation):
     - unique `(group_id, email, status)` where status is `pending` (or enforce in app logic)

3. Migration `007_refactor_routines_to_groups`
   - Add `group_id` and `created_by` to `routines`
   - Drop `routine_participants`

4. Migration `008_refactor_videos_to_routines` (normalized scoping + lifecycle)
   - Add `routine_id`, `uploaded_by` to `videos`
   - Add `status` enum to `videos` with default `pending_upload` and values:
     - `pending_upload`, `uploaded`, `deleted`
   - (Optional) add `original_filename`, `content_type`
   - Drop `video_permissions`
   - Drop `visibility` enum/type if unused

5. Migration `009_refactor_notes_to_routine_scope` (normalized scoping + deletion flag)
   - Add `source`, `video_timestamp_ms`, `details` to `notes`
   - Add `video_deleted` bool default false
   - Replace `video_timestamp` seconds with `video_timestamp_ms`
   - Ensure FK constraints: `notes.video_id -> videos.id` with `SET NULL` on delete
   - Add indexes on `routine_id`, `video_id`

6. Migration `010_remove_user_relations_and_invites`
   - Drop `invites`
   - Drop `user_relations`
   - Drop related enum types if present (`invite_role`, `invite_status`, `user_relation_role`, `user_relation_status`)

Deliverables:

- Migrations run on a fresh Postgres DB from base -> head.

Verification:

- Migration smoke test: upgrade succeeds; downgrade is optional for v0.0.1 but recommended to keep honest.
- Verify enums/types created/dropped as expected.
- Verify constraints for `group_invites.token` uniqueness.

---

### Chunk B — ORM Models (SQLAlchemy)

Scope:

- `models/` only, aligned with the new schema.

Deliverables:

- New models: `Group`, `GroupMembership`, `GroupInvite`
- Updated models: `Routine`, `Video`, `Note`
- Remove: `RoutineParticipant`, `VideoPermission` and any imports/exports referencing them

Verification:

- App imports models without circular import.
- A minimal unit test can create each model and flush to DB.

---

### Chunk C — Schemas (Pydantic)

Scope:

- `schemas/` only

Deliverables:

- `schemas/group.py` (group + membership)
- `schemas/group_invite.py` (invite create/accept responses)
- `schemas/routine.py` (group-scoped routine)
- `schemas/video.py` (upload register/finalize/download; includes status)
- `schemas/note.py` (video_timestamp_ms, details, video_deleted)

Verification:

- Schema tests validate:
  - timestamp non-negative
  - details is object (dict) if present
  - invite email normalization expectations documented (lowercase at persistence layer)

---

### Chunk D — AuthZ Seam (Centralized, Future-Proof)

Scope:

- `core/authorization.py` (or similar), plus dependency wiring if needed.

Deliverables:

- `require_group_member(db, group_id, user_id)` that raises 404 on non-membership
- `require_group_capability(db, group_id, user_id, capability)`:
  - v0.0.1: delegates to membership
  - later: role/capability mapping
- Job owner check helper:
  - `require_job_owner(db, job_id, user_id)` that raises 404 when not owner

Verification:

- Unit tests for:
  - unauthenticated handled at router dependency level (401)
  - unauthorized membership -> 404
  - unauthorized job owner -> 404

---

### Chunk E — Services (Business Logic, Testable Without HTTP)

Scope:

- `services/` layer (recommended), or equivalent functions in routers if you prefer.

Deliverables:

- `GroupsService`
- `GroupInvitesService` (email-first; strict email match; notification stub seam)
- `RoutinesService`
- `VideosService`:
  - `register_upload` creates `pending_upload`
  - `finalize` uploader-only, idempotent; transitions to `uploaded`
  - `soft_delete` sets `deleted` and migrates notes: `video_id=NULL`, `video_deleted=TRUE`
- `NotesService`

Verification:

- Unit tests for each service method (table-driven where applicable).
- Ensure non-leaky behavior is implemented at the service query layer (filter out deleted; enforce uploader-only pending access).

---

### Chunk F — Routers (FastAPI)

Scope:

- `api/v1/*` routers + `main.py` route registration.

Deliverables:

- `groups` router:
  - `POST /groups`
  - `GET /groups`
  - `GET /groups/{group_id}`
  - membership endpoints
- `group_invites` router:
  - `POST /groups/{group_id}/invites`
  - `POST /group-invites/accept` (token-based) (exact path can vary; must follow non-leaky policy)
- `routines` router:
  - group-scoped CRUD
- `routine videos` router:
  - register/finalize/list/download/delete as specified, with non-leaky policy
- `notes` router:
  - routine notes + video notes
- Jobs routing refactor:
  - ensure job artifact endpoints live at `/api/v1/jobs/{job_id}/artifacts/...` and are job-owner private

Verification:

- API tests (FastAPI TestClient) for each router:
  - 401 unauth
  - 404 unauthorized
  - 2xx authorized
  - specific privacy cases: pending_upload non-uploader -> 404; job artifacts non-owner -> 404

---

### Chunk G — Storage Abstraction (Presign + Pluggable)

Scope:

- `services/storage.py` (or similar)

Deliverables:

- `create_presigned_put_url`
- `create_presigned_get_url`
- optional `head_object`
- deterministic key generation strategy (avoid leaking filenames in object keys)

Verification:

- Unit tests with mocked storage client:
  - presign called with the expected key prefix
  - no original_filename leakage into object key unless explicitly intended and sanitized

---

### Chunk H — “Definition of Done” End-to-End Integration Test

Scope:

- One high-level API test that exercises the happy path with the privacy constraints.

Steps:

1. User A creates group.
2. User A creates routine.
3. User A registers pending upload; User B (group member) cannot see it (404 or absent from list).
4. User A finalizes; User B can now see uploaded video and can create notes.
5. User A soft-deletes video; notes migrate to routine with `video_deleted=true`; direct video access returns 404.
6. User B cannot access User A’s job artifacts (404) even if same group.

Deliverables:

- One test that proves the entire privacy story.

---

### 5) Key Methods/Functions and Responsibilities

### 5.1 Groups (service-level responsibilities)

- `GroupsService.create_group(current_user, payload) -> Group`
  - Creates group
  - Creates membership for creator (`role=owner`, `status=active`)
- `GroupsService.add_member(group_id, actor_user, target_user, role) -> GroupMembership`
  - Uses `require_group_capability(group_id, actor_user, "group:manage_members")`
  - v0.0.1: allow any active member; later restrict to owner
- `GroupsService.remove_member(...)`

### 5.1b Group invites (service-level responsibilities; email-first, pre-account)

- `GroupInvitesService.create_invite(group_id, actor_user, email, role?) -> GroupInvite`
  - Uses `require_group_capability(group_id, actor_user, "group:invite:create")`
  - Normalizes email to lowercase
  - Creates a `GroupInvite(status=pending, token, expires_at)`
  - Does NOT create membership
  - Calls a stubbed notification sender (see below)
- `GroupInvitesService.accept_invite(token, accepting_user) -> GroupMembership`
  - Validates token, status, expiration
  - Ensures strict email match:
    - `accepting_user.email` must match invite `email` after normalization to lowercase
    - failure should not leak whether the token exists for another email (prefer a generic error)
  - Creates membership if none exists; sets membership `status=active`
  - Marks invite `accepted`, stamps `accepted_at`, `accepted_by_user_id`
- `GroupInvitesService.revoke_invite(invite_id, actor_user)`
  - Marks invite revoked (or deletes), depending on audit needs
- Notification seam:
  - `NotificationsService.send_group_invite_email_stub(email, token, group_id)` (no email provider yet; make it pluggable)

### 5.2 Routines

- `RoutinesService.create_routine(group_id, actor_user, payload)`
  - `require_group_capability(group_id, actor_user, "routine:create")`
- `RoutinesService.list_routines(group_id, actor_user)`
- `RoutinesService.get_routine(group_id, routine_id, actor_user)`
  - Verify routine belongs to group
- `RoutinesService.update_routine(...)`
- `RoutinesService.delete_routine(...)`

### 5.3 Videos

- `VideosService.register_upload(group_id, routine_id, actor_user, filename, content_type, size?)`
  - `require_group_capability(group_id, actor_user, "video:upload")`
  - Create `Video` row (status = `pending_upload` if you add status)
  - Generate storage_key
  - Return presigned PUT
- `VideosService.finalize_upload(...)`
  - Optionally check storage exists
- `VideosService.list_videos(...)`
- `VideosService.get_download_url(...)`
  - `require_group_capability(group_id, actor_user, "video:download")`

### 5.4 Notes

- `NotesService.create_routine_note(group_id, routine_id, actor_user, payload)`
  - `require_group_capability(group_id, actor_user, "note:create")`
- `NotesService.create_video_note(group_id, routine_id, video_id, actor_user, payload)`
  - Validate `video.routine_id == routine_id` and `video.group_id == group_id`
- `NotesService.list_routine_notes(...)`
- `NotesService.list_video_notes(...)`
- `NotesService.delete_note(...)`
  - v0.0.1: allow note author or any group member (decide)
  - Extensibility: `note:delete:any`

---

## 6) Permissions & Extensibility Points (Designed for Change)

Even though v0.0.1 allows “any group member can upload/comment,” build the seam now:

### Central seam

- All endpoints call:
  - `require_group_member(...)` OR `require_group_capability(...)`

### Role model (minimal now)

- `group_memberships.role` exists even if unused.
- Later:
  - define a capability map:
    - `owner`: manage membership, delete routine/video
    - `coach`: comment, create notes, maybe upload
    - `member`: upload, comment
  - implement in `require_group_capability`

### Future features enabled by current model

- Routine-specific roles (if desired later) can be added as a new join table:
  - `routine_roles (routine_id, user_id, role)` (optional, not now)
- External sharing can reintroduce a permissions table (but should be separate):
  - `video_share_links` or `video_acl` with explicit scope.

---

## 7) Testing Strategy (By Feature)

Testing should be layered:

- Unit tests for service functions and auth hooks
- API tests for endpoints (FastAPI TestClient)
- Minimal integration with DB per test module (transaction rollback)
- Storage is mocked

### 7.1 Groups Tests

- Unit:
  - create group creates membership for creator
  - membership required for reading group
- API:
  - `POST /groups` returns 200 and correct payload
  - `GET /groups` returns created group
  - adding/removing members:
    - member can add member (v0.0.1 behavior)
    - non-member cannot add or list members (403/404)

### 7.2 Routines Tests

- Unit:
  - cannot create routine without group membership
  - routine group_id enforced
- API:
  - create routine in group
  - list routines returns only that group’s routines
  - cross-group access forbidden (attempt to fetch routine from another group)

### 7.3 Videos Tests

- Unit:
  - register upload returns storage_key and presigned URL (mocked)
  - creates `Video(status=pending_upload)` with `uploaded_by=current_user`
  - list videos (default) returns only `status=uploaded`
  - list videos with `?status=pending_upload` returns only caller’s pending uploads (privacy)
  - finalize transitions `pending_upload -> uploaded` (idempotent finalize recommended)
  - download returns 404 when:
    - video is deleted
    - video is not uploaded
    - video is not in routine/group
  - soft delete:
    - sets `status=deleted`
    - moves video notes to routine notes (`video_id=NULL`, `video_deleted=TRUE`)
  - cannot access another group’s video
- API:
  - register upload (pending) + verify not visible to other member via default list
  - finalize + list shows uploaded video to group
  - download URL works only for uploaded video
  - delete + verify 404 for direct access + notes moved
- Storage mocking:
  - verify presign methods called with expected key

### 7.4 Notes Tests

- Unit:
  - create routine note ok for member
  - create video note validates timestamp and video belongs to routine/group
  - list filters by routine/video correctly
- API:
  - create routine note then list
  - create video note then list video notes
  - delete note policy tests (author vs non-author)

### 7.5 Authorization Hook Tests

- Unit tests for:
  - `require_group_member`: success/forbidden
  - `require_group_capability` currently delegates but returns seam for future extension

### 7.6 Jobs & Job Artifacts Tests (Private by Owner)

- Unit/API:
  - only the job owner can retrieve `/api/v1/jobs/{job_id}/artifacts/...`
  - a different user (even if in same group) receives 404/403 per API policy
- Purpose:
  - enforces the v0.0.1 privacy boundary between “group-shared uploads” and “private analysis activity”

---

## 8) Implementation Checklist (Developer To-Do)

### Database / Models

- [ ] Add `Group`, `GroupMembership` models + migrations
- [ ] Refactor `Routine` to include `group_id`, remove `RoutineParticipant`
- [ ] Refactor `Video` to include `routine_id`, remove `VideoPermission` & visibility
- [ ] Refactor `Note` with `group_id`, `source`, `details`, `video_timestamp_ms`

### API / Services

- [ ] Add routers: groups, routines, routine videos, notes
- [ ] Add service modules for each feature (or keep in routers initially, but service layer preferred)
- [ ] Add auth hook module for group membership/capabilities

### Storage

- [ ] Add storage abstraction for presigned PUT/GET
- [ ] Ensure consistent object key strategy

### Cleanups

- [ ] Decide: remove legacy pairwise invites/user_relations now (recommended)
- [ ] Rename job artifact video endpoints to avoid confusion with routine videos

### Tests

- [ ] Add test fixtures: user factory, group factory, auth token helper
- [ ] Add tests per feature boundary (unit + API)

---

## 9) Open Design Questions (Resolve Before Coding)

Keep this section short and action-oriented so an agent doesn’t block on ambiguous requirements.

1. **Group invite acceptance UX shape**

- Token accept endpoint path and payload:
  - e.g., `POST /group-invites/accept { token }`
- Confirm: multiple pending invites per email across different groups is allowed (recommended yes).
- Confirm: resend policy (new token vs reuse existing pending invite).

2. **Soft delete retention**

- Should `videos.status=deleted` videos be permanently purged later by a background task?
- If yes, define retention period (e.g., 30/60/90 days) (not required for v0.0.1 implementation).

3. **Ownership semantics**

- v0.0.1 can allow any active member to invite/add/remove and delete videos, but:
  - keep `role=owner` stable so tightening later is a 1-file policy change.

4. **Timestamp precision**

- Confirm ms-based integer is final (recommended yes).

---

## 10) Success Criteria (Definition of Done)

A refactor is “done” when the following are true on a fresh Postgres DB:

### Database

- Migrations run base->head successfully and create:
  - `groups`, `group_memberships`, `group_invites`
  - `routines` with `group_id`
  - `videos` with `routine_id`, `uploaded_by`, `status`
  - `notes` with `video_timestamp_ms`, `details`, `video_deleted`

### API/Behavior

- Group creation and group invite creation are separate concerns:
  - `POST /groups` creates group + creator membership (owner/active)
  - `POST /groups/{group_id}/invites` creates invite only (no membership created yet)
  - `POST /group-invites/accept` accepts with strict email match and creates/activates membership
- Pending upload privacy:
  - pending uploads are invisible to other group members by default
  - finalize is uploader-only
- Soft delete:
  - deleted videos return 404 for direct access
  - notes migrate to routine with `video_deleted=true`
- Jobs privacy:
  - job artifacts are accessible only to job owner (404 for non-owner, even in same group)

### Tests

- Each router has API tests enforcing:
  - 401 unauthenticated
  - 404 authenticated-but-unauthorized (non-leaky)
  - 2xx authorized
  - pending upload privacy cases
  - job artifact privacy cases
- Service-layer unit tests exist for:
  - invite acceptance strict email match and non-leaky failure behavior
  - video soft delete note migration

### Extensibility

- Permission tightening later requires changes only in the centralized capability policy (not every endpoint).
