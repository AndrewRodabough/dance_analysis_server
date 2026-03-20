# Client App Guide: Groups / Routines / Videos / Notes / Invites (v0.0.1)

This document describes the **client-side implications** of the backend refactor to group-based sharing and privacy-first collaboration. It is intended as an actionable checklist for implementing the new flows in the client.

> Privacy is a primary requirement (youth teams/minors). The backend intentionally uses **non-leaky behavior** (often returning 404 for unauthorized) to avoid revealing existence of groups/videos/jobs.

---

## 1) Mental Model (What Changed)

### Old model (to remove from client assumptions)
- “Pairwise relationships” (partner/coach/student) and pairwise invites.
- Per-video explicit sharing permissions.

### New model (client must adopt)
- **Groups** are the collaboration unit (can be 2+ dancers, coaches, teams).
- **Routines belong to a group**.
- **Routine membership follows group membership** (no per-routine participants UI).
- **Videos belong to a routine** and are group-shared **only after uploaded/finalized**.
- **Notes belong to a routine**, optionally tied to a video timestamp.
- **Group invites** are the only invite mechanism:
  - Invite by email **before an account exists**.
  - Accepting invite requires **strict email match**.

### Privacy boundaries (client-visible)
- **Pending uploads are private to the uploader** by default:
  - Other group members should not see that an upload is in progress.
- **Analysis jobs and job artifacts are private to the job owner**:
  - Even if two users are in the same group, non-owners cannot access each other’s job artifacts.

---

## 2) Feature Boundaries for the Client App

Implement these as separate “features/modules” in the client, with clear testable UI flows:

1. **Auth & Session**
2. **Groups**
3. **Group Invites**
4. **Routines**
5. **Routine Videos (Upload + Library)**
6. **Notes (Routine + Video Notes)**
7. **Jobs & Job Artifacts (Owner-only)**

Each feature should:
- have isolated API calls
- have UI state and error handling rules documented
- be testable with mocked network responses (including non-leaky 404 cases)

---

## 3) Navigation / IA Recommendations

### Primary navigation
- “Groups” (list)
  - Group detail
    - “Routines”
      - Routine detail
        - “Videos”
          - Video detail / player + notes
        - “Routine Notes” (general notes feed)
    - “Members”
    - “Invites” (manage pending invites)

### Jobs
- “My Jobs” (optional) or accessed from video “Analyze” action
  - Job detail
    - Artifacts/report (owner-only)

---

## 4) Core User Flows (End-to-End)

### Flow A: Create a group, then invite others (separation of concerns)
1. User selects “Create Group”
2. Client calls `POST /groups`
3. Group is created and user becomes the first member (owner/active)
4. User navigates to “Invites”
5. Client calls `POST /groups/{group_id}/invites` for each email invite

**Important**: invites never implicitly create the group. The client must call create-group first.

---

### Flow B: Accept an invite after signup/login (strict email match)
1. Invitee receives a token (initially: you may display token in UI or copy/share it manually until email service exists)
2. Invitee signs up with the invited email address (must match; case-insensitive)
3. Invitee uses “Accept invite” UI:
   - calls `POST /group-invites/accept` (token-based)
4. If user email does not match, backend will respond non-leakily (often 404)
5. On success:
   - client refreshes group list and navigates to the group

Client UI requirement:
- If accept fails, show a generic message:
  - “Invite is invalid or expired.”
  - Avoid messaging like “email mismatch” or “token is valid” (do not encourage probing).

---

### Flow C: Create routine in a group
1. Navigate to group
2. Tap “Create Routine”
3. Call `POST /groups/{group_id}/routines`
4. Navigate to routine detail

---

### Flow D: Upload a routine video (privacy-first pending upload)
Key idea: uploading is a 2-step process:
- **register** (creates DB record, returns presigned PUT URL)
- **finalize** (declares upload complete and makes it group-visible)

1. On routine detail → “Add Video”
2. Client calls `POST /groups/{group_id}/routines/{routine_id}/videos`
   - receives `{ video, upload_url, expires_at }`
   - `video.status` will be `pending_upload`
3. Client uploads the file via `PUT upload_url` (direct to storage)
4. Client calls `POST /groups/{group_id}/routines/{routine_id}/videos/{video_id}/finalize`
5. Video becomes visible to group members (status `uploaded`)

Client must implement:
- upload progress UI for the uploader (local-only)
- retry/resume logic (best-effort)
- idempotent finalize behavior (safe to call again if uncertain)

Privacy constraint:
- The video list shown to all group members should only show `uploaded` videos.
- Pending uploads should not appear to other members.

---

### Flow E: View routine videos and add notes
1. Group member opens routine → videos list
2. Select a video → play video
3. Add note:
   - Routine-level note: not tied to video
   - Video note: includes timestamp in milliseconds

---

### Flow F: Soft-delete a video (and how client should display notes)
Soft delete means:
- Video becomes inaccessible and should appear removed in normal lists.
- Notes that were tied to the deleted video:
  - become routine notes (video_id cleared)
  - have `video_deleted=true` so client can render an indicator

Client should:
- Remove deleted video from UI lists by default.
- For notes with `video_deleted=true`:
  - display a chip/badge like “Video unavailable” (exact UX later)
  - still show the note content and timestamp if provided
  - do not try to deep-link to the missing video

---

### Flow G: Analysis jobs and artifacts (job-owner only)
When a user triggers analysis (if/when UI adds it):
- job artifacts/report are visible only to the user who started the job.

Client should:
- Place analysis results under “My Jobs” or show within video UI only for the uploader/job owner.
- If a non-owner tries to fetch job artifacts and receives 404, treat it as “not available”.

---

## 5) API & Data Contract Summary (Client Perspective)

> Paths are illustrative; follow actual OpenAPI once updated. The behavior and constraints are the key points.

### Groups
- Create group: `POST /groups`
- List groups: `GET /groups`
- Group detail: `GET /groups/{group_id}`
- Members: `GET /groups/{group_id}/members`

### Group Invites
- Create invite: `POST /groups/{group_id}/invites`
  - request: email (+ optional role)
  - response: includes invite id, email, status, expires_at, and token (until email service exists)
- Accept invite: `POST /group-invites/accept`
  - request: `{ token }`
  - success: membership active and group becomes visible in `GET /groups`

Strict acceptance rule:
- accepting user email must match invite email (case-insensitive normalized).

### Routines
- Create routine: `POST /groups/{group_id}/routines`
- List routines: `GET /groups/{group_id}/routines`
- Routine detail: `GET /groups/{group_id}/routines/{routine_id}`

### Videos
- Register upload: `POST /groups/{group_id}/routines/{routine_id}/videos`
  - response includes presigned PUT
- Finalize: `POST /groups/{group_id}/routines/{routine_id}/videos/{video_id}/finalize`
- List videos: `GET /groups/{group_id}/routines/{routine_id}/videos`
  - default returns only uploaded videos
- Download: `GET /groups/{group_id}/routines/{routine_id}/videos/{video_id}/download`
  - response likely `{ url }`

Statuses:
- `pending_upload` (uploader-private)
- `uploaded` (group-visible)
- `deleted` (non-existent to client; treat as 404)

Soft delete:
- `DELETE /groups/{group_id}/routines/{routine_id}/videos/{video_id}`

### Notes
- Create routine note: `POST /groups/{group_id}/routines/{routine_id}/notes`
- List routine notes: `GET /groups/{group_id}/routines/{routine_id}/notes`
- Create video note: `POST /groups/{group_id}/routines/{routine_id}/videos/{video_id}/notes`
- List video notes: `GET /groups/{group_id}/routines/{routine_id}/videos/{video_id}/notes`

Fields to expect:
- `video_timestamp_ms` (int, >= 0) for video notes
- `details` (object/dict) optional
- `video_deleted` boolean flag (if video was deleted and note was migrated)

### Jobs & Artifacts
- Jobs list/detail: under `/jobs`
- Artifacts: `/jobs/{job_id}/artifacts/...`
- Authorization: job owner only

---

## 6) Error Handling & Privacy: Client Requirements

### Non-leaky policy means “404 ≈ not accessible”
Client must treat 404 as one of:
- resource truly doesn’t exist
- resource exists but user is not authorized
- resource was soft-deleted
- resource exists but is private (pending upload, job artifacts not owned)

Client guidelines:
- Do not show UI that implies existence based on 404.
- Prefer generic messaging:
  - “Not found or you don’t have access.”
  - For invite accept: “Invite is invalid or expired.”

### 401 vs 404
- 401: show login flow
- 404: show generic “not found” or navigate back; do not reveal details
- Avoid repeated retries on 404 unless user action changes auth context.

---

## 7) UI State & Caching Considerations

### Recommended caching boundaries
- Cache groups list (`GET /groups`) per session.
- Cache group detail/members with short TTL; invalidate when:
  - invite accepted (self)
  - member changes (you add/remove membership)
- Cache routine list per group; invalidate on routine create/update/delete.
- Cache video list per routine; invalidate on finalize/delete/upload success.

### Optimistic UI: be careful
Given privacy constraints:
- Do not optimistic-add pending uploads to the group-visible list.
- For uploader:
  - show pending upload in a local “Uploads” panel, not in shared list.
  - after finalize success, refresh the shared list.

---

## 8) Client Testing Strategy (Actionable)

### Unit tests (API client)
- Parsing:
  - note timestamps in ms
  - status enums
- Error handling:
  - 401 triggers auth handler
  - 404 maps to generic “not found / no access”
  - invite accept failure remains generic

### Integration tests (UI with mocked backend)
- Create group → invite creation → shows pending invite list
- Accept invite:
  - success adds group to groups list
  - failure shows generic message
- Upload flow:
  - register returns upload_url
  - PUT called
  - finalize makes video visible in list
- Pending upload privacy:
  - second user in same group never sees pending upload in list
- Soft delete:
  - video disappears
  - notes show `video_deleted` indicator
- Jobs privacy:
  - non-owner user gets 404 for job artifacts

---

## 9) Implementation Checklist (Client To-Do)

### Required screens / components
- Groups list
- Group detail (tabs: Routines, Members, Invites)
- Create group form
- Invite creation form (email input, optional role)
- Invite accept screen (token input / deep link handling)
- Routine list + routine create
- Routine detail:
  - Uploaded videos list
  - “Upload video” UI (register → PUT → finalize)
  - Routine notes feed + add note
- Video detail:
  - Video player
  - Video notes timeline/list + add timestamp note
- My Jobs (optional in v0.0.1; but plan for it)

### Required behaviors
- Enforce non-leaky error handling UX
- Do not expose pending uploads to other group members
- Treat job artifacts as private to owner
- Strict invite acceptance UX (generic failures)

---

## 10) Notes on Email Service (Not Implemented Yet)
Backend will be “ready” for email by:
- generating invite tokens
- storing invite records with status + expiry
- exposing create/accept endpoints

Client expectation for v0.0.1:
- The invite token may be surfaced directly in the UI after invite creation (temporary).
- Later, when email is added:
  - client flow should support deep links like `myapp://invites/accept?token=...`
  - the accept screen should consume the token and call accept endpoint.

---

## 11) Security Reminders for Client Implementation
- Do not log presigned URLs or tokens in analytics.
- Treat invite tokens as sensitive credentials:
  - avoid copying to clipboard automatically
  - warn users before sharing
- Avoid showing raw emails in group screens to non-members; only members can see membership lists.

---
