# Invitation Flow — Backend Requirements

This document describes the backend changes needed to complete the invitation flow. The frontend accept pages are implemented; the items below are blockers or improvements needed from the API.

---

## Access model

Group membership and session membership are distinct:

- **Group invite → group membership.** If the group is linked to one or more sessions, the backend also creates a session membership for each linked session as a side effect. Accepting a group invite is the primary way a coach adds dancers/viewers to all sessions in a group at once.
- **Session invite → session membership only.** Grants access to one specific session. Does not create or imply any group membership. Used for one-off access (e.g. a guest coach reviewing a single session).

These two invite types are kept as separate resources (`/group-invites` and `/session-invites`) throughout the API.

---

## 1. Group Invite Lookup (unauthenticated) — **Blocker**

**Why it's needed:** The session invite flow shows a preview of the invitation (role, expiry) before the user logs in. Group invites have no equivalent endpoint, so unauthenticated users landing on `/accept-invite/:token` see only a generic "You've been invited to join a group" message with no details.

**New endpoint:**

```
GET /api/v1/group-invites/lookup?token=<token>
```

No auth required. Returns public invite details safe to expose before login:

```json
{
  "group_id": "string",
  "group_name": "string",
  "role": "string",
  "expires_at": "ISO 8601 datetime"
}
```

Errors:

- `404` — token not found
- `410` — invite expired or already accepted/revoked

This mirrors the existing `GET /api/v1/session-invites/lookup?token=<token>` endpoint.

---

## 2. List Pending Invites — **Required for existing-user flow**

Existing logged-in users need a way to discover invitations without clicking an email link. The frontend will show a single chronologically sorted list of all pending invitations on a dedicated `/invitations` route, with a visible type label on each item ("Group" vs "Routine") so the user knows what they're accepting and what access it implies.

**Matching logic for both:** match on the authenticated user's email address, status = `pending`, `expires_at` > now.

The frontend fetches both endpoints in parallel and merges the results, sorted by `created_at` descending.

**Group invites:**

```
GET /api/v1/group-invites/pending
```

Auth required. Accepting grants group membership and, as a side effect, session membership for every session currently linked to that group.

```json
[
  {
    "token": "string",
    "group_id": "string",
    "group_name": "string",
    "role": "string",
    "expires_at": "ISO 8601 datetime",
    "created_at": "ISO 8601 datetime"
  }
]
```

**Session invites:**

```
GET /api/v1/session-invites/pending
```

Auth required. Accepting grants access to that session only — no group membership is created.

```json
[
  {
    "token": "string",
    "session_id": "string",
    "routine_name": "string",
    "role": "string",
    "expires_at": "ISO 8601 datetime",
    "created_at": "ISO 8601 datetime"
  }
]
```

Both responses must include `created_at` so the frontend can sort the merged list. The `type` discriminator (`"group"` or `"session"`) is implicit from which endpoint the item came from — the frontend tags each item at merge time rather than relying on the API to include it.

---

## 3. Session/Routine Name in Invite Responses — **UX improvement**

Currently `SessionInviteLookupResponse` returns only `session_id`, `role`, and `expires_at`. The accept page can only say "You've been invited to join a session as Viewer." — it cannot name the routine.

Add `routine_name` to the session invite lookup response so the page can say "You've been invited to join **Spring Showcase** as Viewer." No group context is included — session invites are standalone.

Updated session invite lookup response:

```json
{
  "session_id": "string",
  "routine_name": "string",
  "role": "string",
  "expires_at": "ISO 8601 datetime"
}
```

For the group invite lookup response (item 1 above), `group_name` is already included. No session details are shown at this stage — the specific sessions the user gains access to are a consequence of group-session links, not something surfaced during the invite accept flow.

---

## 4. Deep Link Configuration — **Required for mobile**

The email links need to open the native app when installed. This is a combination of backend and platform config.

**Backend:** Host an HTTPS deep link association file at the app's domain:

- iOS: `https://<domain>/.well-known/apple-app-site-association`

  ```json
  {
    "applinks": {
      "apps": [],
      "details": [
        {
          "appID": "<TEAM_ID>.<BUNDLE_ID>",
          "paths": ["/accept-invite/*", "/accept-session-invite/*"]
        }
      ]
    }
  }
  ```

- Android: `https://<domain>/.well-known/assetlinks.json`
  ```json
  [
    {
      "relation": ["delegate_permission/common.handle_all_urls"],
      "target": {
        "namespace": "android_app",
        "package_name": "<package>",
        "sha256_cert_fingerprints": ["<fingerprint>"]
      }
    }
  ]
  ```

**Mobile client:** `AndroidManifest.xml` and `Info.plist` need intent filters / associated domains configured pointing to the same domain. This is a separate task for the mobile build config — not covered here.

**Email links:** The SES invite email should link to `https://<domain>/accept-invite/<token>` or `https://<domain>/accept-session-invite/<token>` (not a custom scheme). Universal/App Links work over HTTPS and fall back to the browser automatically when the app is not installed.
