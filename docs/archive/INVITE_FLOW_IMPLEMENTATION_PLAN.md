# Invite Flow Implementation Plan

## Infrastructure to Set Up

These are external services and hosting you need to provision before the backend work can go live.

### Email Service
Choose one transactional email provider and configure it:

- **Recommended:** SendGrid, Postmark, or AWS SES
- You will need:
  - An API key (or SMTP credentials)
  - A verified sender domain/address (e.g. `noreply@yourdomain.com`)
  - A verified sending domain with SPF/DKIM DNS records — required for deliverability
- Invite emails will come from this address, so use a domain you own

### Web Invite Landing Page
A lightweight page served at `dance-note.com/invite` to handle invite links for users who don't have the app yet.

- Lives at `https://dance-note.com/invite?token=...` — a path on the main domain, not a subdomain
- Separate repo from the backend; deployed as a static site on `dance-note.com`
- Needs HTTPS — use a TLS cert (Let's Encrypt or your hosting provider)
- See `INVITE_PAGE_BUILD_GUIDE.md` for implementation details

### Deep Link Scheme (App-Side)
Register a URL scheme for the app so invite links can open it directly when installed:

- iOS: Universal Links (`apple-app-site-association` file served from `dance-note.com/.well-known/`) — preferred since the invite page is on the main domain
- Android: App Links (`assetlinks.json` at `dance-note.com/.well-known/`) or a custom scheme
- The web landing page will include a button that attempts to open this deep link before falling back to the App Store

---

## Phase 1 — Backend Foundations

### 1. Config additions (`core/config.py`)
- Add `FRONTEND_URL` — base URL of the main site (e.g. `https://dance-note.com`)
- Add email service credentials:
  - `SENDGRID_API_KEY` (or `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`)
  - `EMAIL_FROM_ADDRESS`
  - `EMAIL_FROM_NAME`

### 2. Real email sending (`services/group_invite_service.py`)
- Replace `_send_invite_email_stub()` with actual email transport
- Email content should include:
  - Inviter's display name
  - Group name
  - Invite link: `{FRONTEND_URL}/invite?token=...`
  - Expiry notice (7 days)

### 3. Public invite lookup endpoint
- `GET /api/v1/group-invites/lookup?token=...`
- No authentication required
- Returns: group name, inviter display name, role, expiry timestamp
- Returns 404 for invalid, expired, or revoked tokens
- Used by both the web landing page and the app's deep link handler

### 4. Register-with-invite endpoint (`api/v1/auth.py`)
- `POST /api/v1/auth/register-with-invite`
- Request body: `{ email, username, password, token }`
- Atomically: validate token → create user → accept invite → return auth tokens
- Validates that the registration email matches the invite email before creating the user
- On success: user is created, invite is marked accepted, membership is created, and the user is logged in

### 5. Invite management endpoints (`api/v1/group_invites.py`)
- `GET /api/v1/groups/{id}/invites` — list pending invites for a group (admin only)
- `DELETE /api/v1/groups/{id}/invites/{invite_id}` — revoke an invite (logic already exists in service, just needs to be exposed in the router)
- `POST /api/v1/groups/{id}/invites/{invite_id}/resend` — resend the invite email, optionally refresh the expiry

### 6. Database migration
- No schema changes required — the existing `group_invites` model supports all of the above

---

## Phase 2 — Web Landing Page

A single server-rendered HTML page served by the existing FastAPI backend.

- Route: `GET /invite?token=...`
- On load: calls the public lookup endpoint to fetch invite details
- Renders: "You've been invited to join **[Group Name]** by **[Inviter Name]**"
- If token is invalid or expired: shows a clear error message

**Two user paths on this page:**

1. **Already have the app** — a button that attempts to open the deep link (`dancecoach://invite?token=...`), with a fallback to the App Store if the app isn't installed
2. **New user** — an inline registration form (name, username, password) with email pre-filled from the invite; submits to `POST /auth/register-with-invite`; on success, redirects to the deep link or App Store

---

## Phase 3 — In-App Deep Link Handling

This is app-side work, but depends on the backend phases above being complete.

- App registers a handler for `dancecoach://invite?token=...` (and universal link equivalent)
- On open: call `GET /group-invites/lookup?token=...` to show invite context before prompting login
- If user is already logged in: call `POST /group-invites/accept` with the token directly
- If user is not logged in:
  - Offer login → on success, call `POST /group-invites/accept`
  - Offer registration → use `POST /auth/register-with-invite` to handle everything in one step

---

## Future Improvements

- **Unsent invite retry mechanism** — the current model has no way to distinguish a sent invite from one where email delivery silently failed (both are `PENDING`). To fix this:
  - Add a nullable `email_sent_at` column to `group_invites`
  - Set it on successful SES send
  - Add a script `scripts/retry_unsent_invites.py` that queries for `PENDING` invites with `email_sent_at IS NULL` and retries sending

---

## Implementation Order

1. Provision email service and verify sending domain
2. Provision hosting / confirm the existing backend can serve the landing page
3. Add config (`FRONTEND_URL`, email credentials)
4. Implement real email sending
5. Add public lookup endpoint
6. Build web landing page
7. Add `register-with-invite` endpoint
8. Expose revoke + add list/resend endpoints
9. Implement in-app deep link handling
