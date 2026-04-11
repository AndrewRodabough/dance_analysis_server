# SECURITY HARDENING PLAN

This document is a **planning spec** for a coding LLM to implement security and code-quality hardening for the Dance Analysis Server. Follow it step-by-step. The goals are:

- Remove unsafe dev defaults (secrets, credentials).
- Enforce authorization for video/result access.
- Harden configuration for our self-hosted infra:
  - FastAPI backend + analysis workers + MinIO on the same physical device.
  - Cloudflare R2 is used only as an internet-facing upload buffer (clients → R2 → MinIO).

You are expected to make focused, minimal, and testable changes.

---

## 0. High-Level Objectives

1. **Centralize & validate configuration**
   - No hard-coded usable secrets in code.
   - All secrets must come from environment variables (or external config).
   - App should **fail fast** if critical secrets are missing in production mode.

2. **Lock down access to results and videos**
   - All `/videos/{job_id}/...` endpoints must:
     - Require authentication.
     - Enforce that the requesting user owns the job.

3. **Harden R2 usage as an upload intermediary**
   - R2 is only used as a staging area for uploads.
   - After successful import into MinIO, R2 objects should be deleted (configurable).
   - R2 credentials must be required in production.

4. **Reduce PII in logs**
   - Auth logs should not store full user emails in production.
   - No secrets of any kind should ever be logged.

5. **Documentation & ergonomics**
   - Clarify dev vs. prod configuration in docs.
   - Keep the dev experience easy (sensible dev defaults) but impossible to use by accident in production.

---

## 1. Configuration Hardening

### 1.1. Centralize settings in `app/core/config.py`

**Files to modify**

- `backend/app/core/config.py`
- `backend/app/api/v1/analyze.py`
- `backend/app/api/v1/videos.py`
- Any other module that reads `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `JWT_SECRET_KEY`, etc. directly from `os.getenv`.

**Goals**

- Use a single `Settings` class as the source of truth.
- Support `APP_ENV` with values like `"dev"`, `"staging"`, `"prod"`.
- Remove hard-coded **usable** secrets:
  - JWT secret.
  - MinIO/S3 access key and secret key.
  - Database credentials.
- Provide dev-friendly defaults where appropriate, but **only for dev**.

### 1.2. Implementation details

1. **Extend `Settings`**

   In `backend/app/core/config.py`:

   - Ensure `Settings` defines the following attributes:

     - `APP_ENV: Literal["dev", "staging", "prod"]` with default `"dev"` if env var is not set.
     - `DATABASE_URL: str`
     - `JWT_SECRET_KEY: str`
     - `JWT_ALGORITHM: str`
     - `ACCESS_TOKEN_EXPIRE_MINUTES: int`
     - `REDIS_URL: str`
     - `S3_ENDPOINT: str`
     - `S3_ACCESS_KEY: str`
     - `S3_SECRET_KEY: str`
     - `S3_BUCKET: str`
     - `R2_ENDPOINT: str`
     - `R2_ACCESS_KEY: str`
     - `R2_SECRET_KEY: str`
     - `R2_BUCKET: str`
     - `USE_MOCK_ANALYSIS: bool`

   - Behavior:

     - For **dev**, allow dev-friendly defaults for:
       - `DATABASE_URL` (e.g., a local Postgres instance).
       - `S3_ENDPOINT` (`http://minio:9000`).
       - `S3_BUCKET` (e.g., `dance-videos`).
       - `REDIS_URL` (`redis://redis:6379/0`).

     - **Do not** embed usable secrets in defaults:
       - `JWT_SECRET_KEY` default: empty string.
       - `S3_ACCESS_KEY` default: empty string.
       - `S3_SECRET_KEY` default: empty string.
       - For R2 fields, default to empty string.

2. **Add config validation function**

   In `backend/app/core/config.py`:

   - Add a function `validate_settings()` that performs the following in `APP_ENV == "prod"`:

     - `JWT_SECRET_KEY` must be non-empty and sufficiently long (e.g., length >= 32).
     - `DATABASE_URL` must be non-empty and **must not** use `localhost` as host.
     - `S3_ACCESS_KEY` and `S3_SECRET_KEY` must be non-empty.
     - `R2_ENDPOINT`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`, and `R2_BUCKET` must be non-empty.
     - Optionally: Enforce that `S3_ENDPOINT` and `R2_ENDPOINT` use HTTPS in prod.

   - In `APP_ENV != "prod"`:
     - You can skip strict checks or perform weaker ones.
     - Primary goal is to prevent mis-configured production, not restrict dev.

3. **Call `validate_settings()` on startup**

   - Locate application startup code (likely in `backend/app/main.py` or equivalent).
   - At the earliest reasonable point (after creating `settings`), call `validate_settings()`.
   - If validation fails, **raise** a runtime error to prevent the server from starting.

4. **Stop using `os.getenv` directly in other modules**

   - In `backend/app/api/v1/analyze.py`:
     - Replace all direct `os.getenv` calls for S3/MinIO/R2 with reads from `settings`.
   - In `backend/app/api/v1/videos.py`:
     - Replace S3 configuration env reads with values from `settings`.
   - In any additional modules that configure S3 or R2, route through `settings`.

---

## 2. Result & Video Access Control

Currently, `/videos/{job_id}/...` endpoints are publicly accessible. We must enforce that:

- Only authenticated users can access these endpoints.
- Users can only access their own jobs.

### 2.1. Files to inspect and modify

- `backend/app/api/v1/videos.py`
- `backend/app/services/job_service.py` (to reuse job lookup logic)
- `backend/app/core/deps.py` (auth dependency `get_current_active_user`)
- Any job querying logic that ensures job ownership.

### 2.2. Requirements

1. **Add auth dependency**

   For each endpoint in `backend/app/api/v1/videos.py`:

   - `GET /videos/{job_id}/visualization`
   - `GET /videos/{job_id}/keypoints2d`
   - `GET /videos/{job_id}/keypoints3d`
   - `GET /videos/{job_id}/scores`
   - `GET /videos/{job_id}/feedback`

   Add:

   - `current_user: User = Depends(get_current_active_user)`
   - `db: Session = Depends(get_db)`

2. **Enforce job ownership**

   - Before generating a presigned URL for any resource, ensure:

     - The job with `job_id` exists.
     - The job belongs to `current_user` (i.e., matches `current_user.id`).

   - Use the existing pattern from `backend/app/api/v1/analyze.py`:

     - It likely calls `JobService.get_job_by_id(db, job_id, current_user.id)` to ensure ownership.

   - If the job does not exist or does not belong to the user:
     - Raise `HTTPException(status_code=404, detail="Job not found or access denied")`.

3. **Refactor for DRY-ness (optional but recommended)**

   - In `videos.py`, create a helper function like:

     - `_ensure_job_access(job_id: str, current_user: User, db: Session) -> None`
       - Performs the ownership check and raises if unauthorized.

   - Call this helper at the start of each endpoint before generating presigned URLs.

4. **Behavior to preserve**

   - The actual S3 keys and paths should remain the same (e.g., `results/{job_id}/visualization.mp4`).
   - Return shape should stay `{ "url": "<presigned-url>" }`.
   - Presigned URL expiry (currently ~1 hour) can remain unchanged.

---

## 3. R2 Intermediary Hardening

R2 is used as an intermediary:

- Clients upload large videos directly to R2 (using presigned PUT URLs).
- Backend confirms the upload, copies the object from R2 to MinIO, and processes it locally.

We want to:

- Ensure R2 is **mandatory** in production (if the R2 path is enabled).
- Optionally delete objects from R2 after they’ve been imported to MinIO.

### 3.1. Files to modify

- `backend/app/api/v1/analyze.py`
- `backend/app/core/config.py` (already covered above)

### 3.2. Enforce R2 configuration

The configuration validation in `validate_settings()` (see Section 1) should already ensure that:

- If `APP_ENV == "prod"`, then:
  - `R2_ENDPOINT`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`, and `R2_BUCKET` are non-empty.

In `analyze.py`:

- Ensure that S3 clients for R2 are constructed solely from `settings`.
- Assume `validate_settings()` has already run on startup.

### 3.3. Enable R2 object deletion after import

Currently, the code includes commented-out logic to delete from R2 after copy to MinIO.

**Task**

1. Introduce a configuration flag in `Settings`, e.g.:

   - `DELETE_R2_AFTER_IMPORT: bool = os.getenv("DELETE_R2_AFTER_IMPORT", "true").lower() == "true"`

   (Dev can set it to `false` if they want to debug.)

2. In `backend/app/api/v1/analyze.py`:

   - After a successful copy from R2 to MinIO in `confirm_upload_and_start_analysis`, conditionally delete the object:

     - If `settings.DELETE_R2_AFTER_IMPORT` is `True`:
       - Call `r2_client.delete_object(Bucket=R2_BUCKET, Key=object_key)` inside a `try`/`except`.
       - On failure, log a **warning** but do not fail the endpoint.

3. Keep all logging for R2 operations using `log_storage_operation` (if applicable), but **never** log secret values or full URLs.

---

## 4. Logging & PII Handling

We want logs to be useful for debugging and observability, but:

- Emails and user identifiers are PII.
- In production, avoid logging full emails outright.
- Never log passwords, JWTs, access keys, or other secrets.

### 4.1. Files to modify

- `backend/app/core/logging.py`

### 4.2. Adjust `log_auth_event`

Current behavior logs full emails (if provided).

**Objectives**

1. In `APP_ENV == "dev"`:
   - It is acceptable to log full email addresses (dev convenience).

2. In `APP_ENV != "dev"` (staging/prod):
   - Either:
     - Mask email addresses (recommended), or
     - Omit them entirely from logs.

**Implementation guidelines**

- Define a small helper inside `logging.py`, e.g.:

  - `_mask_email(email: str) -> str`

  Behavior example:

  - `"john.doe@example.com"` → `"j***e@example.com"`.
  - If parsing fails, return `"***"`.

- In `log_auth_event`:

  - Access `APP_ENV` via `settings` (import `settings` from `app.core.config`).
  - When adding `email` to `log_data`:

    - If `settings.APP_ENV == "dev"`:
      - Use full email.
    - Else:
      - Use `_mask_email(email)` or skip storing `email` in `log_data`.

3. Ensure that **no other loggers** in `logging.py` or elsewhere log emails, passwords, or tokens in plaintext.

---

## 5. JWT & Secret Handling

The JWT code is mostly fine, but the default secret is dev-only and unsafe in prod.

### 5.1. Files to inspect

- `backend/app/core/config.py`
- `backend/app/core/security.py`
- `backend/app/api/v1/auth.py` (or equivalent auth routes, to ensure no logging of credentials)

### 5.2. Tasks

1. **Remove any hard-coded default JWT secret**

   - `JWT_SECRET_KEY` in `Settings` should default to an empty string.
   - `validate_settings()` must enforce a strong secret for `APP_ENV == "prod"`.

2. **Ensure no token or password logs**

   - Search for any logging of authentication payloads.
   - If found:
     - Remove or redact them.
   - `log_auth_event` must not log raw passwords or access tokens.

3. **Verify JWT creation/decoding**

   - Confirm that `create_access_token` only embeds necessary claims (e.g., `"sub"`).
   - Confirm that `decode_access_token` is used safely (already appears correct, but double-check usage patterns).

---

## 6. Documentation & Dev/Prod Separation

We do not change infrastructure in this pass, but we should improve clarity.

### 6.1. Files to modify

- `README.md`
- `START_HERE.md`
- Optionally add a short section to this `SECURITY_HARDENING_PLAN.md` if you add new env vars.

### 6.2. Tasks

1. **Clarify `APP_ENV` usage**

   - Document that:
     - `APP_ENV=dev` is intended for local development:
       - Allows dev defaults for Postgres, MinIO endpoint, etc.
     - `APP_ENV=prod` is for production:
       - Requires explicit, strong secrets.
       - Requires R2 configuration.
       - Applies stricter logging and config validation.

2. **Document new env vars**

   Document at least:

   - `APP_ENV`
   - `JWT_SECRET_KEY`
   - `S3_ENDPOINT`
   - `S3_ACCESS_KEY`
   - `S3_SECRET_KEY`
   - `S3_BUCKET`
   - `R2_ENDPOINT`
   - `R2_ACCESS_KEY`
   - `R2_SECRET_KEY`
   - `R2_BUCKET`
   - `DELETE_R2_AFTER_IMPORT`

3. **Warn about dev-only defaults**

   - Explicitly state that:
     - Dev credentials like `minioadmin/minioadmin` are **never** to be used in production.
     - Example config snippets for prod must show safe patterns (no real keys, just placeholders).

---

## 7. Implementation Order & Strategy

For efficient and low-risk implementation, follow this order:

1. **Config refactor & validation**
   - Extend `Settings` and add `validate_settings()`.
   - Route all S3/MinIO/R2/JWT/env accesses through `settings`.
   - Wire `validate_settings()` into the app startup.

2. **Access control for `/videos/{job_id}/...`**
   - Add auth dependencies and job ownership checks.
   - Keep response shapes identical.
   - Verify that tests or manual calls to `/videos` now require authentication.

3. **R2 cleanup**
   - Add `DELETE_R2_AFTER_IMPORT` to `Settings`.
   - Implement conditional deletion in `confirm_upload_and_start_analysis`.

4. **Logging PII controls**
   - Implement `_mask_email` in `logging.py`.
   - Adjust `log_auth_event` behavior based on `APP_ENV`.

5. **Documentation updates**
   - Add/update docs for new env vars and `APP_ENV`.
   - Clarify dev vs. prod expectations.

At each stage, keep changes **small and localized**, and preserve existing interfaces as much as possible.

---

## 8. Non-Goals (For This Pass)

The following are explicitly **out of scope** for this plan (but may be candidates for future work):

- Changing infrastructure topology (e.g., splitting backend/workers to different machines).
- Introducing new auth mechanisms or RBAC beyond job ownership.
- Implementing rate limiting or WAF behavior (expected to be handled by Cloudflare / ingress).
- Large-scale refactors of video processing code or models.

---

## 9. Quality Checklist for the LLM

Before considering your changes complete, verify:

- [ ] The app refuses to start in `APP_ENV=prod` when:
  - `JWT_SECRET_KEY` is missing or too short.
  - R2 or MinIO credentials are missing.
- [ ] `/videos/{job_id}/...` endpoints:
  - [ ] Reject unauthenticated users.
  - [ ] Reject authenticated users for jobs they do not own.
  - [ ] Continue to return a JSON object with `{"url": "<presigned-url>"}` for authorized users.
- [ ] R2 object deletion runs after confirmation when `DELETE_R2_AFTER_IMPORT=true`, and failures are logged but do not break the pipeline.
- [ ] Logs:
  - [ ] Do not contain plaintext emails in `APP_ENV=prod`.
  - [ ] Never log secrets, passwords, JWTs, or access keys.
- [ ] Documentation tells a new engineer exactly how to configure dev vs. prod environments safely.

Use this document as the source of truth for your changes. Do not introduce unrelated refactors unless necessary to accomplish the tasks above.
