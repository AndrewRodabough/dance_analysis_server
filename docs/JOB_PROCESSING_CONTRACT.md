# Job Processing Contract (Postgres-Backed Queue + Object Storage)

## 1. Purpose

Define the atomic, idempotent, crash-safe lifecycle for processing a video analysis job.

The Postgres DB is the authoritative system-of-record for:

- Job state
- Attempt ownership
- Canonical result reference
- Retry limits

Object storage holds artifacts but does **not** determine truth.

---

## 2. Job Table (Authoritative State)

**Minimum required fields:**

```/dev/null/JOB_PROCESSING_CONTRACT.sql#L1-20
CREATE TABLE jobs (
  id UUID PRIMARY KEY,
  status TEXT,                  -- available | processing | completed | failed
  attempt_number INT NOT NULL DEFAULT 0,
  max_retries INT NOT NULL,
  worker_id TEXT,
  lease_expires_at TIMESTAMP,
  result_status TEXT,           -- none | writing | complete
  result_object_key TEXT,       -- canonical artifact
  retries INT NOT NULL DEFAULT 0,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
```

**Invariants**

- Only one row per `job.id`
- Only one canonical result per job
- Only the active attempt may finalize
- All state transitions happen in transactions

---

## 3. Job Claiming (Atomic)

Workers must claim jobs using a transaction.

### Claim Query Pattern

```/dev/null/JOB_PROCESSING_CONTRACT.sql#L21-51
BEGIN;

SELECT id
FROM jobs
WHERE status = 'available'
  AND retries < max_retries
ORDER BY created_at
FOR UPDATE SKIP LOCKED
LIMIT 1;

-- If a row is returned:
UPDATE jobs
SET status = 'processing',
    attempt_number = attempt_number + 1,
    worker_id = $worker_id,
    lease_expires_at = NOW() + interval '5 minutes',
    result_status = 'none'
WHERE id = $job_id;

COMMIT;
```

### Guarantees

- `FOR UPDATE SKIP LOCKED` prevents double-claiming
- `attempt_number` uniquely identifies this processing attempt
- Lease timeout enables crash recovery

---

## 4. Lease Expiration / Reclaiming

A job is reclaimable if:

```/dev/null/JOB_PROCESSING_CONTRACT.sql#L52-59
status = 'processing'
AND lease_expires_at < NOW()
```

Reclaiming increments:

- `retries += 1`
- `attempt_number += 1`

Reclaim logic must be **transactional**.

If `retries >= max_retries`, mark:

```/dev/null/JOB_PROCESSING_CONTRACT.sql#L60-63
status = 'failed'
```

---

## 5. Output Writing Contract

### 5.1 Object Storage Layout

All outputs stored under:

```/dev/null/JOB_PROCESSING_CONTRACT.txt#L1-3
results/{job_id}/attempt_{attempt_number}/...
```

**Rules:**

- Never overwrite previous attempts
- Each attempt writes only to its own folder
- Writes are append-only within an attempt

---

## 6. Finalization (Commit Phase)

Finalization must be atomic and **DB-authoritative**.

### Worker Finalization Transaction

```/dev/null/JOB_PROCESSING_CONTRACT.sql#L64-86
BEGIN;

UPDATE jobs
SET result_status = 'complete',
    result_object_key = $object_key,
    status = 'completed'
WHERE id = $job_id
  AND status = 'processing'
  AND attempt_number = $attempt_number;

-- Must check rows_affected == 1

COMMIT;
```

### Critical Rule

If `rows_affected == 0`:

- This worker lost ownership
- **DO NOT** mark output final
- Abort finalization

---

## 7. Idempotency Rules

### DB Layer

- `jobs.id` is unique
- Finalization requires matching `attempt_number`
- Use **conditional updates** to prevent stale writes

### Object Storage Layer

- Deterministic attempt-scoped keys
- No in-place overwrite of canonical result
- Canonical reference exists only in DB (`result_object_key`)

### Safe Re-Execution

If a worker restarts:

- It must re-check DB state
- If `status = 'completed'`, exit immediately
- If `attempt_number` does not match its attempt, abort

---

## 8. Partial Write Handling

If crash occurs:

- **Before** finalization → attempt folder is ignored
- **After** finalization → job is complete
- Reclaimed attempts write to a new folder
- A background cleanup process may delete orphaned attempt folders

---

## 9. Allowed State Transitions

```/dev/null/JOB_PROCESSING_CONTRACT.txt#L4-9
available          → processing
processing         → completed
processing         → available (lease expired)
processing         → failed    (retry limit exceeded)
```

Illegal transitions must not occur.

---

## 10. Worker Responsibilities

A worker **MUST**:

- Claim job transactionally
- Use assigned `attempt_number`
- Write only to its attempt folder
- Finalize using conditional DB update
- Abort if it loses ownership

A worker **MUST NOT**:

- Finalize without DB condition check
- Overwrite another attempt’s folder
- Assume object storage state is authoritative

---

## 11. Source of Truth Hierarchy

```/dev/null/JOB_PROCESSING_CONTRACT.md#L1-8
| Concern               | System of Record  |
|-----------------------|-------------------|
| Job lifecycle         | Postgres          |
| Attempt ownership     | Postgres          |
| Canonical result key  | Postgres          |
| Raw video             | Object storage    |
| Attempt artifacts     | Object storage    |
| Logs                  | Logging system    |
```

Postgres decides which result is authoritative.

---

## 12. Design Principles

- DB controls authority
- Object storage is artifact storage only
- All claims and commits are conditional
- Retries must be safe
- No stale worker may finalize

This contract must be strictly followed by all job-processing agents.
