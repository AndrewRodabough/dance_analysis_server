# User Info Refactor

Adds `first_name` and `last_name` to the user system via a new `user_info` table. Chosen over expanding the `users` table directly because more profile fields are expected in future iterations.

## Database

New table `user_info` — one row per user, created at registration.

```sql
CREATE TABLE user_info (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    first_name  VARCHAR(100),
    last_name   VARCHAR(100),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ
);
```

`UNIQUE` on `user_id` enforces one-to-one. All profile columns are nullable — the row is created empty at registration and populated as the user fills in their profile.

Migration: `alembic/versions/008_add_user_info_table.py`

## Files to Change

### New files

| File | Purpose |
|------|---------|
| `app/models/user_info.py` | SQLAlchemy `UserInfo` model |
| `app/schemas/user_info.py` | `UserInfoResponse`, `UserInfoUpdate` Pydantic schemas |
| `app/services/user_info_service.py` | Create/update logic |
| `app/api/v1/user_info.py` | `GET /api/v1/users/me/info`, `PATCH /api/v1/users/me/info` |
| `alembic/versions/008_add_user_info_table.py` | Migration |

### Modified files

**`app/models/user.py`**
Add relationship:
```python
user_info = relationship("UserInfo", back_populates="user", uselist=False)
```

**`app/models/__init__.py`**
Export `UserInfo`.

**`app/schemas/user.py`**
Add optional `user_info` field to `UserResponse`:
```python
from app.schemas.user_info import UserInfoResponse

class UserResponse(UserBase):
    id: UUID
    is_active: bool
    created_at: datetime
    user_info: UserInfoResponse | None = None

    model_config = ConfigDict(from_attributes=True)
```

**`app/api/v1/auth.py`**
After creating a `User` in both `register` and `register-with-invite`, create a matching empty `UserInfo` row:
```python
new_user_info = UserInfo(user_id=new_user.id)
db.add(new_user_info)
```

**`app/main.py`**
Register the new `user_info` router under `/api/v1/users`.

## API Endpoints

### `GET /api/v1/users/me/info`
Returns the current user's `UserInfo`. Requires auth.

### `PATCH /api/v1/users/me/info`
Updates `first_name` and/or `last_name`. Partial update — omitted fields are left unchanged. Requires auth.

Request body:
```json
{
  "first_name": "Jane",
  "last_name": "Smith"
}
```

## Auth `/me` Response Change

`GET /api/v1/auth/me` now returns the joined `user_info` via the `UserResponse.user_info` field. The relationship must be eager-loaded or the route must explicitly join.

Add to the `/me` route query:
```python
db.query(User).options(joinedload(User.user_info)).filter(...)
```

## Notes

- All `user_info` columns are nullable — never assume `first_name`/`last_name` are set.
- Future profile fields (bio, avatar URL, timezone, etc.) go in `user_info`, not `users`.
- The `users` table stays focused on auth credentials only.
