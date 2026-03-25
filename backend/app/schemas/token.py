"""Pydantic schemas for JWT token responses."""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    # For HTTP-only cookie-based refresh tokens, this remains optional and
    # is typically not sent in responses, but kept for flexibility.
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    user_id: Optional[UUID] = None
