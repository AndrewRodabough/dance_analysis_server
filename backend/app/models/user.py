"""User database model."""

from enum import Enum as PyEnum

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class UserRelationRole(str, PyEnum):
    COACH = "coach"
    STUDENT = "student"
    PARTNER = "partner"


class UserRelationStatus(str, PyEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, username={self.username})>"


class UserRelation(Base):
    __tablename__ = "user_relations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    related_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(Enum(UserRelationRole, name="user_relation_role"), nullable=False)
    status = Column(
        Enum(UserRelationStatus, name="user_relation_status"),
        nullable=False,
        default=UserRelationStatus.PENDING,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self):
        return (
            f"<UserRelation(id={self.id}, user_id={self.user_id}, "
            f"related_user_id={self.related_user_id}, role={self.role}, status={self.status})>"
        )


class Invite(Base):
    __tablename__ = "invites"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(255), unique=True, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    email = Column(String(255), nullable=True)
    phone_number = Column(String(32), nullable=True)
    role = Column(Enum(UserRelationRole, name="invite_role"), nullable=False)
    status = Column(
        Enum(UserRelationStatus, name="invite_status"),
        nullable=False,
        default=UserRelationStatus.PENDING,
    )
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self):
        return (
            f"<Invite(id={self.id}, created_by={self.created_by}, email={self.email}, "
            f"phone_number={self.phone_number}, role={self.role}, status={self.status}, "
            f"expires_at={self.expires_at})>"
        )
