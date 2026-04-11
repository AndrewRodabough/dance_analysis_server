# Authentication Implementation Guide

**Project**: Dance Analysis Server  
**Objective**: Add user authentication with PostgreSQL and migrate job queue from Redis to PostgreSQL  
**Approach**: Vertical slice implementation (each slice delivers a working feature end-to-end)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Changes](#architecture-changes)
3. [Prerequisites](#prerequisites)
4. [Vertical Slices](#vertical-slices)
   - [Slice 1: Database Foundation](#slice-1-database-foundation)
   - [Slice 2: User Registration & Login](#slice-2-user-registration--login)
   - [Slice 3: Protected Endpoints & User Context](#slice-3-protected-endpoints--user-context)
   - [Slice 4: Job Queue Migration](#slice-4-job-queue-migration)
   - [Slice 5: User-Job Associations](#slice-5-user-job-associations)
   - [Slice 6: Job History & Management](#slice-6-job-history--management)
5. [Testing Strategy](#testing-strategy)
6. [Security Checklist](#security-checklist)
7. [Rollback Plan](#rollback-plan)

---

## Overview

This guide provides step-by-step instructions for implementing user authentication and migrating the job queue system from Redis to PostgreSQL. The implementation is divided into **vertical slices** - each slice delivers a complete, testable feature that builds upon the previous one.

### Why PostgreSQL for Job Queue?

- **Simplification**: Fewer microservices to manage (eliminate Redis dependency)
- **ACID Guarantees**: Better data consistency for job state transitions
- **Rich Queries**: Easier to implement features like "user's job history", filtering, sorting
- **Single Source of Truth**: User data and their jobs in one database
- **Simpler Backup/Recovery**: One database to backup instead of Redis + PostgreSQL

### Migration Strategy

We'll implement PostgreSQL alongside Redis initially, then gradually migrate job queue functionality, allowing for safe rollback at any point.

---

## Architecture Changes

### Current Architecture
```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       v
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Backend   │─────>│    Redis    │─────>│   Worker    │
│   (FastAPI) │      │ (Job Queue) │      │  (Process)  │
└──────┬──────┘      └─────────────┘      └──────┬──────┘
       │                                          │
       v                                          v
┌─────────────┐                          ┌─────────────┐
│    MinIO    │<─────────────────────────│  Results    │
│  (Storage)  │                          └─────────────┘
└─────────────┘
```

### Target Architecture
```
┌─────────────┐
│   Client    │ (with JWT token)
└──────┬──────┘
       │
       v
┌─────────────┐      ┌──────────────────┐
│   Backend   │─────>│   PostgreSQL     │
│   (FastAPI) │      │  - Users         │
│             │      │  - Jobs          │
│             │      │  - Job History   │
└──────┬──────┘      └──────────────────┘
       │                     ^
       v                     │
┌─────────────┐      ┌──────┴──────┐
│    MinIO    │<─────│   Worker    │
│  (Storage)  │      │  (Process)  │
└─────────────┘      └─────────────┘
```

---

## Prerequisites

### Required Knowledge
- Python 3.10+
- FastAPI fundamentals
- SQLAlchemy ORM
- Basic SQL
- JWT authentication concepts
- Docker & Docker Compose

### Development Environment
- Docker Desktop installed and running
- Python 3.10+ with venv
- PostgreSQL client (optional, for debugging)
- API testing tool (Postman, curl, or httpie)

---

## Vertical Slices

Each slice should be implemented, tested, and committed before moving to the next. All code examples assume the project structure with `dance_analysis_server` as the root.

---

## Slice 1: Database Foundation

**Goal**: Set up PostgreSQL with SQLAlchemy and Alembic for migrations.

**Deliverable**: PostgreSQL running in Docker, connected to backend, with migration system ready.

### Step 1.1: Add PostgreSQL to Docker Compose

Edit `docker-compose.yml`:

```yaml
# Add to services section
  postgres:
    image: postgres:15-alpine
    container_name: dance-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=danceuser
      - POSTGRES_PASSWORD=dancepass
      - POSTGRES_DB=dancedb
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - dance-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U danceuser -d dancedb"]
      interval: 10s
      timeout: 5s
      retries: 5

# Add to volumes section
volumes:
  postgres-data:
    driver: local
  # ... existing volumes
```

Update the `backend` service to depend on PostgreSQL:

```yaml
  backend:
    # ... existing configuration
    environment:
      # ... existing env vars
      - DATABASE_URL=postgresql://danceuser:dancepass@postgres:5432/dancedb
      - JWT_SECRET_KEY=${JWT_SECRET_KEY:-your-secret-key-change-in-production}
      - JWT_ALGORITHM=HS256
      - ACCESS_TOKEN_EXPIRE_MINUTES=30
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
```

### Step 1.2: Update Python Dependencies

Edit `backend/requirements.txt`, add:

```
# Database
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9

# Authentication
passlib[bcrypt]==1.7.4
python-jose[cryptography]==3.3.0
python-multipart==0.0.6  # Already exists

# ... keep existing dependencies
```

### Step 1.3: Create Database Configuration

Create `backend/app/database.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://danceuser:dancepass@localhost:5432/dancedb")

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    echo=False,  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session.
    Use in FastAPI endpoints with Depends(get_db).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Step 1.4: Initialize Alembic

From `backend/` directory:

```bash
# Activate virtual environment
source ../venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize Alembic
alembic init alembic
```

Edit `backend/alembic.ini`:

```ini
# Line ~58: Update sqlalchemy.url
# sqlalchemy.url = driver://user:pass@localhost/dbname
# Comment it out, we'll use env.py instead
# sqlalchemy.url = 
```

Edit `backend/alembic/env.py`:

```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database import Base, DATABASE_URL
# Import all models here so Alembic can detect them
# from app.models.user import User  # Will add in next slice
# from app.models.job import Job    # Will add in later slice

# This is the Alembic Config object
config = context.config

# Set database URL from environment
config.set_main_option('sqlalchemy.url', DATABASE_URL)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate
target_metadata = Base.metadata

# ... rest of the file remains the same
```

### Step 1.5: Test Database Connection

Create `backend/app/core/__init__.py` (empty file).

Create `backend/app/core/config.py`:

```python
import os
from typing import Optional

class Settings:
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://danceuser:dancepass@localhost:5432/dancedb")
    
    # JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-please")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Redis (will deprecate after migration)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    # S3/MinIO
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://minio:9000")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "dance-videos")
    
    # App
    USE_MOCK_ANALYSIS: bool = os.getenv("USE_MOCK_ANALYSIS", "false").lower() == "true"

settings = Settings()
```

### Step 1.6: Start Services and Verify

```bash
# From project root
docker compose down
docker compose --profile cpu up -d postgres backend redis minio minio-init

# Check PostgreSQL is running
docker compose logs postgres

# Check backend can connect
docker compose logs backend

# Verify connection with psql (optional)
docker compose exec postgres psql -U danceuser -d dancedb -c "SELECT version();"
```

**Success Criteria**:
- ✅ PostgreSQL container starts and is healthy
- ✅ Backend connects to PostgreSQL without errors
- ✅ Alembic is initialized and configured

---

## Slice 2: User Registration & Login

**Goal**: Implement user model, registration, and login endpoints with JWT authentication.

**Deliverable**: Users can register and login, receiving a JWT token.

### Step 2.1: Create User Model

Create `backend/app/models/__init__.py`:

```python
from app.database import Base
# Import models here as they're created
from app.models.user import User
```

Create `backend/app/models/user.py`:

```python
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.database import Base


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
```

### Step 2.2: Create Database Migration

```bash
# From backend/ directory
alembic revision --autogenerate -m "Create users table"

# Review the generated migration in alembic/versions/
# Make sure it looks correct

# Apply migration
alembic upgrade head
```

Verify the table was created:

```bash
docker compose exec postgres psql -U danceuser -d dancedb -c "\dt"
docker compose exec postgres psql -U danceuser -d dancedb -c "\d users"
```

### Step 2.3: Create Authentication Utilities

Create `backend/app/core/security.py`:

```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary to encode in the token (typically {"sub": user_id})
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT access token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None
```

### Step 2.4: Create Pydantic Schemas

Create `backend/app/schemas/__init__.py` (empty file).

Create `backend/app/schemas/user.py`:

```python
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=100)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class UserInDB(UserResponse):
    hashed_password: str
```

Create `backend/app/schemas/token.py`:

```python
from pydantic import BaseModel
from typing import Optional


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None
```

### Step 2.5: Create Authentication Dependency

Create `backend/app/core/deps.py`:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.core.security import decode_access_token
from app.models.user import User

# Security scheme for Swagger UI
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Usage in endpoints:
        current_user: User = Depends(get_current_user)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if payload is None:
        raise credentials_exception
    
    user_id: Optional[int] = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to ensure user is active.
    This is the recommended dependency to use for protected endpoints.
    """
    return current_user
```

### Step 2.6: Create Authentication Endpoints

Create `backend/app/api/v1/auth.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from app.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, UserResponse
from app.schemas.token import Token
from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.config import settings

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    
    - **email**: Valid email address (unique)
    - **username**: Username between 3-50 characters (unique)
    - **password**: Password at least 8 characters
    """
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    existing_username = db.query(User).filter(User.username == user_data.username).first()
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        is_active=True,
        is_superuser=False
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@router.post("/login", response_model=Token)
def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Login and receive a JWT access token.
    
    - **email**: Registered email address
    - **password**: User password
    
    Returns a JWT token to be used in Authorization header as: `Bearer <token>`
    """
    # Find user by email
    user = db.query(User).filter(User.email == user_credentials.email).first()
    
    if not user or not verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user information.
    
    Requires: JWT token in Authorization header
    """
    return current_user
```

### Step 2.7: Register Auth Router in Main App

Edit `backend/app/main.py`:

```python
from fastapi import FastAPI
from pathlib import Path

from app.api.v1 import analyze, health, videos, auth  # Add auth import


def create_app() -> FastAPI:
    """Application factory to build the FastAPI app."""

    app = FastAPI(
        title="Dance Analysis API",
        version="1.0.0",
        description="Video dance analysis with pose estimation via microservices"
    )
    
    # Include routers with prefixes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])  # Add this
    app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])
    app.include_router(videos.router, prefix="/api/v1", tags=["videos"])

    return app


app = create_app()
```

### Step 2.8: Test Authentication Endpoints

Restart the backend:

```bash
docker compose restart backend
```

Test registration:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "securepassword123"
  }'
```

Expected response (201 Created):
```json
{
  "email": "test@example.com",
  "username": "testuser",
  "id": 1,
  "is_active": true,
  "created_at": "2024-01-15T10:30:00.123456Z"
}
```

Test login:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "securepassword123"
  }'
```

Expected response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

Test getting current user (replace `<TOKEN>` with token from login):

```bash
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer <TOKEN>"
```

Expected response:
```json
{
  "email": "test@example.com",
  "username": "testuser",
  "id": 1,
  "is_active": true,
  "created_at": "2024-01-15T10:30:00.123456Z"
}
```

**Success Criteria**:
- ✅ Users can register with email, username, and password
- ✅ Registration validates unique email and username
- ✅ Users can login and receive JWT token
- ✅ Token can be used to access `/api/v1/auth/me` endpoint
- ✅ Invalid credentials are rejected
- ✅ API documentation at http://localhost:8000/docs shows auth endpoints

---

## Slice 3: Protected Endpoints & User Context

**Goal**: Require authentication for existing video analysis endpoints.

**Deliverable**: Video upload and status endpoints require authentication.

### Step 3.1: Review Current Analyze Endpoints

First, examine the current analyze endpoints:

```bash
# View the current analyze router
cat backend/app/api/v1/analyze.py
```

You'll need to understand the current implementation to add authentication properly.

### Step 3.2: Add User Context to Analyze Endpoints

Edit `backend/app/api/v1/analyze.py` to add authentication:

```python
# Add these imports at the top
from app.core.deps import get_current_active_user
from app.models.user import User

# Example: Update the upload endpoint to require authentication
@router.post("/analyze/upload-url")
async def get_upload_url(
    filename: str,
    current_user: User = Depends(get_current_active_user),  # Add this
    # ... other existing parameters
):
    """
    Request a presigned S3 URL for direct video upload.
    
    Requires: JWT authentication
    """
    # Your existing logic, but now you have access to current_user.id
    # You can use current_user.id to organize uploads per user
    pass

# Update other endpoints similarly:
# - /analyze/confirm
# - /analyze/{job_id}/status
# - /analyze/{job_id}/result
```

**Key Changes to Make**:

1. Add `current_user: User = Depends(get_current_active_user)` to all analyze endpoints
2. Use `current_user.id` to scope resources to the user
3. Store user context with job metadata (prepare for next slice)

### Step 3.3: Update Video Endpoints

Similarly, update `backend/app/api/v1/videos.py` if it exists:

```python
from app.core.deps import get_current_active_user
from app.models.user import User

@router.get("/videos/{video_id}")
async def get_video(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
    # ... other parameters
):
    """Get video details - requires authentication."""
    # Verify video belongs to current_user
    pass
```

### Step 3.4: Test Protected Endpoints

Test without token (should fail):

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/upload-url?filename=test.mp4"
```

Expected response (401 Unauthorized):
```json
{
  "detail": "Not authenticated"
}
```

Test with valid token (should succeed):

```bash
# First login to get token
TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "securepassword123"}' \
  | jq -r '.access_token')

# Use token to access protected endpoint
curl -X POST "http://localhost:8000/api/v1/analyze/upload-url?filename=test.mp4" \
  -H "Authorization: Bearer $TOKEN"
```

**Success Criteria**:
- ✅ All analyze endpoints require authentication
- ✅ Requests without token receive 401 Unauthorized
- ✅ Requests with valid token can access endpoints
- ✅ Expired tokens are rejected

---

## Slice 4: Job Queue Migration

**Goal**: Create Job model in PostgreSQL and implement job queue functionality.

**Deliverable**: Jobs are stored in PostgreSQL instead of Redis, with status tracking.

### Step 4.1: Create Job Model

Create `backend/app/models/job.py`:

```python
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum as PyEnum
from app.database import Base


class JobStatus(str, PyEnum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), unique=True, index=True, nullable=False)  # UUID string
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Job details
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    
    # Storage paths
    video_path = Column(String(500))  # S3 path to original video
    result_path = Column(String(500))  # S3 path to result video
    data_path = Column(String(500))  # S3 path to JSON data
    
    # Error tracking
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Relationship to User
    user = relationship("User", backref="jobs")

    def __repr__(self):
        return f"<Job(id={self.id}, job_id={self.job_id}, status={self.status})>"
```

Update `backend/app/models/__init__.py`:

```python
from app.database import Base
from app.models.user import User
from app.models.job import Job, JobStatus  # Add this
```

### Step 4.2: Create Migration

```bash
# From backend/ directory
alembic revision --autogenerate -m "Create jobs table"

# Review migration
# Apply migration
alembic upgrade head
```

Verify:

```bash
docker compose exec postgres psql -U danceuser -d dancedb -c "\d jobs"
```

### Step 4.3: Create Job Schemas

Create `backend/app/schemas/job.py`:

```python
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional
from app.models.job import JobStatus


class JobCreate(BaseModel):
    """Schema for creating a new job."""
    filename: str = Field(..., min_length=1, max_length=255)


class JobResponse(BaseModel):
    """Schema for job response."""
    id: int
    job_id: str
    user_id: int
    status: JobStatus
    filename: str
    video_path: Optional[str] = None
    result_path: Optional[str] = None
    data_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class JobStatusUpdate(BaseModel):
    """Schema for updating job status."""
    status: JobStatus
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    data_path: Optional[str] = None
```

### Step 4.4: Create Job Service

Create `backend/app/services/__init__.py` (empty file).

Create `backend/app/services/job_service.py`:

```python
from sqlalchemy.orm import Session
from typing import Optional, List
import uuid
from datetime import datetime

from app.models.job import Job, JobStatus
from app.schemas.job import JobCreate, JobStatusUpdate


class JobService:
    """Service for managing job queue operations."""
    
    @staticmethod
    def create_job(db: Session, user_id: int, job_data: JobCreate) -> Job:
        """
        Create a new job in the database.
        
        Args:
            db: Database session
            user_id: ID of the user creating the job
            job_data: Job creation data
            
        Returns:
            Created Job object
        """
        job_id = str(uuid.uuid4())
        
        new_job = Job(
            job_id=job_id,
            user_id=user_id,
            filename=job_data.filename,
            status=JobStatus.PENDING
        )
        
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        
        return new_job
    
    @staticmethod
    def get_job_by_id(db: Session, job_id: str, user_id: int) -> Optional[Job]:
        """
        Get a job by job_id, ensuring it belongs to the user.
        
        Args:
            db: Database session
            job_id: UUID string of the job
            user_id: ID of the user requesting the job
            
        Returns:
            Job object or None if not found or unauthorized
        """
        return db.query(Job).filter(
            Job.job_id == job_id,
            Job.user_id == user_id
        ).first()
    
    @staticmethod
    def get_user_jobs(
        db: Session,
        user_id: int,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Job]:
        """
        Get all jobs for a user, optionally filtered by status.
        
        Args:
            db: Database session
            user_id: ID of the user
            status: Optional status filter
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip (pagination)
            
        Returns:
            List of Job objects
        """
        query = db.query(Job).filter(Job.user_id == user_id)
        
        if status:
            query = query.filter(Job.status == status)
        
        return query.order_by(Job.created_at.desc()).limit(limit).offset(offset).all()
    
    @staticmethod
    def update_job_status(
        db: Session,
        job_id: str,
        status_update: JobStatusUpdate
    ) -> Optional[Job]:
        """
        Update job status and related fields.
        
        Args:
            db: Database session
            job_id: UUID string of the job
            status_update: Status update data
            
        Returns:
            Updated Job object or None if not found
        """
        job = db.query(Job).filter(Job.job_id == job_id).first()
        
        if not job:
            return None
        
        # Update status
        job.status = status_update.status
        
        # Update timestamps based on status
        if status_update.status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status_update.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            job.completed_at = datetime.utcnow()
        
        # Update optional fields
        if status_update.error_message:
            job.error_message = status_update.error_message
        if status_update.result_path:
            job.result_path = status_update.result_path
        if status_update.data_path:
            job.data_path = status_update.data_path
        
        db.commit()
        db.refresh(job)
        
        return job
    
    @staticmethod
    def get_next_pending_job(db: Session) -> Optional[Job]:
        """
        Get the next pending job for processing (FIFO queue).
        
        Returns:
            Oldest pending Job or None if queue is empty
        """
        return db.query(Job).filter(
            Job.status == JobStatus.PENDING
        ).order_by(Job.created_at.asc()).first()
    
    @staticmethod
    def delete_job(db: Session, job_id: str, user_id: int) -> bool:
        """
        Delete a job (soft delete by setting status or hard delete).
        
        Args:
            db: Database session
            job_id: UUID string of the job
            user_id: ID of the user requesting deletion
            
        Returns:
            True if deleted, False if not found or unauthorized
        """
        job = db.query(Job).filter(
            Job.job_id == job_id,
            Job.user_id == user_id
        ).first()
        
        if not job:
            return False
        
        db.delete(job)
        db.commit()
        return True
```

### Step 4.5: Update Analyze Endpoints to Use Job Service

Edit `backend/app/api/v1/analyze.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.deps import get_current_active_user
from app.models.user import User
from app.schemas.job import JobCreate, JobResponse
from app.services.job_service import JobService

router = APIRouter()


@router.post("/analyze/upload-url", response_model=dict)
async def get_upload_url(
    filename: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Request a presigned S3 URL for direct video upload.
    
    Steps:
    1. Create a job in the database
    2. Generate a presigned S3 upload URL
    3. Return both the job_id and presigned URL
    """
    # Create job in database
    job_data = JobCreate(filename=filename)
    job = JobService.create_job(db, current_user.id, job_data)
    
    # TODO: Generate presigned S3 URL using job.job_id
    # For now, return placeholder
    presigned_url = f"http://minio:9000/dance-videos/uploads/{job.job_id}/{filename}"
    
    return {
        "job_id": job.job_id,
        "upload_url": presigned_url,
        "expires_in": 900  # 15 minutes
    }


@router.post("/analyze/confirm", response_model=JobResponse)
async def confirm_upload(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Confirm video upload and queue job for processing.
    """
    job = JobService.get_job_by_id(db, job_id, current_user.id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # TODO: Verify file exists in S3
    # TODO: Trigger worker to process job
    
    return job


@router.get("/analyze/{job_id}/status", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get the status of a video analysis job."""
    job = JobService.get_job_by_id(db, job_id, current_user.id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return job


@router.get("/analyze/{job_id}/result")
async def get_job_result(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get the result of a completed video analysis job."""
    job = JobService.get_job_by_id(db, job_id, current_user.id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed (current status: {job.status})"
        )
    
    # TODO: Return presigned URLs for result files
    return {
        "job_id": job.job_id,
        "status": job.status,
        "result_video_url": job.result_path,
        "data_url": job.data_path
    }
```

### Step 4.6: Test Job Creation

```bash
# Login
TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "securepassword123"}' \
  | jq -r '.access_token')

# Request upload URL (creates job)
curl -X POST "http://localhost:8000/api/v1/analyze/upload-url?filename=test_video.mp4" \
  -H "Authorization: Bearer $TOKEN"

# Check job status (use job_id from previous response)
curl -X GET "http://localhost:8000/api/v1/analyze/<job_id>/status" \
  -H "Authorization: Bearer $TOKEN"
```

Verify job in database:

```bash
docker compose exec postgres psql -U danceuser -d dancedb -c "SELECT * FROM jobs;"
```

**Success Criteria**:
- ✅ Jobs are created in PostgreSQL
- ✅ Jobs have unique job_id (UUID)
- ✅ Jobs are associated with user_id
- ✅ Job status can be queried
- ✅ Users can only access their own jobs

---

## Slice 5: User-Job Associations

**Goal**: Implement job listing and management for users.

**Deliverable**: Users can view their job history and manage their jobs.

### Step 5.1: Create Jobs Endpoints

Create `backend/app/api/v1/jobs.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.core.deps import get_current_active_user
from app.models.user import User
from app.models.job import JobStatus
from app.schemas.job import JobResponse
from app.services.job_service import JobService

router = APIRouter()


@router.get("/jobs", response_model=List[JobResponse])
async def list_user_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all jobs for the current user.
    
    - **status**: Optional filter by job status
    - **limit**: Maximum results (1-100, default 50)
    - **offset**: Pagination offset (default 0)
    """
    jobs = JobService.get_user_jobs(
        db=db,
        user_id=current_user.id,
        status=status,
        limit=limit,
        offset=offset
    )
    return jobs


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_details(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get details of a specific job."""
    job = JobService.get_job_by_id(db, job_id, current_user.id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return job


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a job.
    
    Note: This will delete job metadata. Consider implementing cleanup
    of associated S3 files as well.
    """
    deleted = JobService.delete_job(db, job_id, current_user.id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return None
```

### Step 5.2: Register Jobs Router

Edit `backend/app/main.py`:

```python
from fastapi import FastAPI
from pathlib import Path

from app.api.v1 import analyze, health, videos, auth, jobs  # Add jobs


def create_app() -> FastAPI:
    """Application factory to build the FastAPI app."""

    app = FastAPI(
        title="Dance Analysis API",
        version="1.0.0",
        description="Video dance analysis with pose estimation via microservices"
    )
    
    # Include routers with prefixes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
    app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])  # Add this
    app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])
    app.include_router(videos.router, prefix="/api/v1", tags=["videos"])

    return app


app = create_app()
```

### Step 5.3: Test Job Listing

```bash
# Login
TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "securepassword123"}' \
  | jq -r '.access_token')

# Create a few jobs
for i in {1..3}; do
  curl -X POST "http://localhost:8000/api/v1/analyze/upload-url?filename=video_$i.mp4" \
    -H "Authorization: Bearer $TOKEN"
done

# List all jobs
curl -X GET "http://localhost:8000/api/v1/jobs" \
  -H "Authorization: Bearer $TOKEN"

# Filter by status
curl -X GET "http://localhost:8000/api/v1/jobs?status=pending" \
  -H "Authorization: Bearer $TOKEN"

# Get specific job
curl -X GET "http://localhost:8000/api/v1/jobs/<job_id>" \
  -H "Authorization: Bearer $TOKEN"

# Delete job
curl -X DELETE "http://localhost:8000/api/v1/jobs/<job_id>" \
  -H "Authorization: Bearer $TOKEN"
```

**Success Criteria**:
- ✅ Users can list all their jobs
- ✅ Jobs can be filtered by status
- ✅ Pagination works with limit/offset
- ✅ Users cannot see other users' jobs
- ✅ Jobs can be deleted by their owner

---

## Slice 6: Job History & Management

**Goal**: Update worker to use PostgreSQL for job queue instead of Redis.

**Deliverable**: Video processing worker picks jobs from PostgreSQL and updates status.

### Step 6.1: Create Worker Database Connection

Create `video_processing/database.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://danceuser:dancepass@postgres:5432/dancedb")

# Create engine for worker
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session for worker."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()
```

### Step 6.2: Update Worker to Poll PostgreSQL

Create `video_processing/worker.py`:

```python
import time
import sys
import os
from pathlib import Path

# Add backend to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from database import SessionLocal
from app.models.job import Job, JobStatus
from app.services.job_service import JobService

# Import your existing processing logic
from pose_estimator import process_video  # Adjust import as needed


def poll_and_process_jobs(poll_interval: int = 5):
    """
    Continuously poll PostgreSQL for pending jobs and process them.
    
    Args:
        poll_interval: Seconds to wait between polls
    """
    print(f"Worker started. Polling every {poll_interval} seconds...")
    
    while True:
        db = SessionLocal()
        try:
            # Get next pending job
            job = JobService.get_next_pending_job(db)
            
            if job:
                print(f"Processing job {job.job_id}...")
                process_job(db, job)
            else:
                # No pending jobs, wait before next poll
                time.sleep(poll_interval)
                
        except Exception as e:
            print(f"Error in worker loop: {e}")
            time.sleep(poll_interval)
        finally:
            db.close()


def process_job(db, job: Job):
    """
    Process a single job.
    
    Args:
        db: Database session
        job: Job object to process
    """
    try:
        # Update status to processing
        JobService.update_job_status(
            db,
            job.job_id,
            JobStatusUpdate(status=JobStatus.PROCESSING)
        )
        
        # Download video from S3
        # Process video
        # Upload results to S3
        # This is where you'd integrate your existing pose estimation logic
        
        # For now, simulate processing
        print(f"Processing video: {job.filename}")
        time.sleep(5)  # Simulate work
        
        # Update status to completed
        JobService.update_job_status(
            db,
            job.job_id,
            JobStatusUpdate(
                status=JobStatus.COMPLETED,
                result_path=f"s3://dance-videos/results/{job.job_id}/result.mp4",
                data_path=f"s3://dance-videos/results/{job.job_id}/data.json"
            )
        )
        
        print(f"Job {job.job_id} completed successfully")
        
    except Exception as e:
        print(f"Job {job.job_id} failed: {e}")
        
        # Update status to failed
        JobService.update_job_status(
            db,
            job.job_id,
            JobStatusUpdate(
                status=JobStatus.FAILED,
                error_message=str(e)
            )
        )


if __name__ == "__main__":
    poll_and_process_jobs()
```

### Step 6.3: Update Worker Dockerfile

Edit `video_processing/Dockerfile` to include database dependencies:

```dockerfile
# Add to requirements or install
RUN pip install sqlalchemy psycopg2-binary
```

### Step 6.4: Update Docker Compose Worker Configuration

Edit `docker-compose.yml` to update worker service:

```yaml
# Update worker common configuration
x-video-worker-common: &video-worker-common
  build:
    context: ./video_processing
    dockerfile: Dockerfile
  volumes:
    - ./video_processing:/workspace
    - video-temp:/workspace/temp
  environment:
    - DATABASE_URL=postgresql://danceuser:dancepass@postgres:5432/dancedb
    # ... other env vars
  depends_on:
    postgres:
      condition: service_healthy
    minio:
      condition: service_healthy
  networks:
    - dance-network
  restart: unless-stopped
  command: ["python", "worker.py"]  # New command
```

### Step 6.5: Deprecate Redis

Once worker is using PostgreSQL:

1. Comment out Redis from `docker-compose.yml` (or remove it)
2. Remove Redis dependencies from `requirements.txt`
3. Remove Redis-related environment variables

### Step 6.6: Test End-to-End Flow

```bash
# Rebuild and restart services
docker compose down
docker compose --profile cpu up --build -d

# Login
TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "securepassword123"}' \
  | jq -r '.access_token')

# Create job
JOB_RESPONSE=$(curl -X POST "http://localhost:8000/api/v1/analyze/upload-url?filename=test.mp4" \
  -H "Authorization: Bearer $TOKEN")
JOB_ID=$(echo $JOB_RESPONSE | jq -r '.job_id')

# Confirm upload (in reality, you'd upload to S3 first)
curl -X POST "http://localhost:8000/api/v1/analyze/confirm?job_id=$JOB_ID" \
  -H "Authorization: Bearer $TOKEN"

# Watch worker logs
docker compose logs -f video-worker-cpu

# Check job status (should change from pending -> processing -> completed)
watch -n 2 "curl -s -X GET http://localhost:8000/api/v1/jobs/$JOB_ID \
  -H 'Authorization: Bearer $TOKEN' | jq '.status'"
```

**Success Criteria**:
- ✅ Worker polls PostgreSQL for jobs
- ✅ Worker updates job status (pending → processing → completed/failed)
- ✅ Job status is visible via API
- ✅ Multiple jobs are processed in order (FIFO)
- ✅ Failed jobs are marked with error messages
- ✅ Redis is no longer needed

---

## Testing Strategy

### Unit Tests

Create `backend/tests/test_auth.py`:

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db

# Use in-memory SQLite for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_register_user():
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "password123"
        }
    )
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"


def test_login_user():
    # First register
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "test2@example.com",
            "username": "testuser2",
            "password": "password123"
        }
    )
    
    # Then login
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test2@example.com",
            "password": "password123"
        }
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


# Add more tests for jobs, protected endpoints, etc.
```

### Integration Tests

Create `backend/tests/test_job_flow.py`:

```python
def test_complete_job_flow():
    """Test complete flow: register, login, create job, check status."""
    # 1. Register
    # 2. Login
    # 3. Request upload URL
    # 4. Confirm upload
    # 5. Check status
    # 6. Verify job in list
    pass
```

### Load Testing (Optional)

Use `locust` or `k6` to test:
- Concurrent job creation
- Database connection pooling
- Worker throughput

---

## Security Checklist

### Before Production Deployment

- [ ] Change `JWT_SECRET_KEY` to a strong, random value (use `openssl rand -hex 32`)
- [ ] Store secrets in environment variables or secret management system (never in code)
- [ ] Change PostgreSQL password from default
- [ ] Enable SSL/TLS for PostgreSQL connections
- [ ] Set up CORS properly for frontend domain
- [ ] Implement rate limiting on auth endpoints (e.g., `slowapi`)
- [ ] Add email verification for registration
- [ ] Implement password reset flow
- [ ] Add password strength requirements
- [ ] Consider refresh tokens for longer sessions
- [ ] Enable HTTPS/TLS for API
- [ ] Implement request logging and monitoring
- [ ] Add input validation and sanitization
- [ ] Set up database backups
- [ ] Implement account lockout after failed login attempts
- [ ] Add audit logging for sensitive operations

### Optional Security Enhancements

- [ ] Two-factor authentication (2FA)
- [ ] OAuth2 integration (Google, GitHub, etc.)
- [ ] API key system for programmatic access
- [ ] Role-based access control (RBAC)
- [ ] File type validation for uploads
- [ ] Virus scanning for uploaded files
- [ ] S3 bucket encryption

---

## Rollback Plan

### If Issues Occur During Migration

**Phase 1-3 (Auth Added, Redis Still Active)**:
- Remove auth requirements from endpoints
- Revert database migrations: `alembic downgrade -1`
- Restart services

**Phase 4-5 (PostgreSQL Job Queue, Redis Still Available)**:
- Switch worker back to Redis
- Keep both systems running temporarily
- Migrate data back to Redis if needed

**Phase 6 (Redis Removed)**:
- Re-add Redis to `docker-compose.yml`
- Switch worker back to Redis mode
- Export PostgreSQL jobs, import to Redis

### Database Backup Before Each Slice

```bash
# Backup PostgreSQL
docker compose exec postgres pg_dump -U danceuser dancedb > backup_slice_X.sql

# Restore if needed
docker compose exec -T postgres psql -U danceuser dancedb < backup_slice_X.sql
```

---

## Appendix

### Useful Commands

```bash
# View database tables
docker compose exec postgres psql -U danceuser -d dancedb -c "\dt"

# Query users
docker compose exec postgres psql -U danceuser -d dancedb -c "SELECT * FROM users;"

# Query jobs
docker compose exec postgres psql -U danceuser -d dancedb -c "SELECT job_id, status, created_at FROM jobs ORDER BY created_at DESC LIMIT 10;"

# Reset database (CAUTION)
docker compose down -v  # Removes volumes
docker compose up -d postgres
docker compose exec backend alembic upgrade head

# Generate new JWT secret
openssl rand -hex 32
```

### Environment Variables Reference

```bash
# Backend .env example
DATABASE_URL=postgresql://danceuser:dancepass@postgres:5432/dancedb
JWT_SECRET_KEY=your-super-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=dance-videos
```

### Additional Resources

- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [SQLAlchemy ORM Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [PostgreSQL Best Practices](https://wiki.postgresql.org/wiki/Don't_Do_This)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)

---

## Summary

This guide provides a **vertical slice** approach to implementing authentication and migrating to PostgreSQL:

1. **Slice 1**: Database foundation (PostgreSQL + SQLAlchemy + Alembic)
2. **Slice 2**: User registration and login (JWT authentication)
3. **Slice 3**: Protect existing endpoints (require auth)
4. **Slice 4**: Job queue in PostgreSQL (create Job model and API)
5. **Slice 5**: User-job associations (job history, management)
6. **Slice 6**: Worker migration (poll PostgreSQL instead of Redis)

Each slice is independently testable and provides incremental value. If issues arise, you can rollback to the previous working slice.

**Key Benefits**:
- ✅ Fewer microservices (no Redis)
- ✅ Better queryability (SQL vs Redis)
- ✅ Single source of truth (users + jobs)
- ✅ ACID guarantees for job state
- ✅ Easier to implement features (pagination, filtering, search)

Good luck with the implementation! 🚀
