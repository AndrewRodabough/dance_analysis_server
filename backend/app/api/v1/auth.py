"""Authentication endpoints - registration, login, logout, and user info."""

from datetime import timedelta

from app.core.config import settings
from app.core.deps import get_current_active_user
from app.core.logging import log_auth_event
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    get_password_hash,
    verify_password,
)
from app.database import get_db
from app.models.user import User
from app.schemas.token import Token
from app.schemas.user import UserCreate, UserLogin, UserResponse
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

router = APIRouter()

REFRESH_TOKEN_COOKIE_NAME = "refresh_token"


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
        log_auth_event(action="register", email=user_data.email, success=False, error="Email already registered")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Check if username already exists
    existing_username = db.query(User).filter(User.username == user_data.username).first()
    if existing_username:
        log_auth_event(action="register", email=user_data.email, success=False, error="Username already taken")
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

    log_auth_event(action="register", email=user_data.email, user_id=new_user.id, success=True)

    return new_user


@router.post("/login", response_model=Token)
def login(
    user_credentials: UserLogin,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Login and receive a JWT access token.

    - **email**: Registered email address
    - **password**: User password

    Returns a JWT token to be used in Authorization header as: `Bearer <token>`

    Sets a HTTP-only cookie containing the refresh token.
    """
    # Find user by email
    user = db.query(User).filter(User.email == user_credentials.email).first()

    if not user or not verify_password(user_credentials.password, user.hashed_password):
        log_auth_event(action="login", email=user_credentials.email, success=False, error="Invalid credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        log_auth_event(action="login", email=user_credentials.email, user_id=user.id, success=False, error="Inactive account")
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

    # Create refresh token and set it in an HTTP-only cookie
    refresh_token_expires_days = getattr(settings, "REFRESH_TOKEN_EXPIRE_DAYS", 7)
    refresh_token_expires = timedelta(days=refresh_token_expires_days)
    refresh_token = create_refresh_token(
        data={"sub": str(user.id)},
        expires_delta=refresh_token_expires,
    )

    # max_age in seconds
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=getattr(settings, "COOKIE_SECURE", False),
        samesite=getattr(settings, "COOKIE_SAMESITE", "lax"),
        max_age=int(refresh_token_expires.total_seconds()),
        path="/",
    )

    log_auth_event(action="login", user_id=user.id, success=True)

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user information.

    Requires: JWT token in Authorization header
    """
    return current_user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response):
    """
    Logout the current user.

    This clears the HTTP-only refresh token cookie so that the client
    can discard any stored access token and treat the user as logged out.
    """
    response.delete_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        path="/",
    )
    # No body is returned for 204 responses
    return None


@router.post("/refresh", response_model=Token)
def refresh_token(request: Request, response: Response):
    """
    Refresh access token using a valid (non-expired) refresh token stored
    in an HTTP-only cookie.

    If the refresh token is expired or invalid, returns 401 so the client
    can log the user out and clears the refresh cookie.
    """
    refresh_token = request.cookies.get(REFRESH_TOKEN_COOKIE_NAME)
    if not refresh_token:
        # No refresh token cookie present; client should treat as logged out
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    decoded = decode_refresh_token(refresh_token)
    if not decoded:
        # Invalid or expired refresh token: clear cookie and force logout
        response.delete_cookie(
            key=REFRESH_TOKEN_COOKIE_NAME,
            path="/",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = decoded.get("sub")
    if user_id is None:
        response.delete_cookie(
            key=REFRESH_TOKEN_COOKIE_NAME,
            path="/",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create new access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user_id)},
        expires_delta=access_token_expires,
    )

    # Optional: rotate refresh token for better security
    refresh_token_expires_days = getattr(settings, "REFRESH_TOKEN_EXPIRE_DAYS", 7)
    new_refresh_token_expires = timedelta(days=refresh_token_expires_days)
    new_refresh_token = create_refresh_token(
        data={"sub": str(user_id)},
        expires_delta=new_refresh_token_expires,
    )

    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=new_refresh_token,
        httponly=True,
        secure=getattr(settings, "COOKIE_SECURE", False),
        samesite=getattr(settings, "COOKIE_SAMESITE", "lax"),
        max_age=int(new_refresh_token_expires.total_seconds()),
        path="/",
    )

    log_auth_event(action="refresh", user_id=int(user_id), success=True)

    return {"access_token": access_token, "token_type": "bearer"}
