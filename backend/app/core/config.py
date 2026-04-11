"""Application configuration settings."""

import os


class Settings:
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://danceuser:dancepass@localhost:5432/dancedb"
    )

    # JWT
    JWT_SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY",
        "dev-secret-key-change-in-production"
    )
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )

    # Redis (will deprecate after migration)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # S3/MinIO (internal/local storage)
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://minio:9000")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "dance-videos")

    # Cloudflare R2 (video upload/download)
    R2_ENDPOINT: str = os.getenv("R2_ENDPOINT", "https://example.r2.cloudflarestorage.com")
    R2_ACCESS_KEY: str = os.getenv("R2_ACCESS_KEY", "")
    R2_SECRET_KEY: str = os.getenv("R2_SECRET_KEY", "")
    R2_BUCKET: str = os.getenv("R2_BUCKET", "dance-videos")

    # App
    USE_MOCK_ANALYSIS: bool = os.getenv("USE_MOCK_ANALYSIS", "false").lower() == "true"
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    CORS_ORIGINS: list = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000"
    ).split(",")

    # Email (AWS SES)
    AWS_SES_REGION: str = os.getenv("AWS_SES_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    EMAIL_DOMAIN: str = os.getenv("EMAIL_DOMAIN", "")

    # Deep link configuration (Universal Links / App Links)
    # iOS: "<TEAM_ID>.<BUNDLE_ID>" e.g. "ABCDE12345.com.example.dancecoach"
    IOS_APP_ID: str = os.getenv("IOS_APP_ID", "")
    # Android package name e.g. "com.example.dancecoach"
    ANDROID_PACKAGE_NAME: str = os.getenv("ANDROID_PACKAGE_NAME", "")
    # SHA-256 cert fingerprint for the Android signing cert (colon-separated hex)
    ANDROID_CERT_FINGERPRINT: str = os.getenv("ANDROID_CERT_FINGERPRINT", "")


settings = Settings()
