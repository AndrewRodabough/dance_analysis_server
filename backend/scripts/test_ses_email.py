"""Quick script to verify AWS SES credentials and send a test email.

Usage:
    python scripts/test_ses_email.py recipient@yourdomain.com
"""

import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Load .env from the repo root (two levels up from backend/scripts/)
env_file = Path(__file__).resolve().parents[2] / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, ".")
from app.core.config import settings


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_ses_email.py <to_address>")
        sys.exit(1)

    to_address = sys.argv[1]

    print(f"Region:    {settings.AWS_SES_REGION}")
    print(f"From:      invite@{settings.EMAIL_DOMAIN}")
    print(f"To:        {to_address}")
    print()

    ses = boto3.client(
        "ses",
        region_name=settings.AWS_SES_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    try:
        response = ses.send_email(
            Source=f"invite@{settings.EMAIL_DOMAIN}",
            Destination={"ToAddresses": [to_address]},
            Message={
                "Subject": {"Data": "Dance Coach — SES test email"},
                "Body": {
                    "Text": {
                        "Data": "If you received this, AWS SES is configured correctly."
                    }
                },
            },
        )
        print(f"Sent. Message ID: {response['MessageId']}")
    except ClientError as e:
        print(f"Failed: {e.response['Error']['Code']} — {e.response['Error']['Message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
