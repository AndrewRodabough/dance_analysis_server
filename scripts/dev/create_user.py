"""Create a user via the auth registration endpoint."""

import argparse
import json
import sys

import requests

API_BASE_URL = "http://localhost:8000"
REGISTER_ENDPOINT = f"{API_BASE_URL}/api/v1/auth/register"


def register_user(email: str, username: str, password: str) -> dict:
    response = requests.post(
        REGISTER_ENDPOINT,
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "email": email,
            "username": username,
            "password": password
        })
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a new user.")
    parser.add_argument("--email", required=True, help="User email")
    parser.add_argument("--username", required=True, help="Username")
    parser.add_argument("--password", required=True, help="Password (min 8 chars)")
    args = parser.parse_args()

    try:
        user = register_user(args.email, args.username, args.password)
        print(json.dumps(user, indent=2))
        return 0
    except requests.RequestException as exc:
        print(f"Error creating user: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
