"""Request a JWT auth token for local development."""

import argparse
import json
import sys

import requests

API_BASE_URL = "http://localhost:8000"
LOGIN_ENDPOINT = f"{API_BASE_URL}/api/v1/auth/login"


def fetch_token(email: str, password: str) -> str:
    response = requests.post(
        LOGIN_ENDPOINT,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"email": email, "password": password})
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("access_token")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a JWT access token.")
    parser.add_argument("--email", required=True, help="User email")
    parser.add_argument("--password", required=True, help="User password")
    args = parser.parse_args()

    try:
        token = fetch_token(args.email, args.password)
        if not token:
            print("No access_token in response.")
            return 1
        print(token)
        return 0
    except requests.RequestException as exc:
        print(f"Error fetching token: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
