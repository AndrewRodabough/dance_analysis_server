#!/usr/bin/env python3
# scripts/generate_openapi.py
"""Generate OpenAPI spec without running the server."""

import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import app (doesn't start server, just loads it)
from app.main import app

if __name__ == "__main__":
    # Get OpenAPI schema
    openapi_schema = app.openapi()

    # Save to file
    output_file = Path(__file__).parent.parent.parent / "docs" / "openapi.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"✓ OpenAPI spec generated: {output_file}")
