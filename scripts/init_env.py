from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = {
    "AEROLOCATE_JWT_SECRET": lambda: secrets.token_urlsafe(48),
    "POSTGRES_PASSWORD": lambda: secrets.token_urlsafe(24),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create .env from .env.example with fresh random secrets")
    parser.add_argument("--force", action="store_true", help="overwrite an existing .env")
    arguments = parser.parse_args()
    target = ROOT / ".env"
    if target.exists() and not arguments.force:
        print(f"{target} already exists. Use --force to replace it, which discards the current secrets.")
        return 1
    lines = []
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in GENERATED and not value.strip():
            line = f"{key}={GENERATED[key]()}"
        lines.append(line)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {target} with new secrets. Add any flight data API keys to it before starting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
