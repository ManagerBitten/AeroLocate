from __future__ import annotations

import argparse
import getpass
import sys

from sqlalchemy import select

from backend.api.security import hash_password
from backend.models.database import init_database, session_scope
from backend.models.entities import AuditLog, User

MIN_PASSWORD_LENGTH = 12


def _read_password(from_stdin: bool) -> str:
    if from_stdin:
        return sys.stdin.readline().rstrip("\n")
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords do not match.")
    return first


def create_user(email: str, admin: bool, from_stdin: bool) -> None:
    password = _read_password(from_stdin)
    if len(password) < MIN_PASSWORD_LENGTH:
        raise SystemExit(f"Use at least {MIN_PASSWORD_LENGTH} characters.")
    init_database()
    with session_scope() as session:
        normalized = email.strip().lower()
        if session.scalar(select(User).where(User.email == normalized)) is not None:
            raise SystemExit(f"{normalized} already exists. Use 'promote' to make it an administrator.")
        user = User(email=normalized, password_hash=hash_password(password), is_admin=admin)
        session.add(user)
        session.flush()
        session.add(AuditLog(user_id=user.id, action="user.created_by_cli", detail={"admin": admin}))
    print(f"Created {'administrator' if admin else 'user'} {normalized}.")


def promote(email: str) -> None:
    init_database()
    with session_scope() as session:
        user = session.scalar(select(User).where(User.email == email.strip().lower()))
        if user is None:
            raise SystemExit(f"No account for {email}.")
        user.is_admin = True
        session.add(AuditLog(user_id=user.id, action="user.promoted_by_cli", detail={}))
    print(f"{email} is now an administrator.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m backend.cli", description="Manage AeroLocate accounts")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-user", help="create an account")
    create.add_argument("email")
    create.add_argument("--admin", action="store_true", help="allow viewing the audit log")
    create.add_argument("--password-stdin", action="store_true", help="read the password from standard input")
    elevate = commands.add_parser("promote", help="make an existing account an administrator")
    elevate.add_argument("email")
    arguments = parser.parse_args()
    if arguments.command == "create-user":
        create_user(arguments.email, arguments.admin, arguments.password_stdin)
    else:
        promote(arguments.email)


if __name__ == "__main__":
    main()
