#!/usr/bin/env python3
"""Manage chatbot users.

Usage:
    uv run python manage_users.py add <username> <password> [<display_name>]
"""
import sys

from auth import add_user


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "add":
        if len(sys.argv) < 4:
            print("Usage: python manage_users.py add <username> <password> [<display_name>]")
            sys.exit(1)
        username = sys.argv[2]
        password = sys.argv[3]
        display_name = sys.argv[4] if len(sys.argv) > 4 else username
        try:
            add_user(username, password, display_name)
            print(f"User '{username}' (display: '{display_name}') saved to data/users.yaml")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
