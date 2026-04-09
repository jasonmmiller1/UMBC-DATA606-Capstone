from __future__ import annotations

import socket
import sys

from app.runtime import PORT


def main() -> int:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=3.0):
            return 0
    except OSError as exc:
        print(f"Container healthcheck failed on port {PORT}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
