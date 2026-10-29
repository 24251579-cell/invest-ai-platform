"""Sync CSV artifacts from data/ into PostgreSQL.

Usage: python scripts/sync_db.py
Requires DATABASE_URL (default: postgresql+psycopg://invest:invest@localhost:5432/invest)
"""

import json
import logging
import sys

from invest_ai.db import ping, sync_from_csv


def main() -> None:
    status = ping()
    if not status["ok"]:
        print(f"PostgreSQL unavailable: {status.get('error')}", file=sys.stderr)
        sys.exit(1)
    stats = sync_from_csv()
    print(json.dumps({"ping": status["url_host"], "synced": stats, "counts": ping()["counts"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
