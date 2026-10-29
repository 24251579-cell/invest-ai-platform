#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for PostgreSQL..."
python - <<'PY'
import os, time, sys
from sqlalchemy import create_engine, text
url = os.environ.get("DATABASE_URL", "postgresql+psycopg://invest:invest@db:5432/invest")
for i in range(60):
    try:
        with create_engine(url).connect() as conn:
            conn.execute(text("SELECT 1"))
        print("PostgreSQL is ready")
        sys.exit(0)
    except Exception as exc:
        print(f"  attempt {i+1}/60: {exc}")
        time.sleep(2)
print("PostgreSQL did not become ready", file=sys.stderr)
sys.exit(1)
PY

if [[ "${SYNC_DB_ON_START:-1}" == "1" ]]; then
  echo "Syncing CSV -> PostgreSQL (if data/ present)..."
  python scripts/sync_db.py || echo "sync skipped/failed (OK on first empty start)"
fi

exec python scripts/run_web.py --host 0.0.0.0 --port 8000
