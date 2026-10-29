"""Start local web UI: http://127.0.0.1:8000

Usage: python scripts/run_web.py [--host 0.0.0.0] [--port 8000]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=False, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
