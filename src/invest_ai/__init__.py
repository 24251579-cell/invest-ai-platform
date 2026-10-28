import os
from pathlib import Path

from dotenv import load_dotenv

# Must run before transformers / huggingface_hub are imported: they read these at import time.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
os.environ.setdefault("HF_HOME", r"C:\hf_cache")
# Xet downloads hang on this network; plain HTTP works.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Symlinks need admin / Developer Mode on Windows; the cache works without them.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
