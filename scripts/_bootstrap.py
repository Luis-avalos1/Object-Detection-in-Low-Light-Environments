"""Shared import bootstrap so scripts can `import lowlight` from repo root."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
