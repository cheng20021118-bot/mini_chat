# core/paths.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORAGE_DIR = PROJECT_ROOT / "storage"
MEMORY_PATH = STORAGE_DIR / "memory.json"
