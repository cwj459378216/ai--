import os
import subprocess
import sys
from pathlib import Path

from .vectorize_cli import main as vectorize_main


ROOT_DIR = Path(__file__).resolve().parents[1]
VECTORS_DIR = ROOT_DIR / "vectors"
DB_PATH = VECTORS_DIR / "milvus.db"
LOCK_PATH = VECTORS_DIR / ".milvus.db.lock"


def _stop_streamlit() -> None:
    try:
        subprocess.run(["pkill", "-f", "streamlit run gui.py"], check=False)
    except Exception:
        pass


def _cleanup_db_files() -> None:
    try:
        if DB_PATH.exists():
            DB_PATH.unlink()
    except Exception:
        pass
    try:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    except Exception:
        pass
    try:
        os.chmod(VECTORS_DIR, 0o755)
    except Exception:
        pass


def main() -> int:
    _stop_streamlit()
    _cleanup_db_files()
    return vectorize_main()


if __name__ == "__main__":
    raise SystemExit(main())
