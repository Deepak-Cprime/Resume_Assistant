"""Atomic-ish file replacement that survives Windows' habit of briefly locking new files.

On Windows, renaming a just-written file or folder can fail with "Access is denied" while a
virus scanner or the search indexer still has it open; the lock clears within milliseconds.
Every save in the project (indexes, JSON, Markdown) goes through these helpers.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

ATTEMPTS = 8
BASE_DELAY = 0.05  # seconds; grows linearly with each retry


def replace(src: Path, dst: Path) -> None:
    """os.replace with retries on PermissionError (a file; use swap_directory for folders)."""
    for attempt in range(ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(BASE_DELAY * (attempt + 1))


def swap_directory(tmp: Path, dst: Path) -> None:
    """Make `tmp` the new `dst` folder, replacing what was there.

    Retries the rename; if the folder stays locked, falls back to copying it over, so a
    save never fails just because a scanner is slow to let go."""
    for attempt in range(ATTEMPTS):
        shutil.rmtree(dst, ignore_errors=True)
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            time.sleep(BASE_DELAY * (attempt + 1))
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(tmp, dst)
    shutil.rmtree(tmp, ignore_errors=True)
