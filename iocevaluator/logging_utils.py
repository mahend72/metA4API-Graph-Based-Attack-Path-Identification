from __future__ import annotations

import logging
import os


def setup_logging(level: str | int | None = None) -> None:
    """Configure a simple, consistent logging format.

    Priority: explicit `level` arg > LOG_LEVEL env var > INFO.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
