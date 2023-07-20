# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Logger."""

from __future__ import annotations

import logging
import os

_formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)05d: %(name)s %(levelname)s %(pathname)s:%(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Get a formatted logger."""
    logger = logging.getLogger(name)

    handler = logging.StreamHandler()
    handler.setFormatter(_formatter)
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("PERIFLOW_LOG_LEVEL", "INFO"))

    return logger


logger = get_logger("PeriFlow")
