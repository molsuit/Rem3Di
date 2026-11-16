"""The one logging setup of the workspace's entry points (scripts, preparer CLIs)."""

from __future__ import annotations

import logging

#: Time, level, logger name, message: one line format for every entry point.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Log to stderr at ``level`` (a number or a name such as ``"DEBUG"``).

    Like :func:`logging.basicConfig`, it does nothing if the root logger
    already has handlers, so a caller (or pytest) that configured logging keeps
    its setup.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT)
