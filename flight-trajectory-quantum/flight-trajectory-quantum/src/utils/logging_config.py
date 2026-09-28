from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt="%(levelname)-5s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    # Qiskit's transpiler pass manager logs very verbose per-pass timing at
    # INFO level, which drowns out this project's own structured logs.
    # Silence it explicitly rather than lowering our own log level.
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
