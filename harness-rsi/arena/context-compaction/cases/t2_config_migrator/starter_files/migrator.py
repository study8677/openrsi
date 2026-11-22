"""Config migrator module — implement according to the agreed conversation specs."""

from __future__ import annotations
from typing import Any


def migrate_config(raw: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError
