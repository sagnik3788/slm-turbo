"""Adapter base classes (AGENTS.md: adapters/base.py).

Every adapter exposes a uniform lifecycle used by `slm-turbo serve`:
start -> generate -> metrics -> stop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAdapter(ABC):
    """Abstract base class for all serving adapters."""

    @abstractmethod
    def start(self) -> "BaseAdapter":
        """Boot the serving stack. Returns self for chaining."""

    @abstractmethod
    def stop(self) -> None:
        """Shut down the serving stack and free resources."""

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Return a flat dict of live serving metrics.

        Keys must be strings; values should be JSON-serializable scalars.
        """

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> List[str]:
        """Run one or more prompts through the model. Returns decoded text."""
