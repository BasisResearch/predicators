"""Interaction budget: one unit per ``env.step`` call, whoever issued it."""
from __future__ import annotations


class BudgetExhausted(RuntimeError):
    """Raised by ``SimSession.step`` once the interaction cap is reached."""


class InteractionBudget:
    """Counts env interactions against a hard cap."""

    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.used = 0

    @property
    def remaining(self) -> int:
        """Interactions left before the cap."""
        return max(0, self.cap - self.used)

    @property
    def exhausted(self) -> bool:
        """True once no interactions remain."""
        return self.used >= self.cap

    def consume(self, n: int = 1) -> None:
        """Record ``n`` interactions; raise if that crosses the cap."""
        if self.exhausted:
            raise BudgetExhausted(
                f"Interaction budget exhausted ({self.used}/{self.cap}).")
        self.used += int(n)

    def status(self) -> str:
        """Short text for tool results."""
        return f"interactions used {self.used} / {self.cap} (remaining {self.remaining})"
