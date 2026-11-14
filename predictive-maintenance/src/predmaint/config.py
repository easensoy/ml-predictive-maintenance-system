"""Runtime settings, read once from ``PM_*`` environment variables.

Every setting has a safe default (loopback host, debug off, same-origin Socket.IO). A malformed
value fails loudly at start-up instead of being silently replaced.
"""

import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
# The documents whose generated blocks predmaint.report writes and predmaint.audit checks.
DOCUMENTS = (
    PROJECT_ROOT.parent / "README.md",
    PROJECT_ROOT.parent / "MODEL_CARD.md",
    PROJECT_ROOT / "docs" / "evaluation.md",
    PROJECT_ROOT / "docs" / "serving.md",
)


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"not a boolean: {raw!r}")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = PROJECT_ROOT / "data" / "cmapss"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    subset: str = "FD001"
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    replay_interval_s: float = 1.0
    replay_engines: int = 12
    replay_autostart: bool = True
    # Integrated gradients dominates a replay tick (docs/benchmark.json), so each tick refreshes
    # the attribution of at most this many watched engines, round robin.
    explain_per_tick: int = 2
    cors_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.subset not in SUBSETS:
            raise ValueError(f"PM_SUBSET must be one of {SUBSETS}, got {self.subset!r}")
        if not 0 < self.port < 65536:
            raise ValueError(f"PM_PORT out of range: {self.port}")
        # NaN compares False with everything, so `<= 0` alone would let nan and inf through.
        if not (math.isfinite(self.replay_interval_s) and self.replay_interval_s > 0):
            raise ValueError(
                "PM_REPLAY_INTERVAL_S must be a finite positive number, "
                f"got {self.replay_interval_s}"
            )
        if self.replay_engines < 1:
            raise ValueError("PM_REPLAY_ENGINES must be at least 1")
        if self.explain_per_tick < 1:
            raise ValueError("PM_EXPLAIN_PER_TICK must be at least 1")

    @property
    def model_dir(self) -> Path:
        return self.artifacts_dir / self.subset

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for field in fields(cls):
            raw = env.get(f"PM_{field.name.upper()}")
            if raw is None:
                continue
            try:
                if field.type is Path:
                    values[field.name] = Path(raw)
                elif field.type is bool:
                    values[field.name] = _parse_bool(raw)
                elif field.type is int:
                    values[field.name] = int(raw)
                elif field.type is float:
                    values[field.name] = float(raw)
                elif field.name == "cors_origins":
                    values[field.name] = tuple(o.strip() for o in raw.split(",") if o.strip())
                else:
                    values[field.name] = raw
            except ValueError as exc:
                raise ValueError(f"Invalid PM_{field.name.upper()}: {exc}") from exc
        return cls(**values)
