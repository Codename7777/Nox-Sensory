"""Platform-specific storage locations with no implicit POSIX fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cove_sensory_mcp.errors import ErrorCode, SensoryError


@dataclass(frozen=True, slots=True)
class AppPaths:
    """The local configuration and transient-job locations for one system."""

    config_file: Path
    jobs_dir: Path

    @classmethod
    def for_system(
        cls,
        system: str,
        home: Path,
        roaming: Path | None,
        local: Path | None,
    ) -> AppPaths:
        """Return the public storage contract for a supported operating system."""
        if system == "Darwin":
            return cls(
                config_file=home / "Library/Application Support/cove-sensory-mcp/config.yaml",
                jobs_dir=home / "Library/Caches/cove-sensory-mcp/jobs",
            )
        if system == "Windows":
            if roaming is None or local is None:
                raise SensoryError(
                    ErrorCode.CONFIG_INVALID,
                    "Windows application directories are unavailable.",
                )
            return cls(
                config_file=roaming / "CoveSensoryMCP/config.yaml",
                jobs_dir=local / "CoveSensoryMCP/Cache/jobs",
            )
        if system == "Linux":
            # Nox fork: the heart lives on a Linux VPS. Plain XDG locations, nothing implicit.
            config_root = _xdg("XDG_CONFIG_HOME", home / ".config")
            cache_root = _xdg("XDG_CACHE_HOME", home / ".cache")
            return cls(
                config_file=config_root / "cove-sensory-mcp/config.yaml",
                jobs_dir=cache_root / "cove-sensory-mcp/jobs",
            )
        raise SensoryError(ErrorCode.CONFIG_INVALID, "Unsupported operating system.")


def _xdg(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value and Path(value).is_absolute() else default
