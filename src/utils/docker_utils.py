"""Docker-related helpers for local development."""

import subprocess
from pathlib import Path

from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)


def is_docker_available() -> bool:
    """Check whether Docker CLI is available."""
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def compose_up(detached: bool = True) -> int:
    """Run docker compose up from project root."""
    cmd = ["docker", "compose", "up"]
    if detached:
        cmd.append("-d")
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def compose_down() -> int:
    """Run docker compose down from project root."""
    result = subprocess.run(["docker", "compose", "down"], cwd=PROJECT_ROOT)
    return result.returncode
