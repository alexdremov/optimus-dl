import os
from dataclasses import dataclass
from typing import Any


@dataclass
class EnvironmentVariable:
    """Manages a single environment variable with type and default value."""

    name: str
    var_type: type
    default: Any | None = None

    def get(self) -> Any:
        """Get the environment variable value or return default."""
        value = os.getenv(self.name)
        if value is None:
            return self.default
        if self.var_type is bool:
            return value.lower() in ("1", "true", "yes", "on", "y")
        return self.var_type(value)

    def set(self, value: Any) -> None:
        """Set the environment variable."""
        os.environ[self.name] = str(value)


# Activate additional logging for debugging purposes if OPTIMUS_DEBUG is set to True
OPTIMUS_DEBUG = EnvironmentVariable("OPTIMUS_DEBUG", bool, False)

# Timeout in seconds to wait for child processes to terminate before force killing the process
OPTIMUS_EXIT_TIMEOUT = EnvironmentVariable("OPTIMUS_EXIT_TIMEOUT", int, 30)
