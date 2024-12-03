from pathlib import Path

from typing import Dict, Any
from typing_extensions import Self
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parent.parent


class EnviromentVariables(BaseSettings):
    """
    Defines an `EnviromentVariables` class that loads and validates
    environment variables that are used in the project.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8"
    )

    host: str
    app_docker_port: int
    app_vm_port: int
    redis_vm_port: int

    log_dir: Path
    config_dir: Path
    events_store_dir: Path
    offline_recs_dir: Path
    online_recs_dir: Path
    popular_recs_dir: Path

    @model_validator(mode="after")
    def validate_after(self) -> Self:
        """
        Ensures that any directory paths defined as environment
        variables are created if they do not already exist.

        Returns:
            Self:
                The updated `EnviromentVariables` instance with the
                created directories.
        """
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                if not v.exists():
                    v.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_before(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the input data for the `EnviromentVariables` class,
        ensuring that only the fields defined in the class are used.
        Also converts any directory from relative to absolute paths.

        Args:
            data (Dict[str, Any]):
                The input data to be validated.

        Returns:
            Dict[str, Any]:
                The validated data
        """
        out_data = {}
        for k, v in data.items():
            # Remove a field that are not defined in the class
            if k in cls.__fields__:
                out_data[k] = v
                # Adjust path if the field is a directory path
                if k in [
                    "log_dir",
                    "config_dir",
                    "events_store_dir",
                    "offline_recs_dir",
                    "online_recs_dir",
                    "popular_recs_dir",
                ]:
                    out_data[k] = PROJECT_DIR / v
        return out_data


env_vars = EnviromentVariables(
    _env_file=Path(PROJECT_DIR.parent, ".env"), _env_file_encoding="utf-8"
)
