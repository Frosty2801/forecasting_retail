from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application Settings loaded from environment variables or .env file."""

    app_env: str = Field(default="development", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    data_dir: Path = Field(default=Path("data"), validation_alias="DATA_DIR")
    models_dir: Path = Field(default=Path("models"), validation_alias="MODELS_DIR")
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db", validation_alias="MLFLOW_TRACKING_URI"
    )
    kaggle_dataset_handle: str = Field(
        default="c/store-sales-given-data", validation_alias="KAGGLE_DATASET_HANDLE"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """Loads and parses a YAML configuration file.

    Args:
        config_path: Optional path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Global settings instance
settings = AppSettings()
