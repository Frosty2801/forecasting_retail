from src.config.settings import AppSettings, load_yaml_config
from src.utils.logger import get_logger


def test_app_settings_defaults():
    settings = AppSettings()
    assert settings.app_env in ["development", "production", "test"]
    assert str(settings.data_dir) == "data"


def test_load_yaml_config():
    yaml_config = load_yaml_config()
    assert "app" in yaml_config
    assert yaml_config["app"]["name"] == "Retail Sales Forecasting"
    assert "features" in yaml_config


def test_get_logger():
    logger = get_logger("unit_test_logger", level="DEBUG")
    assert logger.name == "unit_test_logger"
    assert logger.level == 10  # DEBUG level
