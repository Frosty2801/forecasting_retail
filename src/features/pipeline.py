import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineeringPipeline:
    """Orchestrates modular feature transformation steps."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        features_cfg = self.config.get("features", {})
        self.lags = features_cfg.get("lags", [1, 7, 14, 28])
        self.rolling_windows = features_cfg.get("rolling_windows", [7, 14, 30])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Running feature engineering pipeline...")
        from src.features.calendar import CalendarFeatureTransformer
        from src.features.lag_rolling import LagAndRollingFeatureTransformer

        # 1. Calendar features
        df = CalendarFeatureTransformer.transform(df)

        # 2. Lag and rolling statistics
        transformer = LagAndRollingFeatureTransformer(
            lags=self.lags, rolling_windows=self.rolling_windows
        )
        df = transformer.transform(df)

        logger.info(f"Feature engineering completed. Final feature count: {df.shape[1]}")
        return df
