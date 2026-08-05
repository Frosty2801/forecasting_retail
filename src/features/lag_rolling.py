import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LagAndRollingFeatureTransformer:
    """Generates time-series lag features and rolling window averages grouped by store and product family."""

    def __init__(
        self,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        target_col: str = "sales",
        group_cols: list[str] | None = None,
    ):
        self.lags = lags or [1, 7, 14, 28]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.target_col = target_col
        self.group_cols = group_cols or ["store_nbr", "family"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.target_col not in df.columns:
            return df

        # Ensure sorted by date within groups for correct lag/rolling calculations
        if "date" in df.columns:
            df = df.sort_values(by=["store_nbr", "family", "date"])

        grouped = df.groupby(self.group_cols, observed=False)[self.target_col]

        for lag in self.lags:
            col_name = f"{self.target_col}_lag_{lag}"
            df[col_name] = grouped.shift(lag)

        for window in self.rolling_windows:
            col_name = f"{self.target_col}_rolling_mean_{window}"
            # Shift by 1 before rolling to avoid data leakage
            df[col_name] = grouped.shift(1).rolling(window=window, min_periods=1).mean()

        logger.debug(f"Generated {len(self.lags)} lag features and {len(self.rolling_windows)} rolling windows.")
        return df
