import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CalendarFeatureTransformer:
    """Extracts calendar and temporal features from datetime index/column."""

    @staticmethod
    def transform(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        df = df.copy()
        if date_col not in df.columns:
            return df

        dt = pd.to_datetime(df[date_col])
        df["year"] = dt.dt.year.astype(int)
        df["month"] = dt.dt.month.astype(int)
        df["week"] = dt.dt.isocalendar().week.astype(int)
        df["day"] = dt.dt.day.astype(int)
        df["dayofweek"] = dt.dt.dayofweek.astype(int)
        df["quarter"] = dt.dt.quarter.astype(int)
        df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

        logger.debug("Calendar features extracted successfully.")
        return df
