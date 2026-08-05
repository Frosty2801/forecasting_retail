import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Cleans and normalizes raw dataset tables."""

    @staticmethod
    def clean_train(df: pd.DataFrame, missing_strategy: str = "drop") -> pd.DataFrame:
        df = df.copy()
        # Normalize date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Handle duplicates
        df = df.drop_duplicates()

        # Handle missing sales/onpromotion
        if missing_strategy == "drop":
            df = df.dropna(subset=["sales"])
        elif missing_strategy == "fill_zero":
            df["sales"] = df["sales"].fillna(0.0)

        if "onpromotion" in df.columns:
            df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)

        # Ensure correct data types
        if "store_nbr" in df.columns:
            df["store_nbr"] = df["store_nbr"].astype(int)
        if "sales" in df.columns:
            df["sales"] = df["sales"].astype(float)
        if "id" in df.columns:
            df["id"] = df["id"].astype(int)

        return df

    @staticmethod
    def clean_oil(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        # Oil prices have missing values over weekends/holidays -> forward fill
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
        return df

    @staticmethod
    def clean_holidays(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def clean_stores(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    @staticmethod
    def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
