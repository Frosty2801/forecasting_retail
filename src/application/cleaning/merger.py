import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetMerger:
    """Merges cleaned relational tables into a single unified DataFrame."""

    @staticmethod
    def merge(
        train: pd.DataFrame,
        stores: pd.DataFrame,
        holidays: pd.DataFrame,
        oil: pd.DataFrame,
        transactions: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Merges train sales data with stores metadata, oil prices, and holidays."""
        logger.info("Merging dataset tables...")

        # 1. Merge train with stores on store_nbr
        merged = pd.merge(train, stores, on="store_nbr", how="left")

        # 2. Merge with oil prices on date
        merged = pd.merge(merged, oil, on="date", how="left")
        # Fill remaining missing oil prices if any
        if "dcoilwtico" in merged.columns:
            merged["dcoilwtico"] = merged["dcoilwtico"].ffill().bfill()

        # 3. Aggregate / join holidays (take the first holiday or indicator per date)
        if holidays is not None and not holidays.empty:
            # Filter out transferred/workday holidays for simplicity or aggregate
            holidays_agg = (
                holidays.groupby("date")
                .agg({"type": "first", "locale": "first", "transferred": "first"})
                .reset_index()
                .rename(
                    columns={
                        "type": "holiday_type",
                        "locale": "holiday_locale",
                        "transferred": "holiday_transferred",
                    }
                )
            )
            merged = pd.merge(merged, holidays_agg, on="date", how="left")
            merged["is_holiday"] = merged["holiday_type"].notnull().astype(int)
        else:
            merged["is_holiday"] = 0

        # 4. Merge transactions if available
        if transactions is not None and not transactions.empty:
            merged = pd.merge(merged, transactions, on=["date", "store_nbr"], how="left")

        logger.info(f"Merge completed successfully. Resulting shape: {merged.shape}")
        return merged
