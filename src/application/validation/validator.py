import pandas as pd

from src.domain.validation import ValidationIssue, ValidationReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validates raw dataset tables against quality constraints."""

    @staticmethod
    def validate_train(df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()
        table = "train.csv"

        # 1. Missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            report.add_issue(
                ValidationIssue(
                    table_name=table,
                    issue_type="missing_values",
                    severity="WARNING",
                    description=f"Found {missing_count} missing values in train set",
                    affected_count=int(missing_count),
                )
            )

        # 2. Duplicated rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            report.add_issue(
                ValidationIssue(
                    table_name=table,
                    issue_type="duplicates",
                    severity="ERROR",
                    description=f"Found {dup_count} duplicate rows in train set",
                    affected_count=int(dup_count),
                )
            )

        # 3. Negative sales
        if "sales" in df.columns:
            neg_sales = (df["sales"] < 0).sum()
            if neg_sales > 0:
                report.add_issue(
                    ValidationIssue(
                        table_name=table,
                        issue_type="negative_sales",
                        severity="ERROR",
                        description=f"Found {neg_sales} rows with negative sales",
                        affected_count=int(neg_sales),
                    )
                )

        # 4. Invalid store IDs (Store Sales dataset expects stores 1 to 54)
        if "store_nbr" in df.columns:
            invalid_stores = ~df["store_nbr"].isin(range(1, 55))
            invalid_count = invalid_stores.sum()
            if invalid_count > 0:
                report.add_issue(
                    ValidationIssue(
                        table_name=table,
                        issue_type="invalid_store_id",
                        severity="ERROR",
                        description=f"Found {invalid_count} rows with invalid store numbers",
                        affected_count=int(invalid_count),
                    )
                )

        # 5. Invalid dates
        if "date" in df.columns:
            try:
                pd.to_datetime(df["date"], errors="raise")
            except (ValueError, TypeError):
                report.add_issue(
                    ValidationIssue(
                        table_name=table,
                        issue_type="invalid_dates",
                        severity="ERROR",
                        description="Found malformed date strings in train set",
                        affected_count=1,
                    )
                )

        return report

    @staticmethod
    def validate_dataset(
        train: pd.DataFrame,
        stores: pd.DataFrame,
        holidays: pd.DataFrame,
        oil: pd.DataFrame,
        transactions: pd.DataFrame = None,
    ) -> ValidationReport:
        """Runs validation across all primary tables."""
        master_report = ValidationReport()
        train_report = DataValidator.validate_train(train)
        master_report.issues.extend(train_report.issues)
        if not train_report.is_valid:
            master_report.is_valid = False

        # Basic stores check
        if stores is not None and stores.isnull().sum().sum() > 0:
            master_report.add_issue(
                ValidationIssue(
                    table_name="stores.csv",
                    issue_type="missing_values",
                    severity="WARNING",
                    description="Missing values found in stores table",
                    affected_count=int(stores.isnull().sum().sum()),
                )
            )

        logger.info(f"Data validation completed. Is Valid: {master_report.is_valid}, Issues: {len(master_report.issues)}")
        return master_report
