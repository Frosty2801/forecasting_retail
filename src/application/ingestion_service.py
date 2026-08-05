import json
from pathlib import Path

import pandas as pd

from src.application.cleaning.cleaner import DataCleaner
from src.application.cleaning.merger import DatasetMerger
from src.application.validation.validator import DataValidator
from src.domain.interfaces import DatasetRepository
from src.domain.validation import ValidationReport
from src.infrastructure.data.downloader import KaggleDatasetRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IngestDatasetUseCase:
    """Orchestrates the complete data download, validation, cleaning, and merging pipeline."""

    def __init__(
        self,
        repository: DatasetRepository | None = None,
        processed_dir: Path = Path("data/processed"),
    ):
        self.repository = repository or KaggleDatasetRepository()
        self.processed_dir = Path(processed_dir)

    def execute(self) -> tuple[ValidationReport, pd.DataFrame]:
        logger.info("Starting Dataset Ingestion Use Case...")

        # 1. Download / locate raw data
        raw_path = self.repository.download_dataset()

        # 2. Load raw tables
        train = self.repository.load_csv("train.csv")
        stores = self.repository.load_csv("stores.csv")
        holidays = self.repository.load_csv("holidays_events.csv")
        oil = self.repository.load_csv("oil.csv")
        transactions = None
        if (raw_path / "transactions.csv").exists():
            transactions = self.repository.load_csv("transactions.csv")

        # 3. Validate raw data
        report = DataValidator.validate_dataset(train, stores, holidays, oil, transactions)

        # 4. Clean data
        clean_train_df = DataCleaner.clean_train(train)
        clean_stores_df = DataCleaner.clean_stores(stores)
        clean_holidays_df = DataCleaner.clean_holidays(holidays)
        clean_oil_df = DataCleaner.clean_oil(oil)
        clean_trans_df = DataCleaner.clean_transactions(transactions) if transactions is not None else None

        # 5. Merge tables
        merged_df = DatasetMerger.merge(
            clean_train_df,
            clean_stores_df,
            clean_holidays_df,
            clean_oil_df,
            clean_trans_df,
        )

        # 6. Persist to Parquet
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / "master.parquet"
        merged_df.to_parquet(output_path, index=False)
        logger.info(f"Master dataset successfully persisted to {output_path}")

        # 7. Save manifest if repository supports it
        if hasattr(self.repository, "generate_manifest"):
            manifest = self.repository.generate_manifest()
            manifest_path = self.processed_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest.__dict__, f, indent=2)
            logger.info(f"Dataset manifest saved to {manifest_path}")

        return report, merged_df
