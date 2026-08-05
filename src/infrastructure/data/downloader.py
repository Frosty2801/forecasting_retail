import hashlib
from datetime import datetime, timezone
from pathlib import Path

import kagglehub
import pandas as pd

from src.domain.dataset import DatasetManifest
from src.domain.interfaces import DatasetRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KaggleDatasetRepository(DatasetRepository):
    """Concrete repository using kagglehub with local filesystem fallback."""

    def __init__(
        self,
        competition_name: str = "store-sales-time-series-forecasting",
        raw_data_dir: Path = Path("data/raw"),
        expected_files: list[str] | None = None,
    ):
        self.competition_name = competition_name
        self.raw_data_dir = Path(raw_data_dir)
        self.expected_files = expected_files or [
            "train.csv",
            "stores.csv",
            "holidays_events.csv",
            "oil.csv",
            "transactions.csv",
        ]

    def download_dataset(self) -> Path:
        """Downloads the competition dataset via kagglehub, falling back to local raw dir if offline/error."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = None

        try:
            logger.info(f"Attempting to download competition dataset: {self.competition_name}")
            path_str = kagglehub.competition_download(self.competition_name)
            downloaded_path = Path(path_str)
            logger.info(f"Successfully downloaded via kagglehub to {downloaded_path}")

            # Copy files into raw_data_dir if not already there
            for file_path in downloaded_path.glob("*.csv"):
                dest_path = self.raw_data_dir / file_path.name
                if not dest_path.exists():
                    dest_path.write_bytes(file_path.read_bytes())
                    logger.debug(f"Copied {file_path.name} to {self.raw_data_dir}")

        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Kagglehub download failed ({e}). Falling back to local directory: {self.raw_data_dir}")

        # Verify expected files exist locally
        missing = [f for f in self.expected_files if not (self.raw_data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing expected dataset CSV files in {self.raw_data_dir}: {missing}. "
                "Please ensure Kaggle credentials are configured or files are placed manually."
            )

        return self.raw_data_dir

    def load_csv(self, filename: str) -> pd.DataFrame:
        file_path = self.raw_data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file {filename} not found at {file_path}")
        logger.debug(f"Loading CSV file: {file_path}")
        return pd.read_csv(file_path)

    def generate_manifest(self) -> DatasetManifest:
        """Computes SHA-256 checksums and record counts for versioning."""
        checksums: dict[str, str] = {}
        counts: dict[str, int] = {}
        hasher_all = hashlib.sha256()

        for filename in self.expected_files:
            file_path = self.raw_data_dir / filename
            if file_path.exists():
                file_bytes = file_path.read_bytes()
                file_hash = hashlib.sha256(file_bytes).hexdigest()
                checksums[filename] = file_hash
                hasher_all.update(file_bytes)

                # Get row count quickly
                df = pd.read_csv(file_path, usecols=[0])
                counts[filename] = len(df)

        version_hash = hasher_all.hexdigest()[:16]
        manifest = DatasetManifest(
            version_hash=version_hash,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            source="kagglehub/local",
            file_checksums=checksums,
            record_counts=counts,
        )
        return manifest
