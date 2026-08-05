from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DatasetManifest:
    """Metadata versioning record for downloaded and processed datasets."""

    version_hash: str
    downloaded_at: str
    source: str
    file_checksums: dict[str, str] = field(default_factory=dict)
    record_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class RawDataset:
    """Container for raw Kaggle dataset tables."""

    train: pd.DataFrame
    stores: pd.DataFrame
    holidays: pd.DataFrame
    oil: pd.DataFrame
    transactions: pd.DataFrame | None = None
    manifest: DatasetManifest | None = None
