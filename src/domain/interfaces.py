from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DatasetRepository(ABC):
    """Abstract repository protocol for downloading and loading dataset files."""

    @abstractmethod
    def download_dataset(self) -> Path:
        """Downloads the dataset (or locates it locally) and returns the raw data directory path."""

    @abstractmethod
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Loads a specific CSV file into a Pandas DataFrame."""
