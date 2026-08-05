from pathlib import Path
from typing import ClassVar

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePreprocessor:
    """Prepares engineered features for tree-based model training.

    Drops non-predictive columns (identifiers, dates) and encodes categorical
    columns into numeric labels so downstream estimators receive numeric input.
    """

    IDENTIFIER_COLUMNS: ClassVar[set[str]] = {"id", "date"}
    TARGET_COLUMNS: ClassVar[set[str]] = {"sales", "id"}

    def __init__(self, target_col: str = "sales"):
        self.target_col = target_col
        self._encoders: dict[str, LabelEncoder] = {}
        self._feature_columns: list[str] = []
        self._categorical_columns: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

        y = df[self.target_col].astype(float)

        feature_df = df.drop(columns=[self.target_col, *self.IDENTIFIER_COLUMNS], errors="ignore")

        self._categorical_columns = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in self._categorical_columns:
            encoder = LabelEncoder()
            feature_df[col] = encoder.fit_transform(feature_df[col].astype(str))
            self._encoders[col] = encoder

        numeric_df = feature_df.select_dtypes(include=["int", "float", "bool"])
        self._feature_columns = numeric_df.columns.tolist()

        logger.info(
            f"Preprocessed dataset: {len(self._feature_columns)} numeric features, "
            f"{len(self._categorical_columns)} categorical columns encoded."
        )
        return numeric_df, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        feature_df = df.drop(columns=[self.target_col, *self.IDENTIFIER_COLUMNS], errors="ignore")

        for col, encoder in self._encoders.items():
            if col in feature_df.columns:
                feature_df[col] = encoder.transform(feature_df[col].astype(str))

        return feature_df[self._feature_columns]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "FeaturePreprocessor":
        return joblib.load(path)
