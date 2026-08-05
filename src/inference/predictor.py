from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Loads a persisted model and generates sales forecasts."""

    def __init__(self, model_path: Path | str = Path("models/lightgbm_model.pkl")):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}. Inference engine initialized without active model.")
            return None
        return joblib.load(self.model_path)

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("No trained model loaded for inference.")
        logger.info(f"Generating predictions for {features_df.shape[0]} records...")
        preds = self.model.predict(features_df)
        return pd.Series(preds, name="predicted_sales")
