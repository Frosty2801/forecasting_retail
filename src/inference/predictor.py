from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.training.preprocessor import FeaturePreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Loads a persisted model and preprocessor, then generates sales forecasts."""

    def __init__(
        self,
        model_path: Path | str = Path("models/lightgbm_model.pkl"),
        preprocessor_path: Path | str = Path("models/preprocessor.pkl"),
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.model = self._load_model()
        self.preprocessor = self._load_preprocessor()

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}. Inference engine initialized without active model.")
            return None
        return joblib.load(self.model_path)

    def _load_preprocessor(self) -> FeaturePreprocessor | None:
        if not self.preprocessor_path.exists():
            logger.warning(
                f"Preprocessor not found at {self.preprocessor_path}. Raw feature input will be used."
            )
            return None
        return FeaturePreprocessor.load(self.preprocessor_path)

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("No trained model loaded for inference.")

        if self.preprocessor is not None:
            features_df = self.preprocessor.transform(features_df)

        logger.info(f"Generating predictions for {features_df.shape[0]} records...")
        preds = self.model.predict(features_df)
        return pd.Series(preds, name="predicted_sales")
