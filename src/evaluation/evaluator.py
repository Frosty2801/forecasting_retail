import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Computes standard time-series and regression performance metrics."""

    @staticmethod
    def evaluate(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        # Avoid division by zero in MAPE
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100)
        else:
            mape = 0.0

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
        }
        logger.info(f"Evaluation metrics computed: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, R2={r2:.4f}")
        return metrics
