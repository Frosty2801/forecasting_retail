import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost

from src.evaluation.evaluator import ModelEvaluator
from src.training.trainers import BaseModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Manages model selection, artifact persistence, and MLflow tracking."""

    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def track_and_train(
        self,
        trainer: BaseModelTrainer,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        experiment_name: str = "retail_sales_forecasting",
        preprocessor: Any | None = None,
    ) -> tuple[Any, dict[str, float]]:
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=trainer.model_name):
            # Log hyperparameters
            mlflow.log_params(trainer.hyperparameters)
            mlflow.log_param("model_type", trainer.model_name)

            # Train
            model = trainer.train(X_train, y_train)

            # Predict & Evaluate
            preds = model.predict(X_val)
            metrics = ModelEvaluator.evaluate(y_val, preds)

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model artifact in MLflow using the native flavor when available
            self._log_model_artifact(model)

            # Save locally
            model_path = self.models_dir / f"{trainer.model_name.lower()}_model.pkl"
            joblib.dump(model, model_path)

            metrics_path = self.models_dir / f"{trainer.model_name.lower()}_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            if preprocessor is not None:
                preprocessor_path = self.models_dir / "preprocessor.pkl"
                preprocessor.save(preprocessor_path)
                mlflow.log_artifact(str(preprocessor_path))

            logger.info(f"Model {trainer.model_name} successfully tracked and persisted.")
            return model, metrics

    @staticmethod
    def _log_model_artifact(model: Any) -> None:
        """Logs a model to MLflow using the framework-native flavor when supported."""
        if isinstance(model, lightgbm.LGBMRegressor):
            mlflow.lightgbm.log_model(model, "model")
        elif isinstance(model, xgboost.XGBRegressor):
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
