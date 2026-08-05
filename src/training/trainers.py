import time
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModelTrainer(ABC):
    """Abstract base trainer adhering to Open/Closed principle."""

    def __init__(self, model_name: str, hyperparameters: dict[str, Any] | None = None):
        self.model_name = model_name
        self.hyperparameters = hyperparameters or {}
        self.model = self._build_model()

    @abstractmethod
    def _build_model(self) -> Any:
        pass

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        pass


class LightGBMTrainer(BaseModelTrainer):
    """LightGBM model trainer."""

    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        super().__init__(model_name="LightGBM", hyperparameters=hyperparameters)

    def _build_model(self) -> Any:
        import lightgbm as lgb

        params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42, **self.hyperparameters}
        return lgb.LGBMRegressor(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        logger.info(f"Training {self.model_name} with {X_train.shape[0]} samples...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        duration = time.time() - start_time
        logger.info(f"Finished training {self.model_name} in {duration:.2f} seconds.")
        return self.model


class XGBoostTrainer(BaseModelTrainer):
    """XGBoost model trainer."""

    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        super().__init__(model_name="XGBoost", hyperparameters=hyperparameters)

    def _build_model(self) -> Any:
        import xgboost as xgb

        params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42, **self.hyperparameters}
        return xgb.XGBRegressor(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        logger.info(f"Training {self.model_name} with {X_train.shape[0]} samples...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        duration = time.time() - start_time
        logger.info(f"Finished training {self.model_name} in {duration:.2f} seconds.")
        return self.model


class RandomForestTrainer(BaseModelTrainer):
    """Random Forest model trainer."""

    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        super().__init__(model_name="RandomForest", hyperparameters=hyperparameters)

    def _build_model(self) -> Any:
        params = {"n_estimators": 50, "max_depth": 15, "random_state": 42, "n_jobs": -1, **self.hyperparameters}
        return RandomForestRegressor(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        logger.info(f"Training {self.model_name} with {X_train.shape[0]} samples...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        duration = time.time() - start_time
        logger.info(f"Finished training {self.model_name} in {duration:.2f} seconds.")
        return self.model
