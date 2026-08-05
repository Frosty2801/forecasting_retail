import pandas as pd

from src.evaluation.evaluator import ModelEvaluator
from src.training.trainers import RandomForestTrainer


def test_evaluator():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    metrics = ModelEvaluator.evaluate(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert "r2" in metrics
    assert metrics["mae"] > 0


def test_random_forest_trainer():
    X_train = pd.DataFrame({"feat1": [1, 2, 3, 4, 5], "feat2": [5, 4, 3, 2, 1]})
    y_train = pd.Series([10, 20, 30, 40, 50])

    trainer = RandomForestTrainer(hyperparameters={"n_estimators": 5, "max_depth": 2})
    model = trainer.train(X_train, y_train)
    assert model is not None

    preds = model.predict(X_train)
    assert len(preds) == 5
