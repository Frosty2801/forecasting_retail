import pandas as pd

from src.evaluation.evaluator import ModelEvaluator
from src.training.preprocessor import FeaturePreprocessor
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


def test_preprocessor_encodes_categorical_and_drops_identifiers():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2013-01-01", "2013-01-02", "2013-01-03"],
            "store_nbr": [1, 1, 2],
            "family": ["A", "B", "A"],
            "city": ["Quito", "Quito", "Cuenca"],
            "onpromotion": [0, 1, 0],
            "sales": [10.0, 20.0, 30.0],
        }
    )

    pre = FeaturePreprocessor()
    X, y = pre.fit_transform(df)

    assert "id" not in X.columns
    assert "date" not in X.columns
    assert "sales" not in X.columns
    assert "family" in X.columns
    assert "city" in X.columns
    assert y.tolist() == [10.0, 20.0, 30.0]


def test_preprocessor_transform_matches_fit():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2013-01-01", "2013-01-02", "2013-01-03"],
            "store_nbr": [1, 1, 2],
            "family": ["A", "B", "A"],
            "city": ["Quito", "Quito", "Cuenca"],
            "onpromotion": [0, 1, 0],
            "sales": [10.0, 20.0, 30.0],
        }
    )

    pre = FeaturePreprocessor()
    X, _ = pre.fit_transform(df)

    new_row = pd.DataFrame(
        {
            "id": [99],
            "date": ["2013-01-04"],
            "store_nbr": [1],
            "family": ["B"],
            "city": ["Quito"],
            "onpromotion": [1],
            "sales": [15.0],
        }
    )
    X_new = pre.transform(new_row)
    assert list(X_new.columns) == list(X.columns)
