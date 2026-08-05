import pandas as pd

from src.features.calendar import CalendarFeatureTransformer
from src.features.lag_rolling import LagAndRollingFeatureTransformer
from src.features.pipeline import FeatureEngineeringPipeline


def test_calendar_transformer():
    df = pd.DataFrame({"date": ["2017-01-01", "2017-01-07"]})
    transformed = CalendarFeatureTransformer.transform(df)
    assert "year" in transformed.columns
    assert "month" in transformed.columns
    assert "is_weekend" in transformed.columns
    assert transformed.loc[0, "is_weekend"] == 1  # Sunday
    assert transformed.loc[1, "is_weekend"] == 1  # Saturday


def test_lag_and_rolling_transformer():
    df = pd.DataFrame(
        {
            "store_nbr": [1, 1, 1, 1],
            "family": ["A", "A", "A", "A"],
            "date": ["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"],
            "sales": [10.0, 20.0, 30.0, 40.0],
        }
    )
    transformer = LagAndRollingFeatureTransformer(lags=[1], rolling_windows=[2])
    res = transformer.transform(df)
    assert "sales_lag_1" in res.columns
    assert "sales_rolling_mean_2" in res.columns
    assert res.loc[1, "sales_lag_1"] == 10.0


def test_feature_engineering_pipeline():
    df = pd.DataFrame(
        {
            "store_nbr": [1, 1, 1],
            "family": ["A", "A", "A"],
            "date": ["2017-01-01", "2017-01-02", "2017-01-03"],
            "sales": [10.0, 20.0, 30.0],
        }
    )
    pipeline = FeatureEngineeringPipeline()
    res = pipeline.fit_transform(df)
    assert "year" in res.columns
    assert "sales_lag_1" in res.columns
