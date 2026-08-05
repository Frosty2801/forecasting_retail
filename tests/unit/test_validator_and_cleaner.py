import pandas as pd

from src.application.cleaning.cleaner import DataCleaner
from src.application.cleaning.merger import DatasetMerger
from src.application.validation.validator import DataValidator


def test_validator_detects_negatives_and_duplicates():
    train_data = pd.DataFrame(
        {
            "id": [1, 2, 2],
            "date": ["2017-01-01", "2017-01-02", "2017-01-02"],
            "store_nbr": [1, 999, 2],  # 999 is invalid store
            "family": ["AUTOMOTIVE", "BABYCARE", "BEAUTY"],
            "sales": [10.5, -5.0, 20.0],  # -5.0 is negative sales
            "onpromotion": [0, 1, 0],
        }
    )
    # Make sure duplicate row check matches across all columns
    train_data.loc[2] = train_data.loc[1]
    report = DataValidator.validate_train(train_data)
    assert not report.is_valid
    issue_types = [issue.issue_type for issue in report.issues]
    assert "duplicates" in issue_types
    assert "negative_sales" in issue_types
    assert "invalid_store_id" in issue_types


def test_cleaner_and_merger():
    train = pd.DataFrame(
        {
            "id": [1, 2],
            "date": ["2017-01-01", "2017-01-02"],
            "store_nbr": [1, 1],
            "family": ["AUTOMOTIVE", "AUTOMOTIVE"],
            "sales": [10.0, 20.0],
            "onpromotion": [0, 1],
        }
    )
    stores = pd.DataFrame(
        {
            "store_nbr": [1],
            "city": ["Quito"],
            "state": ["Pichincha"],
            "type": ["A"],
            "cluster": [13],
        }
    )
    holidays = pd.DataFrame(
        {
            "date": ["2017-01-01"],
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["New Year"],
            "transferred": [False],
        }
    )
    oil = pd.DataFrame({"date": ["2017-01-01", "2017-01-02"], "dcoilwtico": [40.0, 41.0]})

    cleaned_train = DataCleaner.clean_train(train)
    cleaned_oil = DataCleaner.clean_oil(oil)
    cleaned_holidays = DataCleaner.clean_holidays(holidays)
    cleaned_stores = DataCleaner.clean_stores(stores)

    merged = DatasetMerger.merge(cleaned_train, cleaned_stores, cleaned_holidays, cleaned_oil)
    assert not merged.empty
    assert "dcoilwtico" in merged.columns
    assert "city" in merged.columns
    assert "is_holiday" in merged.columns
    assert merged.loc[merged["date"] == "2017-01-01", "is_holiday"].iloc[0] == 1
