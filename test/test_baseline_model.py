from typing import Any


import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_model import add_dummy_baseline, preprocess_data, train_model

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "baseline_raw.csv")


@pytest.fixture
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return preprocess_data(raw_df)


def test_preprocess_data_keeps_expected_columns_and_cleans_rows(raw_df: pd.DataFrame) -> None:
    result = preprocess_data(raw_df)

    assert list(result.columns) == ["subreddit", "post_id", "sentence_range", "text", "label"]
    assert result["text"].isna().sum() == 0
    assert result.duplicated(subset=["post_id", "sentence_range", "text", "label"]).sum() == 0
    assert list(result.index) == list(range(len(result)))
    assert len(result) == 4
    assert set(result["post_id"]) == {"syn_001", "syn_002", "syn_004", "syn_005"}


def test_add_dummy_baseline_adds_baseline_without_mutating_input(clean_df: pd.DataFrame) -> None:
    metrics = {500: {"accuracy": 0.75}}

    result = add_dummy_baseline(metrics, clean_df, clean_df)

    assert 500 in result
    assert result[500] == {"accuracy": 0.75}
    assert "baseline" in result
    assert set(result["baseline"]) >= {"accuracy", "0", "1", "macro avg", "weighted avg"}


def test_train_model_fits_pipeline_and_saves_file(clean_df: pd.DataFrame, tmp_path: Path) -> None:
    save_path = tmp_path / "baseline_model.pkl"
    max_features = 100

    model = train_model(clean_df, max_features=max_features, save_path=str(save_path))

    assert isinstance(model, Pipeline)
    assert set[Any](model.named_steps) == {"vectorizer", "classifier"}
    assert model.named_steps["vectorizer"].max_features == max_features
    assert save_path.exists()

    loaded = joblib.load(save_path)
    preds = loaded.predict(clean_df["text"])
    assert len(preds) == len(clean_df)
    assert set(preds).issubset({0, 1})
