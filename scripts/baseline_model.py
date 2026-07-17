import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv("../data/train_clean.csv")
    test_df = pd.read_csv("../data/test.csv")
    val_df = pd.read_csv("../data/validation.csv")
    return train_df, test_df, val_df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["subreddit", "post_id", "sentence_range", "text", "label"]].dropna(subset=["text"])
    df = df.drop_duplicates(subset=["post_id", "sentence_range", "text", "label"])
    df = df.reset_index(drop=True)
    return df
    

def compute_metrics_by_max_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, max_features_options: list[int]) -> dict:
    """Compute validation classification metrics for each candidate max_features value. """
    
    x_train, y_train = train_df["text"], train_df["label"]
    x_val, y_val = val_df["text"], val_df["label"]

    f1_by_max_features = {}

    for max_feat in max_features_options:
        pipeline = Pipeline(
            [
                ("vectorizer", TfidfVectorizer(max_features=max_feat, ngram_range=(1, 2), stop_words="english")),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipeline.fit(x_train, y_train)
        val_preds = pipeline.predict(x_val)

        report = classification_report(y_val, val_preds, output_dict=True)
        f1_by_max_features[max_feat] = report

    return f1_by_max_features


def add_dummy_baseline(
    metrics_by_max_features: dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Fit a DummyClassifier and return a dict with its report added under the key 'baseline'"""

    x_train, y_train = train_df["text"], train_df["label"]
    x_val, y_val = val_df["text"], val_df["label"]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    val_preds = dummy.predict(x_val)

    report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)

    updated_metrics = metrics_by_max_features.copy()
    updated_metrics["baseline"] = report
    return updated_metrics

def display_metrics_table(metrics_by_max_features: dict) -> pd.DataFrame:
    """Display classification reports in a comparison table, one row per config."""
    rows = []
    for max_feat, report in metrics_by_max_features.items():
        rows.append(
            {
                "max_features": max_feat,
                "accuracy": report["accuracy"],
                "precision_0": report["0"]["precision"],
                "recall_0": report["0"]["recall"],
                "f1_0": report["0"]["f1-score"],
                "precision_1": report["1"]["precision"],
                "recall_1": report["1"]["recall"],
                "f1_1": report["1"]["f1-score"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
            }
        )

    table = pd.DataFrame(rows).set_index("max_features").round(4)
    print(table.to_string())
    return table


def best_max_features_from_metrics(f1_dict: dict) -> tuple[int, dict]:
    """Return the (max_features, report) pair with the highest weighted F1.
    """
    best_max_features = None
    best_f1 = -1.0

    for max_feat, report in f1_dict.items():
        if not isinstance(max_feat, int): # Skips non-numeric keys, so DummyClassifier is not selected.
            continue
        val_f1 = report["weighted avg"]["f1-score"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_max_features = max_feat

    return best_max_features, f1_dict[best_max_features]


def train_model(df: pd.DataFrame, max_features: int, save_path: str) -> Pipeline:
    """
    Train a logistic regression model and save it to a file.
    """

    x_train, y_train = df["text"], df["label"]

    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, save_path)
    return pipeline



def main() -> None:
    train_df, test_df, val_df = load_data()
    train_df = preprocess_data(train_df)
    val_df = preprocess_data(val_df)    

    max_features_options = [500, 1000, 2500, 5000, 10000]
    max_features_results = compute_metrics_by_max_features(train_df, val_df, max_features_options)
    max_features_results = add_dummy_baseline(max_features_results, train_df, val_df)
    best_max_features, _ = best_max_features_from_metrics(max_features_results)

    model = train_model(train_df, best_max_features, save_path="../models/baseline_model.pkl")
    
    print(f"accuracy: {model.score(val_df['text'], val_df['label']):.3f}")
    print(f"best max_features: {best_max_features} with weighted F1 score: {max_features_results[best_max_features]['weighted avg']['f1-score']:.3f}")

if __name__ == "__main__":
    main()