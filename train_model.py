from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vendor.data_helpers import load_stock_csv

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parent
    data_csv: Path = root / "data" / "raw" / "cez_pr.csv"
    model_out: Path = root / "models" / "cez_direction_model.joblib"


FEATURE_COLUMNS = [
    "return_1d",
    "sma_5",
    "sma_10",
    "volatility_5",
    "volume_change_1d",
    "hl_range",
    "open_close",
    "return_3d",
    "sma_ratio",
    "momentum_5",
]


def add_features_and_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1d"] = df["Close"].pct_change()
    df["sma_5"] = df["Close"].rolling(window=5).mean()
    df["sma_10"] = df["Close"].rolling(window=10).mean()
    df["volatility_5"] = df["return_1d"].rolling(window=5).std()
    df["volume_change_1d"] = df["Volume"].pct_change()

    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close"] = (df["Close"] - df["Open"]) / df["Open"]
    df["return_3d"] = df["Close"].pct_change(periods=3)
    df["sma_ratio"] = df["sma_5"] / df["sma_10"]
    df["momentum_5"] = df["Close"] / df["Close"].shift(5) - 1

    df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.iloc[:-1].copy()
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    return df


def time_split(df: pd.DataFrame, train_ratio: float = 0.8):
    if not (0.5 <= train_ratio <= 0.9):
        raise ValueError("train_ratio dej mezi 0.5 a 0.9")

    n = len(df)
    if n < 200:
        raise ValueError(f"Dataset po feature engineering je moc malý ({n} řádků).")

    split_idx = int(n * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def main() -> None:
    p = Paths()
    p.model_out.parent.mkdir(parents=True, exist_ok=True)

    df = load_stock_csv(p.data_csv)
    df = add_features_and_label(df)

    train_df, test_df = time_split(df, train_ratio=0.8)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)

    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("=== CEZ Predictor: vyhodnocení modelu ===")
    print(f"Dataset po úpravách: {len(df)} řádků")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix (řádky=skutečnost, sloupce=predikce):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "train_end_date": str(train_df["Date"].max().date()),
            "test_start_date": str(test_df["Date"].min().date()),
        },
        p.model_out,
    )

    print(f"\nModel uložen do: {p.model_out}")


if __name__ == "__main__":
    main()