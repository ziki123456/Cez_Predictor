from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
]


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Chybí dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    # Někdy bývá místo Date sloupec Datetime
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    # Pokud jsou sloupce "divné" (např. po exportu z MultiIndexu),
    # zkusíme je převést na čisté stringy bez prázdných znaků
    df.columns = [str(c).strip() for c in df.columns]

    # Někdy se stane, že se do CSV dostane navíc sloupec jako "Unnamed: 0"
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Zkusíme najít základní OHLCV sloupce i kdyby byly v jiném zápisu
    # (např. "Adj Close" nechceme povinně)
    col_map = {
        "open": None,
        "high": None,
        "low": None,
        "close": None,
        "volume": None,
    }

    for c in df.columns:
        cl = c.lower().replace("_", " ").strip()
        if cl == "open":
            col_map["open"] = c
        elif cl == "high":
            col_map["high"] = c
        elif cl == "low":
            col_map["low"] = c
        elif cl == "close":
            col_map["close"] = c
        elif cl == "volume":
            col_map["volume"] = c

    # Kontrola, že jsme našli všechno
    missing = [k for k, v in col_map.items() if v is None]
    if "Date" not in df.columns or missing:
        raise ValueError(
            "CSV nemá očekávané sloupce.\n"
            f"Chybí: {['Date'] if 'Date' not in df.columns else []}{missing}\n"
            f"Aktuální sloupce v CSV: {df.columns.tolist()}"
        )

    df = df.rename(
        columns={
            col_map["open"]: "Open",
            col_map["high"]: "High",
            col_map["low"]: "Low",
            col_map["close"]: "Close",
            col_map["volume"]: "Volume",
        }
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    return df


def add_features_and_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # return_1d = procentní změna Close
    df["return_1d"] = df["Close"].pct_change()

    # SMA
    df["sma_5"] = df["Close"].rolling(window=5).mean()
    df["sma_10"] = df["Close"].rolling(window=10).mean()

    # volatilita = směrodatná odchylka returnů
    df["volatility_5"] = df["return_1d"].rolling(window=5).std()

    # změna objemu
    df["volume_change_1d"] = df["Volume"].pct_change()

    df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)

    # poslední řádek nemá label (nemá t+1)
    df = df.iloc[:-1].copy()

    # odstraníme NaN vzniklé rolling/pct_change
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

    df = load_data(p.data_csv)
    df = add_features_and_label(df)

    train_df, test_df = time_split(df, train_ratio=0.8)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)

    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
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

    print(f"\n Model uložen do: {p.model_out}")


if __name__ == "__main__":
    main()