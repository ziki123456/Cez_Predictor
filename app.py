from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vendor.data_helpers import load_stock_csv


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parent
    data_csv: Path = root / "data" / "raw" / "cez_pr.csv"
    model_path: Path = root / "models" / "cez_direction_model.joblib"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def main() -> None:
    p = Paths()

    if not p.model_path.exists():
        raise FileNotFoundError(
            "Chybí model. Nejdřív spusť train_model.py, aby se vytvořil soubor v models/."
        )

    model_obj = joblib.load(p.model_path)

    pipeline = model_obj.get("pipeline")
    feature_columns = model_obj.get("feature_columns")

    if pipeline is None or feature_columns is None:
        raise ValueError("Model soubor nemá očekávanou strukturu (pipeline/feature_columns).")

    df = load_stock_csv(p.data_csv)
    df = add_features(df)

    valid = df.dropna(subset=feature_columns).copy()
    if valid.empty:
        raise ValueError("Nejde spočítat features (moc krátká data nebo samé NaN).")

    last_row = valid.iloc[-1].copy()
    last_date = pd.to_datetime(last_row["Date"]).date().isoformat()

    X_last = last_row[feature_columns].to_numpy(dtype=float).reshape(1, -1)

    pred = int(pipeline.predict(X_last)[0])

    if hasattr(pipeline, "predict_proba"):
        proba_up = float(pipeline.predict_proba(X_last)[0][1])
    else:
        proba_up = float("nan")

    verdict = "NAHORU" if pred == 1 else "DOLŮ"

    print("CEZ Predictor")
    print("Ticker: CEZ.PR")
    print(f"Poslední den v datech: {last_date}")
    print(f"Poslední Close: {float(last_row['Close']):.2f}")

    if not np.isnan(proba_up):
        print(f"Pravděpodobnost růstu: {proba_up:.4f}")
    else:
        print("Pravděpodobnost růstu: (model ji neumí spočítat)")

    print(f"Predikce dalšího dne: {verdict}")


if __name__ == "__main__":
    main()