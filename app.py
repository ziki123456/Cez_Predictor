from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parent
    data_csv: Path = root / "data" / "raw" / "cez_pr.csv"
    model_path: Path = root / "models" / "cez_direction_model.joblib"


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Chybí dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV nemá povinné sloupce: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1d"] = df["Close"].pct_change()
    df["sma_5"] = df["Close"].rolling(window=5).mean()
    df["sma_10"] = df["Close"].rolling(window=10).mean()
    df["volatility_5"] = df["return_1d"].rolling(window=5).std()
    df["volume_change_1d"] = df["Volume"].pct_change()

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

    df = load_data(p.data_csv)
    df = add_features(df)

    last_row = df.iloc[-1].copy()
    last_date = pd.to_datetime(last_row["Date"]).date().isoformat()

    # poslední řádek musí mít všechny features (kvůli rolling oknům)
    if df[feature_columns].iloc[-1].isna().any():
        # když by to někdy spadlo (např. příliš krátká historie), zkusíme vzít poslední řádek bez NaN
        valid = df.dropna(subset=feature_columns)
        if valid.empty:
            raise ValueError("Nejde spočítat features (moc krátká data nebo samé NaN).")
        last_row = valid.iloc[-1].copy()
        last_date = pd.to_datetime(last_row["Date"]).date().isoformat()

    X_last = last_row[feature_columns].to_numpy(dtype=float).reshape(1, -1)

    pred = int(pipeline.predict(X_last)[0])

    # pravděpodobnost růstu (třída 1)
    if hasattr(pipeline, "predict_proba"):
        proba_up = float(pipeline.predict_proba(X_last)[0][1])
    else:
        proba_up = float("nan")

    verdict = "NAHORU" if pred == 1 else "DOLŮ"

    print("CEZ Predictor")
    print(f"Ticker: CEZ.PR")
    print(f"Poslední den v datech: {last_date}")
    if not np.isnan(proba_up):
        print(f"Pravděpodobnost růstu: {proba_up:.4f}")
    else:
        print("Pravděpodobnost růstu: (model ji neumí spočítat)")
    print(f"Predikce dalšího dne: {verdict}")


if __name__ == "__main__":
    main()