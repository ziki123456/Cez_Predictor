from pathlib import Path

import pandas as pd


def load_stock_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Chybí dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    df.columns = [str(c).strip() for c in df.columns]

    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

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