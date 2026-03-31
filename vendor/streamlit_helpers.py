from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import altair as alt
import pandas as pd
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT_DIR / "data" / "raw" / "cez_pr.csv"
DATA_INFO = ROOT_DIR / "data" / "data_info.txt"
TRAIN_SCRIPT = ROOT_DIR / "train_model.py"

TICKER = "CEZ.PR"
INTERVAL = "1d"


def download_latest_data() -> tuple[bool, str]:
    try:
        end_date = datetime.now(timezone.utc).date()
        start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=10)).date()

        df = yf.download(
            TICKER,
            start=str(start_date),
            end=str(end_date),
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            return False, "Stažený dataset je prázdný."

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_CSV, index=False)

        downloaded_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        start_real = pd.to_datetime(df["Date"]).min().date().isoformat()
        end_real = pd.to_datetime(df["Date"]).max().date().isoformat()
        columns_text = ", ".join(map(str, df.columns.tolist()))

        info_text = (
            f"Source: Yahoo Finance (via yfinance)\n"
            f"Ticker: {TICKER}\n"
            f"Downloaded: {downloaded_utc}\n"
            f"Interval: {INTERVAL}\n"
            f"Start date: {start_real}\n"
            f"End date: {end_real}\n"
            f"Records: {len(df)}\n"
            f"Columns: {columns_text}\n"
        )

        DATA_INFO.write_text(info_text, encoding="utf-8")

        return True, f"Data byla aktualizována. Poslední den v datech: {end_real}"

    except Exception as e:
        return False, f"Aktualizace dat selhala: {e}"


def train_model_from_app() -> tuple[bool, str]:
    try:
        if not TRAIN_SCRIPT.exists():
            return False, f"Chybí soubor: {TRAIN_SCRIPT}"

        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        output_parts = []
        if result.stdout.strip():
            output_parts.append(result.stdout.strip())
        if result.stderr.strip():
            output_parts.append(result.stderr.strip())

        output_text = "\n\n".join(output_parts).strip()
        if not output_text:
            output_text = "Trénink doběhl bez výstupu."

        if result.returncode != 0:
            return False, output_text

        return True, output_text

    except Exception as e:
        return False, f"Trénink selhal: {e}"


def read_data_info() -> str:
    if not DATA_INFO.exists():
        return "Soubor data_info.txt zatím neexistuje."

    return DATA_INFO.read_text(encoding="utf-8")


def get_data_info_lines() -> dict[str, str]:
    if not DATA_INFO.exists():
        return {}

    lines = DATA_INFO.read_text(encoding="utf-8").splitlines()
    info = {}

    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()

    return info


def filter_chart_data(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    if df.empty:
        return df

    last_date = df["Date"].max()

    period_map = {
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1R": pd.DateOffset(years=1),
        "3R": pd.DateOffset(years=3),
        "5R": pd.DateOffset(years=5),
        "Vše": None,
    }

    offset = period_map[period_label]
    if offset is None:
        return df.copy()

    start_date = last_date - offset
    return df[df["Date"] >= start_date].copy()


def build_price_chart(chart_df: pd.DataFrame, show_sma: bool) -> alt.Chart:
    chart_data = chart_df.copy()

    chart_data["SMA_5"] = chart_data["Close"].rolling(window=5).mean()
    chart_data["SMA_10"] = chart_data["Close"].rolling(window=10).mean()

    chart_data["DateText"] = chart_data["Date"].dt.strftime("%Y-%m-%d")
    chart_data["CloseRounded"] = chart_data["Close"].round(2)
    chart_data["SMA5Rounded"] = chart_data["SMA_5"].round(2)
    chart_data["SMA10Rounded"] = chart_data["SMA_10"].round(2)

    base = alt.Chart(chart_data).encode(
        x=alt.X("Date:T", title="Datum")
    )

    price_line = base.mark_line().encode(
        y=alt.Y("Close:Q", title="Cena"),
        color=alt.value("#7ec8ff"),
        tooltip=[
            alt.Tooltip("DateText:N", title="Datum"),
            alt.Tooltip("CloseRounded:Q", title="Close", format=".2f"),
        ],
    )

    chart = price_line

    if show_sma:
        sma5_line = base.mark_line(strokeDash=[6, 3]).encode(
            y=alt.Y("SMA_5:Q"),
            color=alt.value("#f4d35e"),
            tooltip=[
                alt.Tooltip("DateText:N", title="Datum"),
                alt.Tooltip("SMA5Rounded:Q", title="SMA 5", format=".2f"),
            ],
        )

        sma10_line = base.mark_line(strokeDash=[2, 2]).encode(
            y=alt.Y("SMA_10:Q"),
            color=alt.value("#ee964b"),
            tooltip=[
                alt.Tooltip("DateText:N", title="Datum"),
                alt.Tooltip("SMA10Rounded:Q", title="SMA 10", format=".2f"),
            ],
        )

        chart = alt.layer(price_line, sma5_line, sma10_line)

    return chart.properties(height=420).interactive()