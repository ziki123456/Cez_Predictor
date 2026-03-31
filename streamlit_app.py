from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from vendor.data_helpers import load_stock_csv
from vendor.streamlit_helpers import (
    build_price_chart,
    download_latest_data,
    filter_chart_data,
    get_data_info_lines,
    read_data_info,
    train_model_from_app,
)


ROOT_DIR = Path(__file__).resolve().parent
DATA_CSV = ROOT_DIR / "data" / "raw" / "cez_pr.csv"
MODEL_PATH = ROOT_DIR / "models" / "cez_direction_model.joblib"


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


def load_model(model_path: Path) -> tuple[object, list[str]]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Chybí model: {model_path}. Nejdřív spusť train_model.py nebo tlačítko pro trénink."
        )

    model_obj = joblib.load(model_path)

    pipeline = model_obj.get("pipeline")
    feature_columns = model_obj.get("feature_columns")

    if pipeline is None or feature_columns is None:
        raise ValueError("Model soubor nemá správnou strukturu.")

    return pipeline, feature_columns


def predict_last_day(df: pd.DataFrame, pipeline: object, feature_columns: list[str]) -> dict:
    df_with_features = add_features(df)

    valid_df = df_with_features.dropna(subset=feature_columns).copy()
    if valid_df.empty:
        raise ValueError("V datech nejsou žádné řádky, ze kterých jde spočítat features.")

    last_row = valid_df.iloc[-1].copy()
    X_last = last_row[feature_columns].to_numpy(dtype=float).reshape(1, -1)

    pred = int(pipeline.predict(X_last)[0])

    if hasattr(pipeline, "predict_proba"):
        proba_up = float(pipeline.predict_proba(X_last)[0][1])
    else:
        proba_up = float("nan")

    verdict = "NAHORU" if pred == 1 else "DOLŮ"

    return {
        "date": pd.to_datetime(last_row["Date"]).date().isoformat(),
        "verdict": verdict,
        "proba_up": proba_up,
        "close": float(last_row["Close"]),
    }


def format_prediction_text(verdict: str, probability: float) -> str:
    if np.isnan(probability):
        return f"Predikce modelu: {verdict}"

    return f"Predikce modelu: {verdict} | Pravděpodobnost růstu: {probability:.4f}"


def main() -> None:
    st.set_page_config(
        page_title="CEZ Predictor",
        layout="wide",
    )

    st.title("CEZ Predictor")
    st.write(
        "Jednoduchá aplikace pro odhad směru dalšího obchodního dne akcie ČEZ "
        "na základě historických dat a ML modelu."
    )

    top_col1, top_col2, top_col3 = st.columns([1, 1, 2])

    with top_col1:
        if st.button("Aktualizovat data"):
            ok, message = download_latest_data()
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with top_col2:
        if st.button("Natrénovat model"):
            with st.spinner("Probíhá trénink modelu..."):
                ok, message = train_model_from_app()

            st.session_state["train_output"] = message

            if ok:
                st.success("Model byl znovu natrénován.")
                st.rerun()
            else:
                st.error("Trénink modelu selhal.")

    with top_col3:
        st.write("Data se načítají z lokálního CSV souboru v projektu.")
        st.write("Můžeš je aktualizovat z Yahoo Finance a pak znovu natrénovat model.")

    if "train_output" in st.session_state and st.session_state["train_output"]:
        with st.expander("Výstup z posledního tréninku", expanded=False):
            st.text(st.session_state["train_output"])

    try:
        df = load_stock_csv(DATA_CSV)
        pipeline, feature_columns = load_model(MODEL_PATH)
        result = predict_last_day(df, pipeline, feature_columns)
        data_info = get_data_info_lines()
    except Exception as e:
        st.error(f"Chyba: {e}")
        st.stop()

    st.subheader("Výsledek predikce")

    if result["verdict"] == "NAHORU":
        st.success(format_prediction_text(result["verdict"], result["proba_up"]))
    else:
        st.warning(format_prediction_text(result["verdict"], result["proba_up"]))

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric("Ticker", "CEZ.PR")

    with summary_col2:
        st.metric("Poslední den v datech", result["date"])

    with summary_col3:
        st.metric("Poslední Close", f"{result['close']:.2f}")

    with summary_col4:
        downloaded_value = data_info.get("Downloaded", "neuvedeno")
        st.metric("Poslední aktualizace", downloaded_value)

    st.subheader("Graf Close ceny")

    chart_period = st.segmented_control(
        "Období grafu",
        options=["3M", "6M", "1R", "3R", "5R", "Vše"],
        default="1R",
        selection_mode="single",
    )

    show_sma = st.checkbox("Zobrazit SMA 5 a SMA 10", value=False)

    filtered_chart_df = filter_chart_data(df, chart_period)

    if filtered_chart_df.empty:
        st.warning("Pro vybrané období nejsou k dispozici žádná data.")
    else:
        st.altair_chart(
            build_price_chart(filtered_chart_df, show_sma=show_sma),
            use_container_width=True
        )

        chart_start = filtered_chart_df["Date"].min().date().isoformat()
        chart_end = filtered_chart_df["Date"].max().date().isoformat()
        st.caption(
            f"Zobrazené období: {chart_start} až {chart_end} | "
            f"Počet řádků: {len(filtered_chart_df)}"
        )

        if show_sma:
            st.caption("Barvy v grafu: modrá = Close, žlutá = SMA 5, oranžová = SMA 10")

    st.subheader("Základní statistiky datasetu")

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    with stats_col1:
        st.metric("Počet řádků", f"{len(df)}")

    with stats_col2:
        st.metric("Min Close", f"{df['Close'].min():.2f}")

    with stats_col3:
        st.metric("Max Close", f"{df['Close'].max():.2f}")

    with stats_col4:
        st.metric("Průměr Close", f"{df['Close'].mean():.2f}")

    dataset_start = df["Date"].min().date().isoformat()
    dataset_end = df["Date"].max().date().isoformat()

    st.write(f"Období datasetu: {dataset_start} až {dataset_end}")

    st.subheader("Posledních 10 řádků dat")
    preview_df = df.tail(10).copy()
    preview_df["Date"] = preview_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(preview_df, use_container_width=True)

    with st.expander("Obsah data_info.txt"):
        st.text(read_data_info())

    st.divider()

    st.caption(
        "Upozornění: Tento nástroj je pouze demonstrační projekt pro analýzu historických dat akcie ČEZ. "
        "Predikce modelu vychází pouze z minulých cen a objemů obchodování. "
        "Nejedná se o investiční doporučení a skutečný vývoj ceny může být ovlivněn mnoha dalšími faktory."
    )


if __name__ == "__main__":
    main()