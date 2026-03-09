
# CEZ Predictor (PV projekt)

Jednoduchá aplikace pro analýzu historických dat akcie ČEZ (CEZ.PR), která pomocí strojového učení zkusí odhadnout směr dalšího obchodního dne (NAHORU / DOLŮ).

Projekt pracuje s reálnými historickými daty akcie ČEZ, ze kterých vytváří technické atributy (features), trénuje klasifikační model a následně zobrazuje predikci v jednoduché webové aplikaci.

---

# Spuštění programu ve škole (bez IDE)

Projekt lze spustit pouze pomocí příkazové řádky (CMD / PowerShell). Není potřeba PyCharm ani jiné IDE.

## 1) Stažení projektu

Stáhni projekt z GitHubu jako ZIP nebo pomocí `git clone` a rozbal ho do složky.

Poté otevři CMD a přejdi do složky projektu například:

```
cd C:\Users\USERNAME\Desktop\Cez_Predictor
```

---

## 2) Vytvoření virtuálního prostředí

V kořenové složce projektu spusť:

```
py -3.13 -m venv .venv
```

Pokud verze 3.13 není dostupná:

```
py -m venv .venv
```

---

## 3) Aktivace virtuálního prostředí

```
.venv\Scripts\activate
```

Po aktivaci by se mělo zobrazit:

```
(.venv)
```

---

## 4) Instalace knihoven

Nainstaluj všechny potřebné knihovny:

```
pip install -r requirements.txt
```

---

## 5) Trénink modelu

Spusť:

```
py train_model.py
```

Po dokončení se vytvoří soubor:

```
models/cez_direction_model.joblib
```

a v konzoli se zobrazí metriky modelu (accuracy, confusion matrix, classification report).

---

## 6) Spuštění aplikace

Aplikace používá jednoduché webové rozhraní pomocí knihovny Streamlit.

Spusť:

```
streamlit run streamlit_app.py
```

Po spuštění se otevře webová stránka v prohlížeči.

V aplikaci je možné:

- zobrazit graf historické ceny
- zobrazit statistiky datasetu
- zobrazit klouzavé průměry (SMA 5 a SMA 10)
- natrénovat model
- zobrazit predikci dalšího dne

---

# Struktura projektu

```
Cez_Predictor/
│
├── streamlit_app.py
├── train_model.py
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── cez_pr.csv
│   └── data_info.txt
│
├── models/
│   └── cez_direction_model.joblib
│
├── notebooks/
│   └── CEZ_ML_training.ipynb
│
└── vendor/
```

---

# Data (původ dat)

Zdroj dat: **Yahoo Finance**  
Ticker: `CEZ.PR`  
Interval: `1d`

Data byla získána pomocí knihovny:

```
yfinance
```

Dataset obsahuje historická denní data:

- Open
- High
- Low
- Close
- Volume

Součástí projektu:

```
data/raw/cez_pr.csv
```

Metadata datasetu:

```
data/data_info.txt
```

Metadata obsahují:

- zdroj dat
- ticker
- datum stažení
- rozsah dat
- počet záznamů

Dataset obsahuje více než **1500 záznamů**.

---

# Jak funguje ML model

## Cíl (label)

Pro každý den `t`:

```
label = 1 pokud Close(t+1) > Close(t)
label = 0 jinak
```

Model tedy predikuje směr dalšího dne.

---

## Features (atributy)

Z historických dat se počítají technické atributy:

- return_1d
- sma_5
- sma_10
- volatility_5
- volume_change_1d
- hl_range
- open_close
- return_3d
- sma_ratio
- momentum_5

Tyto atributy popisují krátkodobý vývoj ceny a objemu obchodování.

---

## Preprocessing dat

Před tréninkem probíhá:

- odstranění NaN hodnot
- odstranění nekonečných hodnot
- výpočet technických atributů

Data se dělí časově:

```
80 % train
20 % test
```

Bez náhodného míchání, aby byl zachován časový pořádek.

---

## Škálování

Features jsou škálovány pomocí:

```
StandardScaler
```

---

## Použitý model

Model:

```
Logistic Regression
```

Model klasifikuje pravděpodobnost, že cena další den poroste nebo klesne.

---

## Vyhodnocení modelu

Při tréninku se vypisují metriky:

- accuracy
- confusion matrix
- classification report

Cílem projektu není vytvořit dokonalý trading model, ale ukázat celý proces:

- sběr dat
- preprocessing dat
- vytvoření features
- trénování modelu
- vyhodnocení modelu
- jednoduchá aplikace

---

# Poznámka

Tento projekt je demonstrační projekt pro práci s historickými daty a strojovým učením.

Predikce modelu vychází pouze z historických dat a nebere v úvahu ekonomické zprávy, makroekonomické události ani jiné faktory.

Nejedná se o investiční doporučení.
