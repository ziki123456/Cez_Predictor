# CEZ Predictor (PV projekt)

Jednoduchá aplikace, která z historických denních dat akcie ČEZ (CEZ.PR) spočítá technické atributy, natrénuje model a potom zkusí odhadnout směr dalšího obchodního dne (NAHORU / DOLŮ).

## Co to umí
- stáhnout reálná data (řešeno v Google Colabu přes yfinance)
- uložit dataset do CSV + uložit metadata o stažení do data_info.txt
- lokálně natrénovat ML model (Logistic Regression)
- lokálně spustit aplikaci, která vypíše poslední den v datech a predikci dalšího dne

## Struktura projektu
## Struktura projektu

```
Cez_Predictor/
│
├── app.py
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
│
└── vendor/
```
## Data (původ a důkaz sběru)
Zdroj dat: Yahoo Finance (přes knihovnu yfinance)  
Ticker: CEZ.PR  
Interval: 1 den (1d)

V projektu je:
- `data/raw/cez_pr.csv` – stažený dataset
- `data/data_info.txt` – metadata o stažení (datum stažení, rozsah, počet řádků, sloupce)

Poznámka: dataset není žádný hotový dataset z internetu. Data byla stažena skriptem v Colabu.

## Jak funguje ML část
### Cíl (label)
Pro každý den `t`:
- label = 1 pokud `Close(t+1) > Close(t)`
- label = 0 jinak

To znamená, že predikujeme směr dalšího dne, ne cenu.

### Features (atributy)
Z denních dat se počítá minimálně těchto 5 features:
- `return_1d` – denní procentní změna Close
- `sma_5` – klouzavý průměr Close za 5 dní
- `sma_10` – klouzavý průměr Close za 10 dní
- `volatility_5` – směrodatná odchylka return_1d za posledních 5 dní
- `volume_change_1d` – denní procentní změna Volume

Řádky s NaN (vzniklé kvůli rolling oknům) se odstraní.
Pokud vznikne inf (např. dělení nulou), nahradí se NaN a taky se to odstraní.

### Preprocessing
- odstranění NaN řádků po výpočtu features
- časové dělení na train/test bez míchání:
  - 80 % starší data train
  - 20 % novější data test
- škálování: StandardScaler (jen na features)
- model: LogisticRegression

### Vyhodnocení
Při tréninku se vypíše:
- accuracy
- confusion matrix
- classification report

Cílem není “dokonalá přesnost”, ale správný proces (sběr dat, preprocessing, model, vyhodnocení, aplikace).

## Jak to spustit (PyCharm doma)
1) Nainstaluj knihovny:
- `pip install -r requirements.txt`

2) Trénink modelu:
- spusť `train_model.py`
- vytvoří se `models/cez_direction_model.joblib`

3) Predikce:
- spusť `app.py`
- vypíše poslední datum v datech, pravděpodobnost a NAHORU/DOLŮ

## Spuštění ve škole (bez IDE)
Ve škole se to bude spouštět z příkazové řádky (CMD/PowerShell). Postup doplním v rámci testování deploye.

## Omezení
- Predikce není investiční doporučení a nezaručuje zisk.
- Akciové trhy jsou hodně náhodné, takže přesnost může být nízká.
- Model používá jen jednoduché technické atributy, není to profesionální trading systém.