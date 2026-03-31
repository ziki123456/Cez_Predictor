# CEZ Predictor (PV projekt)

Jednoduchá aplikace, která z historických denních dat akcie ČEZ (CEZ.PR) spočítá technické atributy, natrénuje model a potom zkusí odhadnout směr dalšího obchodního dne (NAHORU / DOLŮ).

## Co to umí
- pracovat s reálnými historickými daty akcie ČEZ
- uložit dataset do CSV + uložit metadata o stažení do `data_info.txt`
- lokálně natrénovat ML model (`Logistic Regression`)
- lokálně spustit konzolovou aplikaci s predikcí
- lokálně spustit Streamlit aplikaci s grafem, statistikami a predikcí
- oddělit pomocné a knihovní části do složky `vendor/`

## Struktura projektu

```text
Cez_Predictor/
│
├── app.py
├── streamlit_app.py
├── train_model.py
├── requirements.txt
├── README.md
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
    ├── data_helpers.py
    └── streamlit_helpers.py
```

## Rozdělení autorského a pomocného kódu

V projektu je hlavní autorský kód oddělen od pomocných částí.

### Autorský kód
Za hlavní autorskou část projektu je považováno:
- návrh projektu
- práce s reálnými daty akcie ČEZ
- návrh cílové proměnné (`label`)
- návrh a výpočet features
- sestavení a trénink ML modelu
- vyhodnocení modelu
- logika predikce dalšího dne
- hlavní tok konzolové a Streamlit aplikace

Tyto části jsou hlavně v souborech:
- `train_model.py`
- `app.py`
- `streamlit_app.py`

### Pomocný / oddělený kód
Pomocné části, které nejsou jádrem projektu, jsou oddělené ve složce `vendor/`.

Jde hlavně o:
- obecné načítání a čištění CSV
- pomocné funkce pro aktualizaci dat
- pomocné funkce pro graf ve Streamlitu
- pomocné technické obslužné části aplikace

Tyto části jsou v souborech:
- `vendor/data_helpers.py`
- `vendor/streamlit_helpers.py`

Tím je splněno oddělení pomocného a neautorského kódu od hlavní logiky projektu.

## Data (původ a důkaz sběru)

Zdroj dat: Yahoo Finance (přes knihovnu `yfinance`)  
Ticker: `CEZ.PR`  
Interval: 1 den (`1d`)

V projektu je:
- `data/raw/cez_pr.csv` – stažený dataset
- `data/data_info.txt` – metadata o stažení (datum stažení, rozsah, počet řádků, sloupce)

Poznámka:
- dataset není převzatý jako hotový připravený dataset pro ML
- jde o reálná historická tržní data získaná pomocí skriptu
- data byla v průběhu vývoje získávána přes `yfinance`
- ve finální verzi projektu se s nimi pracuje lokálně přes CSV soubor uložený v projektu

## Jak funguje ML část

### Cíl (label)
Pro každý den `t`:
- `label = 1`, pokud `Close(t+1) > Close(t)`
- `label = 0` jinak

To znamená, že model predikuje směr dalšího obchodního dne, ne přesnou cenu.

### Features (atributy)
Z denních dat se počítá těchto 10 features:
- `return_1d` – denní procentní změna `Close`
- `sma_5` – klouzavý průměr `Close` za 5 dní
- `sma_10` – klouzavý průměr `Close` za 10 dní
- `volatility_5` – směrodatná odchylka `return_1d` za posledních 5 dní
- `volume_change_1d` – denní procentní změna `Volume`
- `hl_range` – relativní rozdíl mezi `High` a `Low`
- `open_close` – relativní rozdíl mezi `Open` a `Close`
- `return_3d` – procentní změna `Close` za 3 dny
- `sma_ratio` – poměr `sma_5 / sma_10`
- `momentum_5` – změna ceny vůči hodnotě před 5 dny

Řádky s `NaN` (vzniklé kvůli rolling oknům) se odstraní.  
Pokud vznikne `inf` (např. dělení nulou), nahradí se `NaN` a taky se to odstraní.

### Preprocessing
- načtení a vyčištění CSV
- převod datumů a číselných sloupců
- odstranění duplicit a neplatných řádků
- výpočet features
- odstranění `NaN` řádků po výpočtu features
- časové dělení na train/test bez míchání:
  - 80 % starší data = train
  - 20 % novější data = test
- škálování: `StandardScaler`
- model: `LogisticRegression`

### Vyhodnocení
Při tréninku se vypíše:
- accuracy
- confusion matrix
- classification report

Cílem projektu není dokonalá přesnost, ale správný proces:
- práce s reálnými daty
- preprocessing
- feature engineering
- model
- vyhodnocení
- použití v aplikaci

## Spuštění programu (bez IDE – pouze CMD)

Tento postup je určen pro spuštění projektu bez použití PyCharmu nebo jiného IDE.

### 1) Otevření příkazové řádky

Stiskni:

```text
Win + R
```

Napiš:

```text
cmd
```

Potvrď Enter.

---

### 2) Přechod do složky projektu

Pokud máš projekt stažený jako ZIP a rozbalený například na ploše:

```text
cd C:\Users\TVOJE_JMENO\Desktop\Cez_Predictor
```

Pokud je projekt jinde, přejdi do odpovídající složky pomocí `cd`.

---

### 3) Vytvoření virtuálního prostředí

V kořenové složce projektu spusť:

```text
py -3.13 -m venv .venv
```

Pokud by verze 3.13 nebyla dostupná:

```text
py -m venv .venv
```

---

### 4) Aktivace virtuálního prostředí

```text
.venv\Scripts\activate
```

Po aktivaci by se mělo zobrazit:

```text
(.venv) C:\...
```

---

### 5) Instalace potřebných knihoven

```text
pip install -r requirements.txt
```

Počkej, než se všechny balíčky nainstalují.

---

### 6) Trénink modelu

Spusť:

```text
py train_model.py
```

Po dokončení se vytvoří soubor:

```text
models\cez_direction_model.joblib
```

Zároveň se v konzoli vypíše accuracy a další metriky.

---

### 7) Spuštění konzolové aplikace (predikce)

Spusť:

```text
py app.py
```

Aplikace vypíše:
- Ticker
- Poslední den v datech
- Pravděpodobnost růstu
- Predikci dalšího dne (NAHORU / DOLŮ)

---

### 8) Spuštění Streamlit aplikace

Spusť:

```text
streamlit run streamlit_app.py
```

Ve webovém rozhraní se zobrazí:
- výsledek predikce
- základní statistiky datasetu
- graf Close ceny
- poslední řádky dat
- možnost aktualizovat data
- možnost znovu natrénovat model

## Poznámka

- Projekt nevyžaduje žádné IDE.
- Stačí mít nainstalovaný Python.
- Internet je potřeba pouze pro instalaci knihoven a případnou aktualizaci dat.
- Samotná predikce funguje offline, pokud už jsou data a model vytvořeny.

## Omezení
- Predikce není investiční doporučení a nezaručuje zisk.
- Akciové trhy jsou těžko předvídatelné, takže přesnost může být nízká.
- Model používá pouze historická tržní data a jednoduché technické atributy.
- Nejde o profesionální trading systém.