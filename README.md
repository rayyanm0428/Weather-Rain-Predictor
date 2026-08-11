# Next-Day Rain Predictor

Predicts whether it'll rain tomorrow in Philadelphia based on today's weather (temperature, humidity, pressure, wind). Built end-to-end: real API data, two models compared, honest evaluation.

## Data

Five years of real daily weather (1,826 days), pulled from [Open-Meteo](https://open-meteo.com)'s free archive API via `src/fetch_historical_data.py`. A synthetic data generator (`data/generate_sample_data.py`) is included as an offline fallback. `src/fetch_live_weather.py` pulls today's real conditions for making a live prediction.

## Approach

- **Baseline** (`train_baseline.py`): Logistic Regression on raw daily readings
- **Improved** (`train_improved.py`): Random Forest with added multi-day trend features

Both use a time-based train/test split (no shuffling) to avoid leaking future data into training.

## Results

| Model | Accuracy | F1 | vs. naive baseline |
|---|---|---|---|
| Naive (majority class) | 56.7% | - | - |
| Logistic Regression | 67.4% | 0.573 | +10.7 pts |
| Random Forest | 66.6% | 0.567 | +9.9 pts |

Both beat the naive baseline. Notably, the Random Forest didn't outperform the simpler model. Likely the extra trend features add noise rather than signal at this data size. Both models are also better at predicting dry days than catching rainy ones (66% precision, 51% recall on "Rain"), which would be the first thing to improve next.

~67% accuracy is real signal, not a production forecaster. Next-day weather from a handful of surface readings alone is a genuinely hard problem.

## Running it

```bash
pip install -r requirements.txt
python src/fetch_historical_data.py --lat 39.95 --lon -75.16 --years 5
python src/train_baseline.py
python src/train_improved.py
python src/fetch_live_weather.py --lat 39.95 --lon -75.16
```

## Project structure

```
weather-predictor/
├── data/                          # generated dataset (real or synthetic)
├── src/
│   ├── features.py                # feature engineering, train/test split
│   ├── train_baseline.py          # logistic regression
│   ├── train_improved.py          # random forest
│   ├── fetch_historical_data.py   # real multi-year history (Open-Meteo)
│   └── fetch_live_weather.py      # real current conditions (Open-Meteo)
├── models/                        # saved trained models
└── requirements.txt
```