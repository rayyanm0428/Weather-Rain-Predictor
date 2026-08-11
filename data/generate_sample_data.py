"""
generate_sample_data.py

Creates a synthetic-but-realistic daily weather history so the rest of the
pipeline (feature engineering, training, evaluation) can be built and tested
end-to-end without needing a live API key.

This is a STAND-IN for real historical data. When you're ready to use real
data, replace this file's output with data pulled from:
  - NOAA Climate Data Online: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
  - Or export history from OpenWeatherMap's One Call API (needs a paid key
    for history), or Open-Meteo's free historical API (no key needed):
    https://open-meteo.com/en/docs/historical-weather-api

The synthetic generator below models:
  - Seasonal temperature curve (sine wave across the year + yearly noise)
  - Humidity that rises when temperature drops (loosely realistic)
  - Pressure that drifts and drops before rain
  - Rain probability driven by humidity + pressure drop (so the model has
    real signal to learn, not pure noise)
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate(n_years: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_days = n_years * 365
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")

    day_of_year = dates.dayofyear.values
    # Seasonal temperature curve (peak in summer, trough in winter), Fahrenheit
    seasonal_temp = 55 + 25 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Hidden "storm system" index: real weather systems persist for several
    # days, which is WHY tomorrow's weather is predictable from today's at
    # all. Model this as a slow-moving AR(1) process (high persistence)
    # rather than pure day-to-day noise.
    storm_index = np.zeros(n_days)
    for i in range(1, n_days):
        storm_index[i] = 0.85 * storm_index[i-1] + rng.normal(0, 0.5)

    temp = seasonal_temp - 3 * storm_index + rng.normal(0, 3, n_days)

    # Humidity: rises with storm index (stormier = more humid)
    humidity = 55 + 8 * storm_index + rng.normal(0, 5, n_days)
    humidity = np.clip(humidity, 15, 100)

    # Pressure: falls when storm index rises
    pressure = 1015 - 6 * storm_index + rng.normal(0, 1.5, n_days)
    pressure_drop = np.concatenate([[0], np.diff(pressure)])

    # Rain probability driven mainly by the persistent storm index (so
    # today's conditions carry real information about tomorrow), plus a
    # same-day nudge from humidity/pressure.
    rain_signal = storm_index + (humidity - 55) / 40 - pressure_drop / 4
    rain_prob = 1 / (1 + np.exp(-1.4 * rain_signal))  # sigmoid
    rained_today = rng.binomial(1, np.clip(rain_prob, 0.03, 0.92))

    wind_speed = np.clip(rng.normal(8, 4, n_days), 0, None)

    df = pd.DataFrame({
        "date": dates,
        "temp_f": temp.round(1),
        "humidity_pct": humidity.round(1),
        "pressure_hpa": pressure.round(1),
        "wind_mph": wind_speed.round(1),
        "rained": rained_today,
    })
    return df

if __name__ == "__main__":
    out_path = Path(__file__).parent / "weather_history.csv"
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} days of sample weather data -> {out_path}")
    print(df.head())
    print(f"\nRain frequency: {df['rained'].mean():.1%} of days")
