"""
fetch_historical_data.py

Pulls REAL multi-year daily weather history from the Open-Meteo Archive API
(https://open-meteo.com) — free, no API key required. This replaces
data/generate_sample_data.py's synthetic output with actual observed
weather, in the exact same CSV schema, so nothing downstream (features.py,
train_baseline.py, train_improved.py) needs to change.

Run this on a machine with real internet access (this environment's sandbox
can't reach external weather APIs):

    python src/fetch_historical_data.py --lat 39.95 --lon -75.16 --years 5

Find coordinates for any city at latlong.net.
"""

import argparse
import requests
import pandas as pd
from pathlib import Path

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Same variables/units as fetch_live_weather.py's fetch_recent_history(),
# so historical and live data always line up.
DAILY_VARIABLES = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
]


def fetch_historical(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARIABLES),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "temp_f": daily["temperature_2m_mean"],
        "humidity_pct": daily["relative_humidity_2m_mean"],
        "pressure_hpa": daily["surface_pressure_mean"],
        "wind_mph": daily["wind_speed_10m_max"],
        "rained": [1 if (p or 0) > 0.1 else 0 for p in daily["precipitation_sum"]],
    })

    # Archive data can have occasional gaps (station outages, etc.) — drop
    # incomplete rows rather than silently training on missing values.
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} incomplete row(s) out of {before}.")

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch real historical weather data.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--years", type=int, default=5, help="Years of history to pull (default: 5)")
    parser.add_argument(
        "--output", type=str, default="data/weather_history.csv",
        help="Output CSV path (default overwrites the synthetic dataset)",
    )
    args = parser.parse_args()

    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=2)  # archive lags a couple days
    start = end - pd.Timedelta(days=365 * args.years)

    print(f"Fetching {args.years} years of real weather for ({args.lat}, {args.lon})...")
    print(f"Range: {start.date()} to {end.date()}")

    df = fetch_historical(args.lat, args.lon, start.date().isoformat(), end.date().isoformat())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} days of real weather data -> {out_path}")
    print(df.head())
    print(f"\nRain frequency: {df['rained'].mean():.1%} of days")


if __name__ == "__main__":
    main()
