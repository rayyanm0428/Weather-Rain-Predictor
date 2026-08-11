"""
fetch_live_weather.py

Pulls REAL current + recent weather data from the Open-Meteo API
(https://open-meteo.com) — free, no API key required, no rate-limit signup.

This is the "API" component of the project: it hits a live endpoint and
returns fresh data in the same shape as our training data, so the trained
model can make a prediction on today's actual conditions.

Run this on your own machine (it needs real internet access):
    python src/fetch_live_weather.py --lat 39.95 --lon -75.16   # Philadelphia

Swap --lat/--lon for any city. Find coordinates at latlong.net.
"""

import argparse
import requests
import pandas as pd

def fetch_current(lat: float, lon: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["current"]
    return {
        "temp_f": data["temperature_2m"],
        "humidity_pct": data["relative_humidity_2m"],
        "pressure_hpa": data["surface_pressure"],
        "wind_mph": data["wind_speed_10m"],
        "precip_now_mm": data["precipitation"],
    }

def fetch_recent_history(lat: float, lon: float, days: int = 14) -> pd.DataFrame:
    """Pulls the last N days of real observed weather - useful for computing
    the pressure_drop feature the model expects, or for periodically topping
    up your training set with fresh real data instead of synthetic data."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,surface_pressure_mean,wind_speed_10m_max,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    daily = resp.json()["daily"]
    df = pd.DataFrame({
        "date": daily["time"],
        "temp_f": daily["temperature_2m_mean"],
        "humidity_pct": daily["relative_humidity_2m_mean"],
        "pressure_hpa": daily["surface_pressure_mean"],
        "wind_mph": daily["wind_speed_10m_max"],
        "rained": [1 if p > 0.1 else 0 for p in daily["precipitation_sum"]],
    })
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    args = parser.parse_args()

    print("Fetching current conditions...")
    current = fetch_current(args.lat, args.lon)
    print(current)

    print("\nFetching last 14 days of real history...")
    history = fetch_recent_history(args.lat, args.lon)
    print(history.tail())
