"""Feature engineering shared by validation and final prediction.

Only load characteristics that are also available in the December chart
inputs are used: lane (pickup/delivery + coordinates), distance, equipment,
weight and day of week. market_index and quote_signal are deliberately NOT
model inputs (see README / report: they did not improve out-of-time accuracy
and the December chart does not provide them).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
EARTH_RADIUS_MI = 3958.8

FEATURES = [
    "distance", "log_distance", "weight", "weight_missing", "equipment",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "pickup", "delivery", "haversine", "distance_ratio", "dow",
]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def city_levels(*frames: pd.DataFrame) -> list[str]:
    """Fixed, sorted city vocabulary so category codes match across datasets."""
    cities: set[str] = set()
    for f in frames:
        cities |= set(f["pickup"]) | set(f["delivery"])
    return sorted(cities)


def build_features(df: pd.DataFrame, cities: list[str], extra: list[str] | None = None) -> pd.DataFrame:
    """Return the model matrix for a frame with the raw load columns.

    Data-quality handling done here:
      * negative weights are sign-flipped (|w| matches the positive distribution)
      * missing weight stays NaN (LightGBM handles it natively) + a missing flag
    """
    X = pd.DataFrame(index=df.index)
    X["distance"] = df["distance"].astype(float)
    X["log_distance"] = np.log(X["distance"])
    X["weight"] = df["weight"].abs()
    X["weight_missing"] = df["weight"].isna().astype(int)
    X["equipment"] = pd.Categorical(df["equipment"], EQUIPMENT)
    for c in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        X[c] = df[c].astype(float)
    X["pickup"] = pd.Categorical(df["pickup"], cities)
    X["delivery"] = pd.Categorical(df["delivery"], cities)
    X["haversine"] = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    # road distance / straight-line distance: catches mis-keyed distances
    X["distance_ratio"] = X["distance"] / X["haversine"]
    X["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    cols = FEATURES + (extra or [])
    for name in extra or []:
        X[name] = df[name]
    return X[cols]
