"""Data loading utilities for the Kenya monthly climate/vegetation dashboard."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "kenya_monthly_ingest"
PARQUET_CACHE = DATA_DIR / "combined.parquet"

# Metadata for each variable: display label, unit, and a Plotly colorscale
# chosen to make sense for that quantity (e.g. green for vegetation).
VARIABLES: dict[str, dict[str, str]] = {
    "max_temp_c": {
        "label": "Max Temperature",
        "unit": "\u00b0C",
        "colorscale": "Inferno",
    },
    "min_temp_c": {
        "label": "Min Temperature",
        "unit": "\u00b0C",
        "colorscale": "Cividis",
    },
    "mean_temp_c": {
        "label": "Mean Temperature",
        "unit": "\u00b0C",
        "colorscale": "Magma",
    },
    "elevation_m": {
        "label": "Elevation",
        "unit": "m",
        "colorscale": "Earth",
    },
    "ndvi": {
        "label": "Vegetation Greenness (NDVI)",
        "unit": "",
        "colorscale": "RdYlGn",
    },
    "veg_cover_pct": {
        "label": "Vegetation Coverage",
        "unit": "%",
        "colorscale": "Greens",
    },
    "rain_mm": {
        "label": "Rainfall",
        "unit": "mm",
        "colorscale": "Blues",
    },
    "humidity_rh_pct": {
        "label": "Relative Humidity",
        "unit": "%",
        "colorscale": "Teal",
    },
    "soil_moisture_m3m3": {
        "label": "Soil Moisture",
        "unit": "m\u00b3/m\u00b3",
        "colorscale": "YlGnBu",
    },
}


def _read_all_csv() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("kenya_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}. Run modis_kenya_monthly_ingest.py "
            "with --csv first."
        )
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    return df


def load_combined() -> pd.DataFrame:
    """Load the combined dataset, preferring the compact parquet cache.

    The parquet cache (built by running this file directly, e.g.
    `python dashboard_data.py`) is what gets committed to git for
    deployment: it's a fraction of the size of the raw monthly CSVs and
    loads far faster, which matters for cloud cold-starts.
    """
    if PARQUET_CACHE.exists():
        df = pd.read_parquet(PARQUET_CACHE)
        df["month"] = pd.to_datetime(df["month"])
        return df
    return _read_all_csv()


def build_parquet_cache() -> Path:
    """Rebuild the compact parquet cache from all monthly CSVs.

    Run this after generating new months with modis_kenya_monthly_ingest.py
    so the committed cache (and deployed dashboard) picks up the new data.
    """
    df = _read_all_csv()
    for col in VARIABLES:
        # Older CSVs may predate a variable added later (e.g. min_temp_c);
        # skip gracefully instead of erroring on a missing column.
        if col in df.columns:
            df[col] = df[col].astype("float32")
    df["lat"] = df["lat"].astype("float32")
    df["lon"] = df["lon"].astype("float32")
    PARQUET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_CACHE, index=False, compression="snappy")
    return PARQUET_CACHE


if __name__ == "__main__":
    path = build_parquet_cache()
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"Wrote {path} ({size_mb:.1f} MB)")
