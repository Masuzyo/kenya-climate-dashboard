"""Data loading utilities for the Zambia monthly climate/vegetation dashboard."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "zambia_monthly_ingest"
PARQUET_CACHE = DATA_DIR / "combined.parquet"
DISTRICT_MAPPING = DATA_DIR / "zambia_district_mapping.csv"

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
    "dtr_c": {
        "label": "Diurnal Temperature Range",
        "unit": "\u00b0C",
        "colorscale": "RdBu_r",
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
    "evi": {
        "label": "Enhanced Vegetation Index (EVI)",
        "unit": "",
        "colorscale": "YlGn",
    },
    "veg_cover_pct": {
        "label": "Vegetation Coverage",
        "unit": "%",
        "colorscale": "Greens",
    },
    "forest_pct": {
        "label": "Forest Cover",
        "unit": "%",
        "colorscale": "Greens",
    },
    "savanna_pct": {
        "label": "Savanna Cover",
        "unit": "%",
        "colorscale": "YlOrBr",
    },
    "wetland_pct": {
        "label": "Wetland Cover",
        "unit": "%",
        "colorscale": "Teal",
    },
    "cropland_pct": {
        "label": "Cropland Cover",
        "unit": "%",
        "colorscale": "YlOrRd",
    },
    "urban_pct": {
        "label": "Urban Cover",
        "unit": "%",
        "colorscale": "Reds",
    },
    "surface_water_occurrence_pct": {
        "label": "Surface Water Occurrence",
        "unit": "%",
        "colorscale": "Blues",
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
    "wind_speed_ms": {
        "label": "Wind Speed (10 m)",
        "unit": "m/s",
        "colorscale": "Purples",
    },
}


def _read_all_csv() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("zambia_*.csv"))
    # Exclude the district mapping file which also matches the prefix
    files = [f for f in files if "district_mapping" not in f.name]
    
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}. Run modis_zambia_monthly_ingest.py "
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

        # Merge district mapping
        if DISTRICT_MAPPING.exists():
            mapping_df = pd.read_csv(DISTRICT_MAPPING)
            
            # Cast to float32 to ensure exact match with parquet columns
            df['lat_match'] = df['lat'].astype('float32')
            df['lon_match'] = df['lon'].astype('float32')
            mapping_df['lat_match'] = mapping_df['lat'].astype('float32')
            mapping_df['lon_match'] = mapping_df['lon'].astype('float32')
            
            df = df.merge(
                mapping_df[['lat_match', 'lon_match', 'district']], 
                on=['lat_match', 'lon_match'], 
                how='left'
            )
            df = df.drop(columns=['lat_match', 'lon_match'])
            df['district'] = df['district'].fillna('Unknown')
            
        return df
    return _read_all_csv()


def district_monthly_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Return a lightweight dataframe grouped by month and district."""
    if 'district' not in df.columns:
        return df
    return df.groupby(["month", "district"])[list(VARIABLES.keys())].mean().reset_index()


def build_parquet_cache() -> Path:
    """Rebuild the compact parquet cache from all monthly CSVs.

    Run this after generating new months with modis_zambia_monthly_ingest.py
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
