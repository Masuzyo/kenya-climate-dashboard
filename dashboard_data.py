"""Data loading utilities for the Kenya monthly climate/vegetation dashboard."""

from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

DATA_DIR = Path(__file__).parent / "kenya_monthly_ingest"
PARQUET_CACHE = DATA_DIR / "combined.parquet"
COUNTY_MAPPING = DATA_DIR / "kenya_county_mapping.csv"

# GeoTIFF multi-band order produced by modis_kenya_monthly_ingest.py
TIFF_BAND_NAMES = [
    "max_temp_c",
    "min_temp_c",
    "mean_temp_c",
    "dtr_c",
    "elevation_m",
    "ndvi",
    "evi",
    "veg_cover_pct",
    "forest_pct",
    "savanna_pct",
    "wetland_pct",
    "cropland_pct",
    "urban_pct",
    "surface_water_occurrence_pct",
    "rain_mm",
    "humidity_rh_pct",
    "soil_moisture_m3m3",
    "wind_speed_ms",
]

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


def read_tiff_file(tif_path: Path) -> pd.DataFrame:
    """Extract georeferenced spatial coordinates and 18 variable bands from a GeoTIFF raster."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required to read GeoTIFF (.tif) files. Please install rasterio.")
    
    # Extract month YYYY-MM from filename (e.g. kenya_2001_01.tif)
    match = re.search(r"(\d{4})_(\d{2})", tif_path.name)
    month_str = f"{match.group(1)}-{match.group(2)}" if match else "unknown"
    
    with rasterio.open(tif_path) as src:
        data = src.read()  # Shape: (bands, height, width)
        transform = src.transform
        height, width = src.height, src.width
        
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        lons = np.array(xs).flatten()
        lats = np.array(ys).flatten()
        
        df_dict = {
            "month": month_str,
            "lat": lats.astype("float32"),
            "lon": lons.astype("float32")
        }
        
        for idx, band_name in enumerate(TIFF_BAND_NAMES):
            if idx < data.shape[0]:
                df_dict[band_name] = data[idx].flatten().astype("float32")
                
        df_tif = pd.DataFrame(df_dict)
        # Filter out pixels outside the Kenya boundary polygon & replace band sentinels with NaN
        if "max_temp_c" in df_tif.columns:
            df_tif = df_tif[df_tif["max_temp_c"] > -9000].copy()
        for col in TIFF_BAND_NAMES:
            if col in df_tif.columns:
                df_tif.loc[df_tif[col] < -9000, col] = np.nan
        return df_tif


def _read_all_tiff() -> pd.DataFrame:
    """Read all monthly GeoTIFF rasters from DATA_DIR."""
    files = sorted(DATA_DIR.glob("kenya_*.tif"))
    files = [f for f in files if "lookup" not in f.name]
    
    if not files:
        raise FileNotFoundError(
            f"No GeoTIFF (.tif) files found in {DATA_DIR}. Run modis_kenya_monthly_ingest.py first."
        )
    
    frames = [read_tiff_file(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    return df


def _read_all_csv() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("kenya_*.csv"))
    # Exclude county mapping and lookup files
    files = [f for f in files if "county_mapping" not in f.name and "lookup" not in f.name]
    
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}. Run modis_kenya_monthly_ingest.py first."
        )
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    return df


def _read_raw_data() -> pd.DataFrame:
    """Read raw monthly data, prioritizing GeoTIFF (.tif) rasters over CSV files."""
    tif_files = [f for f in DATA_DIR.glob("kenya_*.tif") if "lookup" not in f.name]
    if HAS_RASTERIO and tif_files:
        print(f"Reading {len(tif_files)} GeoTIFF raster files...")
        return _read_all_tiff()
    print("Reading raw CSV files...")
    return _read_all_csv()


def load_combined() -> pd.DataFrame:
    """Load the combined dataset, preferring the compact parquet cache.

    The parquet cache (built by running this file directly, e.g.
    `python dashboard_data.py`) is committed to git in <45MB chunks
    (`combined_part*.parquet`) for fast loading and GitHub limits.
    """
    parquet_files = sorted(DATA_DIR.glob("combined*.parquet"))
    if parquet_files:
        frames = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(frames, ignore_index=True)
        df["month"] = pd.to_datetime(df["month"])
    else:
        df = _read_raw_data()

    # Clean any remaining NoData sentinels (< -9000) to NaN across all variables
    for col in VARIABLES:
        if col in df.columns:
            df.loc[df[col] < -9000, col] = np.nan

    # Merge county mapping
    if COUNTY_MAPPING.exists():
        mapping_df = pd.read_csv(COUNTY_MAPPING)
        
        # Cast to float32 to ensure exact match with parquet/tiff columns
        df['lat_match'] = df['lat'].astype('float32')
        df['lon_match'] = df['lon'].astype('float32')
        mapping_df['lat_match'] = mapping_df['lat'].astype('float32')
        mapping_df['lon_match'] = mapping_df['lon'].astype('float32')
        
        df = df.merge(
            mapping_df[['lat_match', 'lon_match', 'county']], 
            on=['lat_match', 'lon_match'], 
            how='left'
        )
        df = df.drop(columns=['lat_match', 'lon_match'])
        df['county'] = df['county'].fillna('Unknown')
        
    return df


def county_monthly_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Return a lightweight dataframe grouped by month and county."""
    if 'county' not in df.columns:
        return df
    available_vars = [v for v in VARIABLES.keys() if v in df.columns]
    return df.groupby(["month", "county"])[available_vars].mean().reset_index()


def build_parquet_cache(num_chunks: int = 4) -> list[Path]:
    """Rebuild the compact parquet cache directly from all monthly GeoTIFF rasters or CSVs.

    Sorts by (lat, lon, month) and splits into <45MB ZSTD-compressed chunk files
    (`combined_part01.parquet` .. `combined_part04.parquet`) so each file easily complies
    with GitHub's 50MB warning / 100MB hard limit.
    """
    df = _read_raw_data()
    for col in VARIABLES:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    df["lat"] = df["lat"].astype("float32")
    df["lon"] = df["lon"].astype("float32")
    
    # Sort for optimal compression & layout
    df = df.sort_values(by=["lat", "lon", "month"]).reset_index(drop=True)
    
    # Remove existing chunk or monolithic parquet files
    for old_p in DATA_DIR.glob("combined*.parquet"):
        try:
            old_p.unlink()
        except Exception:
            pass

    years = sorted(df["month"].dt.year.unique())
    chunk_years = np.array_split(years, num_chunks)
    
    saved_paths = []
    for idx, yrs in enumerate(chunk_years, 1):
        sub_df = df[df["month"].dt.year.isin(yrs)].copy()
        chunk_path = DATA_DIR / f"combined_part{idx:02d}.parquet"
        sub_df.to_parquet(chunk_path, index=False, compression="zstd")
        size_mb = chunk_path.stat().st_size / 1024 / 1024
        print(f"Wrote {chunk_path.name} ({size_mb:.1f} MB, years {yrs[0]}-{yrs[-1]})")
        saved_paths.append(chunk_path)
        
    return saved_paths


if __name__ == "__main__":
    paths = build_parquet_cache()
    total_size = sum(p.stat().st_size for p in paths) / 1024 / 1024
    print(f"Done! Wrote {len(paths)} chunked parquet files totaling {total_size:.1f} MB.")

