"""One-time backfill: add elevation_m to all existing monthly CSVs.

Elevation (SRTM) is static -- it doesn't vary by month -- so unlike the
temperature backfill, this only needs a single Earth Engine call. The
resulting (lat, lon) -> elevation_m lookup is merged into every existing
monthly CSV in kenya_monthly_ingest/, then the combined parquet cache is
rebuilt.

Run once with:
    python backfill_elevation.py --project YOUR_GCP_PROJECT_ID
"""

import argparse
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio

import modis_kenya_monthly_ingest as ingest
import dashboard_data

DATA_DIR = Path(__file__).parent / "kenya_monthly_ingest"


def fetch_elevation_lookup(project: str | None, scale_m: int = 5000) -> pd.DataFrame:
    ingest.initialize_ee(project)
    elevation = (
        ee.Image("USGS/SRTMGL1_003")
        .select("elevation")
        .toFloat()  # native int16 can't represent the NaN fill used downstream
        .rename("elevation_m")
        .clip(ingest.kenya_geometry())
        .unmask(ingest.NODATA_SENTINEL, sameFootprint=False)
    )

    tmp_tif = DATA_DIR / "_elevation_lookup.tif"
    ingest.download_geotiff(elevation, tmp_tif, scale_m)

    # This is a single-band file, so read it directly with rasterio rather
    # than ingest.geotiff_to_csv (which assumes the full 9-band monthly layout).
    with rasterio.open(tmp_tif) as src:
        data = src.read(1, masked=True)
        transform = src.transform
        rows, cols = data.shape
        records = []
        for row in range(rows):
            for col in range(cols):
                val = data[row, col]
                if np.ma.is_masked(val):
                    continue
                lon, lat = transform * (col + 0.5, row + 0.5)
                records.append((round(lat, 6), round(lon, 6), float(val)))
    df = pd.DataFrame(records, columns=["lat", "lon", "elevation_m"])

    tmp_tif.unlink(missing_ok=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None)
    parser.add_argument("--scale-m", type=int, default=5000)
    args = parser.parse_args()

    print("Fetching elevation lookup from Earth Engine...")
    elevation_lookup = fetch_elevation_lookup(args.project, args.scale_m)
    print(f"Got {len(elevation_lookup):,} elevation pixels.")

    files = sorted(DATA_DIR.glob("kenya_*.csv"))
    print(f"Backfilling elevation_m into {len(files)} monthly CSVs...")
    updated = 0
    for f in files:
        df = pd.read_csv(f)
        if "elevation_m" in df.columns:
            continue
        merged = df.merge(elevation_lookup, on=["lat", "lon"], how="left")
        merged.to_csv(f, index=False)
        updated += 1
        if updated % 50 == 0:
            print(f"  ...{updated}/{len(files)} files updated")

    print(f"Updated {updated} files (skipped {len(files) - updated} already done).")

    print("Rebuilding parquet cache...")
    path = dashboard_data.build_parquet_cache()
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"Wrote {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
