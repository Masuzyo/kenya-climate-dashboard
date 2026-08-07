"""Monthly Kenya climate & vegetation raster ingest, sourced from Google Earth Engine.

DATA SOURCES
------------
| Variable                              | EE collection                  | Band(s) used                                                          | Native resolution        | Native temporal coverage (rolling)          |
|----------------------------------------|---------------------------------|------------------------------------------------------------------------|---------------------------|----------------------------------------------|
| max_temp_c                             | MODIS/061/MOD11A2               | LST_Day_1km                                                             | 1 km, 8-day composite     | 2000-02-18 -> ~1-2 months behind today       |
| ndvi, veg_cover_pct                     | MODIS/061/MOD13Q1               | NDVI                                                                    | 250 m, 16-day composite   | 2000-02-18 -> ~1-2 months behind today       |
| rain_mm                                 | UCSB-CHG/CHIRPS/DAILY           | precipitation                                                           | ~5.5 km (0.05 deg), daily | 1981-01-01 -> ~1-2 months behind today       |
| humidity_rh_pct, soil_moisture_m3m3     | ECMWF/ERA5_LAND/MONTHLY_AGGR    | temperature_2m, dewpoint_temperature_2m, volumetric_soil_water_layer_1  | ~11 km (0.1 deg), monthly | 1950-02-01 -> ~2-3 months behind today       |

Google's rolling end dates shift forward roughly monthly as each provider
publishes; ERA5-Land is consistently the slowest to update, so it is the
binding constraint on how recent a month can be requested (see
`monthly_stack`'s explicit RuntimeError when a month isn't published yet).

PER-MONTH AGGREGATION
----------------------
- max_temp_c: MOD11A2 8-day LST composites overlapping the calendar month are
  converted Kelvin -> Celsius (`value * 0.02 - 273.15`), then combined with a
  per-pixel MAX across the ~3-4 composites/month that overlap (a composite is
  included if any of its 8 days falls in the month, so edge composites can
  pull in a day or two from the neighboring month).
- ndvi: MOD13Q1 16-day NDVI composites overlapping the month, scaled by
  0.0001, combined with a per-pixel MEAN across the ~2 composites/month.
- veg_cover_pct: derived from the monthly `ndvi` above via a standard linear
  NDVI-to-fractional-vegetation-cover proxy (not a native MODIS product):
  `fCover = clamp((NDVI - 0.2) / (0.8 - 0.2), 0, 1) * 100`. NDVI<=0.2 reads as
  0% (bare soil/water); NDVI>=0.8 reads as 100% (full canopy).
- rain_mm: CHIRPS daily precipitation, SUMMED over every day in the calendar
  month -> total monthly rainfall in mm.
- humidity_rh_pct: derived (not a direct ERA5-Land band) from ERA5-Land
  MONTHLY_AGGR's monthly-mean 2 m air temperature and 2 m dewpoint
  temperature via the Tetens saturation-vapor-pressure formula:
  `es(T) = 0.6108 * exp(17.27*T / (T+237.3))`, `RH% = 100 * es(Td) / es(T)`.
  This approximates monthly-mean relative humidity from monthly-mean
  temperatures; it is not derived from sub-monthly RH readings.
- soil_moisture_m3m3: ERA5-Land MONTHLY_AGGR `volumetric_soil_water_layer_1`
  (0-7 cm depth) used as-is; it is already a monthly mean in m3/m3.

All six bands are combined, clipped to Kenya's boundary
(USDOS/LSIB_SIMPLE/2017), then `.unmask(NODATA_SENTINEL, sameFootprint=False)`
so pixels with no valid data (outside Kenya, or missing from a source that
month) are explicitly flagged rather than silently defaulting to 0.

ACHIEVABLE TIME RANGE & RESOLUTION
------------------------------------
- Longest span for ALL SIX variables together: ~2000-03 (first full calendar
  month after MODIS Terra data begins 2000-02-18) through ~2-3 months behind
  the current date (bounded by ERA5-Land MONTHLY_AGGR's publication lag).
  Rainfall alone (CHIRPS) could go back to 1981-01; MODIS-only variables
  (max_temp_c, ndvi, veg_cover_pct) could start from 2000-02 if humidity/soil
  moisture aren't needed.
- Finest native compositing period per source: MOD11A2 8-day, MOD13Q1 16-day,
  CHIRPS 1-day, ERA5-Land hourly natively (but MONTHLY_AGGR only exposes a
  monthly product). This script aggregates everything to MONTHLY. If a finer
  common cadence were wanted across all six variables at once, MOD13Q1's
  16-day cycle is the limiting factor (NDVI/veg_cover_pct cannot go finer
  without interpolation). Individually, max_temp_c could go to 8-day, rain_mm
  to daily, and humidity/soil_moisture to hourly or daily (by switching to
  ECMWF/ERA5_LAND/HOURLY), but there is no shared cadence finer than 16 days
  across the full variable set.
"""

import argparse
import csv
import datetime as dt
from pathlib import Path

import ee
import numpy as np
import requests
import rasterio


KENYA: ee.Geometry | None = None

# Sentinel written for masked/missing pixels (e.g. mixed-resolution border
# edge effects) so they're distinguishable from genuine 0 readings.
NODATA_SENTINEL = -9999.0


def kenya_geometry() -> ee.Geometry:
    """Lazily build the Kenya boundary geometry (requires ee.Initialize() first)."""
    global KENYA
    if KENYA is None:
        KENYA = (
            ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
            .filter(ee.Filter.eq("country_na", "Kenya"))
            .geometry()
        )
    return KENYA


def month_starts(start_yyyy_mm: str, end_yyyy_mm: str) -> list[dt.date]:
    start = dt.datetime.strptime(start_yyyy_mm, "%Y-%m").date().replace(day=1)
    end = dt.datetime.strptime(end_yyyy_mm, "%Y-%m").date().replace(day=1)
    if end < start:
        raise ValueError("end must be >= start (format: YYYY-MM)")

    months: list[dt.date] = []
    cur = start
    while cur <= end:
        months.append(cur)
        year = cur.year + (1 if cur.month == 12 else 0)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = dt.date(year, month, 1)
    return months


def saturation_vapor_pressure_celsius(temp_c: ee.Image) -> ee.Image:
    # Tetens formula in kPa.
    return temp_c.expression(
        "0.6108 * exp((17.27 * t) / (t + 237.3))", {"t": temp_c}
    )


def monthly_stack(month_start: dt.date) -> ee.Image:
    month_end = (
        dt.date(month_start.year + (1 if month_start.month == 12 else 0), 1, 1)
        if month_start.month == 12
        else dt.date(month_start.year, month_start.month + 1, 1)
    )
    start = ee.Date(month_start.isoformat())
    end = ee.Date(month_end.isoformat())

    # MODIS day land-surface temperature (8-day, Kelvin * 0.02) -> Celsius.
    lst_c = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(start, end)
        .select("LST_Day_1km")
        .map(lambda img: img.multiply(0.02).subtract(273.15))
        .max()
        .rename("max_temp_c")
    )

    # MODIS NDVI (16-day, scale factor 0.0001).
    ndvi = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(start, end)
        .select("NDVI")
        .map(lambda img: img.multiply(0.0001))
        .mean()
        .rename("ndvi")
    )

    # Monthly vegetation coverage proxy: fractional vegetation cover from NDVI.
    veg_cover = (
        ndvi.subtract(0.2)
        .divide(0.8 - 0.2)
        .clamp(0, 1)
        .multiply(100)
        .rename("veg_cover_pct")
    )

    # Rainfall: CHIRPS daily precipitation summed monthly (mm/month).
    rain = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .select("precipitation")
        .sum()
        .rename("rain_mm")
    )

    # Humidity + soil moisture from ERA5-Land monthly aggregates.
    # NOTE: "ECMWF/ERA5_LAND/MONTHLY" is deprecated and stops at 2023-04-01;
    # "MONTHLY_AGGR" is the actively updated successor with the same band names.
    era5_collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(start, end)
    )
    era5_size = era5_collection.size().getInfo()
    if era5_size == 0:
        raise RuntimeError(
            f"No ERA5-Land monthly data available for "
            f"{month_start.strftime('%Y-%m')}. This dataset is only "
            "published a few months behind real-time; try an earlier month."
        )
    era5 = era5_collection.first()
    t2m_c = ee.Image(era5.select("temperature_2m")).subtract(273.15)
    d2m_c = ee.Image(era5.select("dewpoint_temperature_2m")).subtract(273.15)

    rh = (
        saturation_vapor_pressure_celsius(d2m_c)
        .divide(saturation_vapor_pressure_celsius(t2m_c))
        .multiply(100)
        .rename("humidity_rh_pct")
    )

    soil_moisture = ee.Image(era5.select("volumetric_soil_water_layer_1")).rename(
        "soil_moisture_m3m3"
    )

    return (
        lst_c.addBands(ndvi)
        .addBands(veg_cover)
        .addBands(rain)
        .addBands(rh)
        .addBands(soil_moisture)
        .clip(kenya_geometry())
        # Explicitly flag masked/missing pixels (outside Kenya's polygon within
        # the export's bounding rectangle, or border edge effects between
        # mixed-resolution sources) instead of letting the GeoTIFF export burn
        # them in as a silent 0. sameFootprint=False fills the *entire*
        # exported rectangle, not just pixels within the clipped footprint.
        .unmask(NODATA_SENTINEL, sameFootprint=False)
        .set("month", month_start.strftime("%Y-%m"))
    )


def download_geotiff(image: ee.Image, out_file: Path, scale_m: int) -> None:
    url = image.getDownloadURL(
        {
            "name": out_file.stem,
            "region": kenya_geometry(),
            "scale": scale_m,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        }
    )

    response = requests.get(url, timeout=300)
    response.raise_for_status()
    out_file.write_bytes(response.content)

    # GEE's GeoTIFF export doesn't tag a NoData value, so pixels we unmasked
    # to NODATA_SENTINEL would otherwise look like valid data. Tag it here.
    with rasterio.open(out_file, "r+") as dst:
        dst.nodata = NODATA_SENTINEL


def initialize_ee(project: str | None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def geotiff_to_csv(tif_path: Path, csv_path: Path, month: str) -> None:
    """Extract GeoTIFF pixels to CSV: month, lat, lon, and one column per band."""
    with rasterio.open(tif_path) as src:
        data = src.read(masked=True)  # masked array: NoData pixels become masked
        transform = src.transform

        # Prefer band descriptions written by GEE; fall back to positional names.
        default_names = [
            "max_temp_c",
            "ndvi",
            "veg_cover_pct",
            "rain_mm",
            "humidity_rh_pct",
            "soil_moisture_m3m3",
        ]
        band_names = [
            desc if desc else default_names[i]
            for i, desc in enumerate(src.descriptions)
        ]

        rows, cols = data.shape[1], data.shape[2]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["month", "lat", "lon"] + band_names)

            for row in range(rows):
                for col in range(cols):
                    pixel = data[:, row, col]
                    if np.ma.is_masked(pixel) and pixel.mask.any():
                        continue  # skip pixels with any NoData band

                    lon, lat = transform * (col + 0.5, row + 0.5)
                    values = [f"{v:.4f}" for v in pixel.filled(np.nan)]
                    writer.writerow([month, f"{lat:.6f}", f"{lon:.6f}"] + values)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download monthly Kenya climate/vegetation rasters."
    )
    parser.add_argument("--start", required=True, help="Start month (YYYY-MM).")
    parser.add_argument("--end", required=True, help="End month (YYYY-MM).")
    parser.add_argument(
        "--output-dir",
        default="kenya_monthly_ingest",
        help="Folder for GeoTIFF outputs.",
    )
    parser.add_argument(
        "--scale-m",
        type=int,
        default=5000,
        help="Output pixel size in meters (default: 5000).",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional Google Earth Engine cloud project id.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export a CSV file (pixel-level table) alongside each GeoTIFF.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initialize_ee(args.project)
    months = month_starts(args.start, args.end)

    failed_months: list[str] = []
    for month in months:
        month_label = month.strftime("%Y-%m")
        try:
            image = monthly_stack(month)
            out_file = output_dir / f"kenya_{month.strftime('%Y_%m')}.tif"
            print(f"Downloading {month_label} -> {out_file}")
            download_geotiff(image, out_file, args.scale_m)

            # Convert to CSV if requested
            if args.csv:
                csv_file = output_dir / f"kenya_{month.strftime('%Y_%m')}.csv"
                print(f"Converting to CSV -> {csv_file}")
                geotiff_to_csv(out_file, csv_file, month_label)
        except Exception as exc:
            print(f"WARNING: skipped {month_label}: {exc}")
            failed_months.append(month_label)

    print(f"Done. Files saved in: {output_dir.resolve()}")
    if args.csv:
        print(f"CSV files also generated in: {output_dir.resolve()}")
    if failed_months:
        print(f"Skipped {len(failed_months)} month(s): {', '.join(failed_months)}")


if __name__ == "__main__":
    main()
