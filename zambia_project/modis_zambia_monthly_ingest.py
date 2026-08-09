"""Monthly Zambia climate & vegetation raster ingest, sourced from Google Earth Engine.

DATA SOURCES
------------
| Variable                                          | EE collection                  | Band(s) used                                                          | Native resolution        | Native temporal coverage (rolling)          |
|-----------------------------------------------------|---------------------------------|------------------------------------------------------------------------|---------------------------|----------------------------------------------|
| max_temp_c, min_temp_c, mean_temp_c, dtr_c          | MODIS/061/MOD11A2               | LST_Day_1km, LST_Night_1km                                              | 1 km, 8-day composite     | 2000-02-18 -> ~1-2 months behind today       |
| elevation_m                                         | USGS/SRTMGL1_003                | elevation                                                               | 30 m, static (no time dim)| n/a (single fixed snapshot, ~2000)           |
| ndvi, veg_cover_pct, evi                            | MODIS/061/MOD13Q1               | NDVI, EVI                                                               | 250 m, 16-day composite   | 2000-02-18 -> ~1-2 months behind today       |
| forest, savanna, wetland, cropland, urban_pct       | MODIS/061/MCD12Q1               | LC_Type1                                                                | 500 m, annual composite   | 2000-01-01 -> 2023-12-31 (latest used for now)|
| surface_water_occurrence_pct                        | JRC/GSW1_4/GlobalSurfaceWater   | occurrence                                                              | 30 m, static              | n/a (static occurrence probability)          |
| rain_mm                                             | UCSB-CHG/CHIRPS/DAILY           | precipitation                                                           | ~5.5 km (0.05 deg), daily | 1981-01-01 -> ~1-2 months behind today       |
| humidity_rh_pct, soil_moisture_m3m3, wind_speed_ms  | ECMWF/ERA5_LAND/MONTHLY_AGGR    | temperature_2m, dewpoint_temperature_2m, volumetric_soil_water_layer_1, u/v_component_of_wind_10m  | ~11 km (0.1 deg), monthly | 1950-02-01 -> ~2-3 months behind today       |

Google's rolling end dates shift forward roughly monthly as each provider
publishes; ERA5-Land is consistently the slowest to update, so it is the
binding constraint on how recent a month can be requested (see
`monthly_stack`'s explicit RuntimeError when a month isn't published yet).

PER-MONTH AGGREGATION
----------------------
- max_temp_c: MOD11A2 8-day LST_Day_1km composites overlapping the calendar
  month are converted Kelvin -> Celsius (`value * 0.02 - 273.15`), then
  combined with a per-pixel MAX across the ~3-4 composites/month that overlap
  (a composite is included if any of its 8 days falls in the month, so edge
  composites can pull in a day or two from the neighboring month).
- min_temp_c: same conversion applied to LST_Night_1km, combined with a
  per-pixel MIN across the month's composites (coldest night of the month).
  Night temperature is used as the monthly minimum since it's typically the
  daily low; relevant because cold nights can halt mosquito survival /
  parasite development even when daytime max temp looks warm enough.
- mean_temp_c: average of each composite's day and night reading
  (`(day + night) / 2`), then averaged again (MEAN) across the composites
  overlapping the month. Not a native MODIS band.
- dtr_c: diurnal temperature range, derived as `max_temp_c - min_temp_c`.
  Large DTR (> 15 °C) significantly slows Plasmodium parasite development
  and reduces vector competence, even when mean temperature is in the
  optimal 25-28 °C range. Highland Kenya has characteristically large DTR.
- elevation_m: SRTM 30 m DEM, used as-is (static -- no time filtering). The
  same per-pixel value is included in every month's export for convenience.
  Included because East African highland malaria transmission is strongly
  bounded by altitude (largely via its effect on temperature).
- ndvi: MOD13Q1 16-day NDVI composites overlapping the month, scaled by
  0.0001, combined with a per-pixel MEAN across the ~2 composites/month.
- evi: MOD13Q1 16-day EVI composites overlapping the month, scaled by
  0.0001, combined with a per-pixel MEAN. EVI is more sensitive than NDVI in
  dense canopy (where NDVI saturates) and corrects for atmospheric and
  soil-background effects; canopy structure affects mosquito resting
  behaviour and microclimate.
- veg_cover_pct: derived from the monthly `ndvi` above via a standard linear
  NDVI-to-fractional-vegetation-cover proxy (not a native MODIS product):
  `fCover = clamp((NDVI - 0.2) / (0.8 - 0.2), 0, 1) * 100`. NDVI<=0.2 reads as
  0% (bare soil/water); NDVI>=0.8 reads as 100% (full canopy).
- forest_pct, savanna_pct, wetland_pct, cropland_pct, urban_pct: 
  MODIS MCD12Q1 annual land cover, extracting fractions of IGBP classes
  per export pixel. Since it's annual and ends in 2023, the most recent
  available map prior to or during the current month is used.
- surface_water_occurrence_pct: JRC Global Surface Water static occurrence
  band (0-100%). Represents the frequency of water presence. Unmasked to 0
  where water never occurs, averaged to the export pixel scale.
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
- wind_speed_ms: derived from ERA5-Land MONTHLY_AGGR 10 m u- and v-wind
  components as `sqrt(u² + v²)`. Wind speeds > 1.5-2 m/s significantly
  reduce Anopheles host-seeking; regions with persistent trade winds
  (coastal Kenya, Turkana corridor) may have lower transmission.

All eighteen bands are combined, clipped to Kenya's boundary
(USDOS/LSIB_SIMPLE/2017), then `.unmask(NODATA_SENTINEL, sameFootprint=False)`
so pixels with no valid data (outside Kenya, or missing from a source that
month) are explicitly flagged rather than silently defaulting to 0.

ACHIEVABLE TIME RANGE & RESOLUTION
------------------------------------
- Longest span for ALL variables together: ~2000-03 (first full calendar
  month after MODIS Terra data begins 2000-02-18) through ~2-3 months behind
  the current date (bounded by ERA5-Land MONTHLY_AGGR's publication lag).
  Rainfall alone (CHIRPS) could go back to 1981-01; MODIS-only variables
  (max_temp_c, min_temp_c, mean_temp_c, ndvi, veg_cover_pct) could start from
  2000-02 if humidity/soil moisture aren't needed. elevation_m has no
  temporal dimension (it's a single static snapshot from ~2000) so it never
  constrains the range.
- Finest native compositing period per source: MOD11A2 8-day, MOD13Q1 16-day,
  CHIRPS 1-day, ERA5-Land hourly natively (but MONTHLY_AGGR only exposes a
  monthly product). This script aggregates everything to MONTHLY. If a finer
  common cadence were wanted across all variables at once, MOD13Q1's 16-day
  cycle is the limiting factor (NDVI/veg_cover_pct cannot go finer without
  interpolation). Individually, the temperature bands could go to 8-day,
  rain_mm to daily, and humidity/soil_moisture to hourly or daily (by
  switching to ECMWF/ERA5_LAND/HOURLY), but there is no shared cadence finer
  than 16 days across the full variable set.
"""

import argparse
import csv
import datetime as dt
from pathlib import Path

import ee
import numpy as np
import requests
import rasterio


ZAMBIA: ee.Geometry | None = None

# Sentinel written for masked/missing pixels (e.g. mixed-resolution border
# edge effects) so they're distinguishable from genuine 0 readings.
NODATA_SENTINEL = -9999.0


def zambia_geometry() -> ee.Geometry:
    """Lazily build the Zambia boundary geometry (requires ee.Initialize() first)."""
    global ZAMBIA
    if ZAMBIA is None:
        ZAMBIA = (
            ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
            .filter(ee.Filter.eq("country_na", "Zambia"))
            .geometry()
        )
    return ZAMBIA


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


def monthly_stack(month_start: dt.date, scale_m: int = 5000) -> ee.Image:
    month_end = (
        dt.date(month_start.year + (1 if month_start.month == 12 else 0), 1, 1)
        if month_start.month == 12
        else dt.date(month_start.year, month_start.month + 1, 1)
    )
    start = ee.Date(month_start.isoformat())
    end = ee.Date(month_end.isoformat())

    # MODIS day land-surface temperature (8-day, Kelvin * 0.02) -> Celsius.
    lst_day_c_composites = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(start, end)
        .select("LST_Day_1km")
        .map(lambda img: img.multiply(0.02).subtract(273.15))
    )
    max_temp_c = lst_day_c_composites.max().rename("max_temp_c")

    # Diurnal temperature range: difference between monthly max daytime and
    # min nighttime LST.  Large DTR (>15 °C) slows Plasmodium development
    # and reduces vector competence even at favourable mean temperatures.

    # MODIS night land-surface temperature, same scaling. Used as a monthly
    # MIN (coldest night of the month) since night temperature is usually the
    # daily minimum, and cold nights can halt transmission of temperature-
    # sensitive processes (e.g. mosquito survival, parasite development) even
    # when daytime max temperature looks warm enough.
    lst_night_c_composites = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(start, end)
        .select("LST_Night_1km")
        .map(lambda img: img.multiply(0.02).subtract(273.15))
    )
    min_temp_c = lst_night_c_composites.min().rename("min_temp_c")

    # Mean temperature: average of each composite's day/night reading, then
    # averaged again across the composites overlapping the month.
    mean_temp_c = (
        lst_day_c_composites.mean()
        .add(lst_night_c_composites.mean())
        .divide(2)
        .rename("mean_temp_c")
    )

    # Build DTR after both LST aggregates are available.
    dtr_c = max_temp_c.subtract(min_temp_c).rename("dtr_c")

    # Elevation (SRTM 30m, static/no time dimension). Included as a per-pixel
    # constant in every monthly export for convenience; highland malaria in
    # East Africa is strongly bounded by altitude via its effect on
    # temperature, so this is a key covariate even though it never changes.
    elevation_m = (
        ee.Image("USGS/SRTMGL1_003")
        .select("elevation")
        .toFloat()  # native int16 can't represent the NaN NoData fill used downstream
        .rename("elevation_m")
    )

    # MODIS NDVI (16-day, scale factor 0.0001).
    mod13_collection = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(start, end)
    )
    ndvi = (
        mod13_collection
        .select("NDVI")
        .map(lambda img: img.multiply(0.0001))
        .mean()
        .rename("ndvi")
    )

    # EVI: Enhanced Vegetation Index from the same MOD13Q1 collection.
    # More sensitive than NDVI in dense canopy (where NDVI saturates) and
    # corrects for atmospheric/soil-background effects; canopy structure
    # affects mosquito resting behaviour and microclimate.
    evi = (
        mod13_collection
        .select("EVI")
        .map(lambda img: img.multiply(0.0001))
        .mean()
        .rename("evi")
    )

    # Monthly vegetation coverage proxy: fractional vegetation cover from NDVI.
    veg_cover = (
        ndvi.subtract(0.2)
        .divide(0.8 - 0.2)
        .clamp(0, 1)
        .multiply(100)
        .rename("veg_cover_pct")
    )

    # Land Cover fractions (MODIS MCD12Q1, 500 m, annual)
    # The collection ends in 2023. We get the latest available map prior to or during the current month.
    lc = (
        ee.ImageCollection("MODIS/061/MCD12Q1")
        .filterDate("2000-01-01", end)
        .sort("system:time_start", False)
        .first()
        .select("LC_Type1")
    )

    forest = lc.gte(1).And(lc.lte(5))
    savanna = lc.gte(8).And(lc.lte(9))
    wetland = lc.eq(11)
    cropland = lc.eq(12).Or(lc.eq(14))
    urban = lc.eq(13)

    lc_fractions = (
        ee.Image([forest, savanna, wetland, cropland, urban])
        .rename(["forest_pct", "savanna_pct", "wetland_pct", "cropland_pct", "urban_pct"])
        .multiply(100)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs="EPSG:4326", scale=scale_m)
    )

    # Surface Water (JRC Global Surface Water, 30 m, static)
    # Using 'occurrence' band (0-100%) as a static map of water pooling propensity.
    surface_water = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .unmask(0)
        .rename("surface_water_occurrence_pct")
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

    # Wind speed: vector magnitude of 10 m u- and v-wind components.
    # Wind > 1.5-2 m/s significantly reduces Anopheles host-seeking;
    # persistent trade winds (coastal Kenya, Turkana corridor) may lower
    # transmission beyond what temperature/humidity alone would predict.
    u_wind = ee.Image(era5.select("u_component_of_wind_10m"))
    v_wind = ee.Image(era5.select("v_component_of_wind_10m"))
    wind_speed = u_wind.pow(2).add(v_wind.pow(2)).sqrt().rename("wind_speed_ms")

    return (
        max_temp_c.addBands(min_temp_c)
        .addBands(mean_temp_c)
        .addBands(dtr_c)
        .addBands(elevation_m)
        .addBands(ndvi)
        .addBands(evi)
        .addBands(veg_cover)
        .addBands(lc_fractions)
        .addBands(surface_water)
        .addBands(rain)
        .addBands(rh)
        .addBands(soil_moisture)
        .addBands(wind_speed)
        .clip(zambia_geometry())
        # Explicitly flag masked/missing pixels (outside Zambia's polygon within
        # the export's bounding rectangle, or border edge effects between
        # mixed-resolution sources) instead of letting the GeoTIFF export burn
        # them in as a silent 0. sameFootprint=False fills the *entire*
        # exported rectangle, not just pixels within the clipped footprint.
        .unmask(NODATA_SENTINEL, sameFootprint=False)
        .set("month", month_start.strftime("%Y-%m"))
    )


def download_geotiff(image: ee.Image, out_path: Path, scale_m: int = 5000) -> None:
    """Download an Earth Engine Image as a GeoTIFF covering Zambia."""
    url = image.getDownloadURL(
        {
            "name": out_path.stem,
            "region": zambia_geometry().bounds().getInfo()["coordinates"],
            "scale": scale_m,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        }
    )

    response = requests.get(url, timeout=300)
    response.raise_for_status()
    out_path.write_bytes(response.content)

    # GEE's GeoTIFF export doesn't tag a NoData value, so pixels we unmasked
    # to NODATA_SENTINEL would otherwise look like valid data. Tag it here.
    with rasterio.open(out_path, "r+") as dst:
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
        # Must match the addBands() order in monthly_stack().
        default_names = [
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
        description="Download monthly Zambia climate/vegetation rasters."
    )
    parser.add_argument("--start", required=True, help="Start month (YYYY-MM).")
    parser.add_argument("--end", required=True, help="End month (YYYY-MM).")
    parser.add_argument(
        "--output-dir",
        default="zambia_monthly_ingest",
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
            image = monthly_stack(month, args.scale_m)
            out_file = output_dir / f"zambia_{month.strftime('%Y_%m')}.tif"
            print(f"Downloading {month_label} -> {out_file}")
            download_geotiff(image, out_file, args.scale_m)

            # Convert to CSV if requested
            if args.csv:
                csv_file = output_dir / f"zambia_{month.strftime('%Y_%m')}.csv"
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
