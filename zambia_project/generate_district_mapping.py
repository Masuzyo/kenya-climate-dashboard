import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import urllib.request
import json
import zipfile
import os
from pathlib import Path

def generate_mapping():
    print("Fetching Zambia ADM2 (Districts) from geoBoundaries...")
    # geoBoundaries API for Zambia ADM2
    url = "https://www.geoboundaries.org/api/current/gbOpen/ZMB/ADM2/"
    
    req = urllib.request.urlopen(url)
    data = json.loads(req.read())
    geojson_url = data['gjDownloadURL']
    
    print(f"Downloading GeoJSON from {geojson_url}...")
    districts = gpd.read_file(geojson_url)
    
    # Save geojson for the dashboard
    out_dir = Path("zambia_monthly_ingest")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    districts.to_file(out_dir / "districts.geojson", driver="GeoJSON")
    print(f"Saved {out_dir / 'districts.geojson'}")
    
    # Now reverse geocode the pixels
    # Find any existing CSV to get the master list of lat/lons
    csv_files = list(out_dir.glob("zambia_*.csv"))
    if not csv_files:
        print("No CSV files found. Please run the ingest script first.")
        return
        
    print(f"Reading lat/lon points from {csv_files[0].name}...")
    df = pd.read_csv(csv_files[0])
    points = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf_points = gpd.GeoDataFrame(df[['lat', 'lon']], geometry=points, crs="EPSG:4326")
    
    print("Performing spatial join (reverse geocoding)... This may take a minute.")
    joined = gpd.sjoin(gdf_points, districts, how="left", predicate="within")
    
    # "shapeName" is standard for geoBoundaries
    joined = joined.rename(columns={"shapeName": "district"})
    
    mapping_df = joined[['lat', 'lon', 'district']].copy()
    mapping_df['district'] = mapping_df['district'].fillna('Unknown')
    
    out_csv = out_dir / "zambia_district_mapping.csv"
    mapping_df.to_csv(out_csv, index=False)
    print(f"Saved district mapping to {out_csv}")
    print("Done! You can now run the dashboard.")

if __name__ == "__main__":
    generate_mapping()
