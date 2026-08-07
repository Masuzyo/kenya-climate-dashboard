# Kenya Monthly Climate & Vegetation Dashboard

Interactive Streamlit dashboard visualizing monthly climate and vegetation
data over Kenya at 5 km resolution: max temperature, NDVI (vegetation
greenness), vegetation coverage, rainfall, humidity, and soil moisture.

**Live dashboard:** _add your deployed Streamlit Community Cloud URL here_

## Data sources

All data is pulled from Google Earth Engine. See the module docstring in
[`modis_kenya_monthly_ingest.py`](modis_kenya_monthly_ingest.py) for full
details on data sources, native resolution/coverage, and the exact
per-variable aggregation formulas (MODIS LST/NDVI, CHIRPS rainfall,
ERA5-Land humidity/soil moisture).

## Repo layout

- `modis_kenya_monthly_ingest.py` — pulls monthly rasters + CSVs from Earth
  Engine into `kenya_monthly_ingest/` (requires a Google Cloud project with
  the Earth Engine API enabled; **not** needed to run the dashboard itself).
- `dashboard_data.py` — loads the combined dataset. Prefers the compact
  `kenya_monthly_ingest/combined.parquet` cache (committed to git) over the
  raw per-month CSVs (gitignored — regenerate locally as needed).
- `dashboard.py` — the Streamlit app (map, variable/month controls, trends).

## Running locally

```powershell
pip install -r requirements.txt
streamlit run dashboard.py
```

## Refreshing the data

```powershell
# 1. Pull new months from Earth Engine (requires GEE auth + a GCP project)
python modis_kenya_monthly_ingest.py --start 2026-02 --end 2026-02 --csv --project YOUR_PROJECT_ID

# 2. Rebuild the compact parquet cache the dashboard actually reads
python dashboard_data.py

# 3. Commit the updated cache
git add kenya_monthly_ingest/combined.parquet
git commit -m "Update dashboard data"
git push
```

Streamlit Community Cloud automatically redeploys on every push to the
connected branch.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public, or private with a Streamlit account
   linked to your GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**.
3. Select this repo/branch and set the main file path to `dashboard.py`.
4. Deploy — you'll get a permanent URL like
   `https://<name>-<hash>.streamlit.app`.
