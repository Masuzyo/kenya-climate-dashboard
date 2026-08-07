"""Interactive dashboard: monthly Kenya climate & vegetation data on a map.

Run with:
    streamlit run dashboard.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_data import VARIABLES, load_combined

st.set_page_config(
    page_title="Kenya Climate & Vegetation Dashboard",
    layout="wide",
)

st.title("\U0001f1f0\U0001f1ea Kenya Monthly Climate & Vegetation Dashboard")
st.caption(
    "Sources: MODIS (max temperature, NDVI), CHIRPS (rainfall), "
    "ERA5-Land (humidity, soil moisture) \u2014 5 km monthly grid."
)

with st.sidebar:
    st.header("Controls")
    if st.button("\U0001f504 Reload data from disk"):
        st.cache_data.clear()

load_combined_cached = st.cache_data(show_spinner="Loading Kenya monthly dataset...")(
    load_combined
)
df = load_combined_cached()

months = sorted(df["month"].unique())
month_labels = [pd.Timestamp(m).strftime("%Y-%m") for m in months]

with st.sidebar:
    variable = st.selectbox(
        "Variable",
        options=list(VARIABLES.keys()),
        format_func=lambda v: (
            f"{VARIABLES[v]['label']} ({VARIABLES[v]['unit']})"
            if VARIABLES[v]["unit"]
            else VARIABLES[v]["label"]
        ),
    )
    month_idx = st.select_slider(
        "Month",
        options=list(range(len(months))),
        value=len(months) - 1,
        format_func=lambda i: month_labels[i],
    )
    map_style = st.radio(
        "Map style",
        ["Pixel grid (accurate)", "Smoothed density (approximate)"],
        index=0,
        help=(
            "Pixel grid shows the true value at each 5 km cell. Smoothed "
            "density applies a spatial kernel and can visually blur real "
            "variation when points are as dense as this dataset."
        ),
    )
    st.markdown("---")
    st.caption(f"Dataset covers {month_labels[0]} \u2192 {month_labels[-1]} "
               f"({len(months)} months, {len(df):,} pixel-months).")

selected_month = months[month_idx]
meta = VARIABLES[variable]
month_df = df[df["month"] == selected_month]

# --- KPI row -----------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean", f"{month_df[variable].mean():.2f} {meta['unit']}".strip())
col2.metric("Min", f"{month_df[variable].min():.2f} {meta['unit']}".strip())
col3.metric("Max", f"{month_df[variable].max():.2f} {meta['unit']}".strip())
col4.metric("Valid pixels", f"{len(month_df):,}")

# --- Map -----------------------------------------------------------------
st.subheader(f"{meta['label']} \u2014 {month_labels[month_idx]}")

map_kwargs = dict(
    lat="lat",
    lon="lon",
    color_continuous_scale=meta["colorscale"],
    center=dict(lat=0.5, lon=37.9),
    zoom=5.2,
    map_style="open-street-map",
    height=650,
    labels={variable: f"{meta['label']} ({meta['unit']})".strip()},
)

if map_style.startswith("Smoothed"):
    fig = px.density_map(month_df, z=variable, radius=12, **map_kwargs)
else:
    fig = px.scatter_map(month_df, color=variable, **map_kwargs)
    fig.update_traces(marker=dict(size=7, opacity=0.9))

fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")

# --- National monthly trend ----------------------------------------------
st.subheader("National monthly trend")
summary = df.groupby("month")[variable].mean().reset_index()
trend_fig = px.line(
    summary,
    x="month",
    y=variable,
    labels={variable: meta["label"], "month": "Month"},
)
trend_fig.add_vline(x=selected_month, line_dash="dash", line_color="red")
trend_fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(trend_fig, width="stretch")

# --- Raw data / export -----------------------------------------------------
with st.expander("Raw data (selected month)"):
    st.dataframe(month_df.head(1000), width="stretch")
    st.download_button(
        "Download this month's full CSV",
        month_df.to_csv(index=False),
        file_name=f"kenya_{month_labels[month_idx]}_{variable}.csv",
        mime="text/csv",
    )
