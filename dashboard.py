"""Interactive dashboard: monthly Kenya climate & vegetation data on a map.

Run with:
    streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import json
import pathlib
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV

from dashboard_data import PARQUET_CACHE, VARIABLES, load_combined, county_monthly_averages

st.set_page_config(
    page_title="Kenya Climate & Vegetation Dashboard",
    layout="wide",
)

st.title("Kenya Monthly Climate & Vegetation Dashboard")
st.caption("Sources: MODIS (temperature, NDVI, EVI, land cover), CHIRPS (rainfall), "
    "ERA5-Land (humidity, soil moisture, wind speed), JRC (surface water) — 5 km monthly GeoTIFF rasters."
)

with st.sidebar:
    st.header("Controls")
    if st.button("\U0001f504 Reload data from disk"):
        st.cache_data.clear()

load_combined_cached = st.cache_data(show_spinner="Loading Kenya monthly dataset...")(
    load_combined
)
df = load_combined_cached()

@st.cache_data(show_spinner="Loading boundaries...")
def load_geojson():
    geojson_path = pathlib.Path("kenya_monthly_ingest/counties.geojson")
    if geojson_path.exists():
        with open(geojson_path, "r") as f:
            return json.load(f)
    return None

counties_geojson = load_geojson()
county_df = county_monthly_averages(df)

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
    resolution = st.radio("Resolution", ["5 km Grid", "County Averages"])
    animate = st.checkbox("Enable Animation ⏵️")
    
    if animate:
        if resolution == "5 km Grid":
            years = sorted(list(set([pd.Timestamp(m).year for m in months])))
            selected_year = st.selectbox("Select Year to Animate", options=years, index=len(years)-1)
            anim_months = [m for m in months if pd.Timestamp(m).year == selected_year]
            month_idx = len(months) - 1
        else:
            anim_months = months
            month_idx = len(months) - 1
    else:
        month_idx = st.select_slider(
            "Month",
            options=list(range(len(months))),
            value=len(months) - 1,
            format_func=lambda i: month_labels[i],
        )
    st.markdown("---")
    st.caption(f"Dataset covers {month_labels[0]} \u2192 {month_labels[-1]} "
               f"({len(months)} months, {len(df):,} pixel-months).")

meta = VARIABLES[variable]
if animate:
    if resolution == "5 km Grid":
        month_df = df[df["month"].isin(anim_months)].copy()
        month_df['month_str'] = month_df['month'].dt.strftime("%Y-%m")
    else:
        county_month_df = county_df.copy()
        county_month_df['month_str'] = county_month_df['month'].dt.strftime("%Y-%m")
        # Ensure it's sorted by date for animation
        county_month_df = county_month_df.sort_values("month")
    selected_month = anim_months[-1] if resolution == "5 km Grid" else months[-1]
    
    # Ensure month_df is always defined for the raw data export at the bottom
    if resolution != "5 km Grid":
        month_df = df[df["month"] == selected_month]
else:
    selected_month = months[month_idx]
    month_df = df[df["month"] == selected_month]
    county_month_df = county_df[county_df["month"] == selected_month]

tab_climate, tab_malaria, tab_modeling = st.tabs(["Climate & Vegetation", "Malaria Resistance", "Statistical Modeling"])

with tab_climate:
    # --- KPI row -----------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    if resolution == "5 km Grid":
        col1.metric("Mean", f"{month_df[variable].mean():.2f} {meta['unit']}".strip())
        col2.metric("Min", f"{month_df[variable].min():.2f} {meta['unit']}".strip())
        col3.metric("Max", f"{month_df[variable].max():.2f} {meta['unit']}".strip())
        col4.metric("Valid pixels", f"{len(month_df):,}")
    else:
        col1.metric("National Mean", f"{county_month_df[variable].mean():.2f} {meta['unit']}".strip())
        col2.metric("Lowest County", f"{county_month_df[variable].min():.2f} {meta['unit']}".strip())
        col3.metric("Highest County", f"{county_month_df[variable].max():.2f} {meta['unit']}".strip())
        col4.metric("Valid Counties", f"{len(county_month_df):,}")

    # --- Map -----------------------------------------------------------------
    st.subheader(f"{meta['label']} \u2014 {month_labels[month_idx]}")

    if resolution == "5 km Grid":
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
        fig = px.scatter_map(month_df.dropna(subset=[variable]), color=variable, **map_kwargs)
        fig.update_traces(marker=dict(size=7, opacity=0.8))
    else:
        # County Choropleth Map
        if counties_geojson is None:
            st.error("County boundary GeoJSON not found. Please run the mapping script.")
            fig = px.scatter(title="Error: GeoJSON missing")
        else:
            fig = px.choropleth_map(
                county_month_df,
                geojson=counties_geojson,
                locations="county",
                featureidkey="properties.shapeName",
                color=variable,
                color_continuous_scale=meta["colorscale"],
                center=dict(lat=0.5, lon=37.9),
                zoom=5.2,
                map_style="open-street-map",
                height=650,
                labels={variable: f"{meta['label']} ({meta['unit']})".strip()},
                opacity=0.7
            )

    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    # --- Monthly trend ----------------------------------------------
    st.subheader("Monthly trend")

    if resolution == "5 km Grid":
        summary = df.groupby("month")[variable].mean().reset_index()
        trend_fig = px.line(
            summary,
            x="month",
            y=variable,
            labels={variable: meta["label"], "month": "Month"},
            title="National Average"
        )
    else:
        selected_counties = st.multiselect(
            "Compare specific counties:",
            options=sorted(county_df["county"].unique()),
            default=[]
        )

        if selected_counties:
            trend_df = county_df[county_df["county"].isin(selected_counties)]
            trend_fig = px.line(
                trend_df,
                x="month",
                y=variable,
                color="county",
                labels={variable: meta["label"], "month": "Month"},
                title="County Comparison"
            )
        else:
            summary = county_df.groupby("month")[variable].mean().reset_index()
            trend_fig = px.line(
                summary,
                x="month",
                y=variable,
                labels={variable: meta["label"], "month": "Month"},
                title="National Average (Across Counties)"
            )

    trend_fig.add_vline(x=selected_month, line_dash="dash", line_color="red")
    trend_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(trend_fig, width="stretch")


    # --- Download full dataset -------------------------------------------------
    @st.cache_data(show_spinner=False)
    def _read_parquet_bytes() -> bytes:
        return PARQUET_CACHE.read_bytes()


    @st.cache_data(show_spinner=False)
    def _national_monthly_averages(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("month")[list(VARIABLES.keys())].mean().reset_index()


    st.subheader("Download data")
    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

    with dl_col1:
        st.markdown("**Full dataset (Parquet)**")
        st.caption(f"All {len(months)} months \u00d7 all variables.")
        if PARQUET_CACHE.exists():
            st.download_button(
                "Download combined.parquet",
                _read_parquet_bytes(),
                file_name="kenya_climate_vegetation_all_months.parquet",
                mime="application/octet-stream",
            )
        else:
            st.info("Parquet cache not found.")

    with dl_col2:
        st.markdown("**Full dataset + code (ZIP)**")
        st.caption("Every raw GeoTIFF raster (.tif) plus source code, via GitHub.")
        st.link_button(
            "Download GitHub repo (.zip)",
            "https://github.com/Masuzyo/kenya-climate-dashboard/archive/refs/heads/master.zip",
        )

    with dl_col3:
        st.markdown("**County averages (CSV)**")
        st.caption("One row per month per county.")
        st.download_button(
            "Download county averages CSV",
            county_df.to_csv(index=False),
            file_name="kenya_monthly_county_averages.csv",
            mime="text/csv",
        )

    with dl_col4:
        st.markdown("**National averages (CSV)**")
        st.caption("One row per month: Kenya-wide mean.")
        st.download_button(
            "Download national averages CSV",
            _national_monthly_averages(df).to_csv(index=False),
            file_name="kenya_monthly_national_averages.csv",
            mime="text/csv",
        )

    # --- Raw data / export -----------------------------------------------------
    with st.expander("Raw data (selected month)"):
        st.dataframe(month_df.head(1000), width="stretch")
        st.download_button(
            "Download this month's full CSV",
            month_df.to_csv(index=False),
            file_name=f"kenya_{month_labels[month_idx]}_{variable}.csv",
            mime="text/csv",
        )

with tab_malaria:
    st.header("Simulated Malaria Resistance")
    st.caption("Based on a synthetic database of 100,000 patients, generated using historical temperature and rainfall baselines.")
    
    @st.cache_data(show_spinner="Loading simulated patients...")
    def load_patients(file_mtime):
        import pathlib
        p = pathlib.Path('simulated_malaria_patients.csv')
        if p.exists():
            return pd.read_csv(p)
        return None
        
    p = pathlib.Path('simulated_malaria_patients.csv')
    mtime = p.stat().st_mtime if p.exists() else 0
    pdf = load_patients(mtime)
    
    if pdf is not None:
        pdf['is_mutant'] = pdf['human_genotype'] == 'HbAS (Sickle Trait / Resistant)'
        total_patients = len(pdf)
        res_cases = pdf['is_mutant'].sum()
        res_rate = (res_cases / total_patients) * 100
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Simulated Patients", f"{total_patients:,}")
        m_col2.metric("Total Resistant (HbAS)", f"{res_cases:,}")
        m_col3.metric("Overall HbAS Trait Rate", f"{res_rate:.1f}%")
        
        st.subheader("HbAS (Sickle Trait) Prevalence by County")
        county_res = pdf.groupby('county')['is_mutant'].mean().reset_index()
        county_res['Prevalence Rate (%)'] = county_res['is_mutant'] * 100
        
        if counties_geojson:
            fig_res = px.choropleth_map(
                county_res,
                geojson=counties_geojson,
                locations="county",
                featureidkey="properties.shapeName",
                color="Prevalence Rate (%)",
                color_continuous_scale="Reds",
                center=dict(lat=0.5, lon=37.9),
                zoom=5.2,
                map_style="open-street-map",
                height=500,
                opacity=0.7
            )
            fig_res.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_res, width="stretch")
            
        st.subheader("Demographics & Climate Correlation")
        dem_col1, dem_col2, dem_col3 = st.columns(3)
        with dem_col1:
            sex_res = pdf.groupby(['sex', 'human_genotype']).size().reset_index(name='count')
            fig_sex = px.bar(sex_res, x='sex', y='count', color='human_genotype', title="Genotypes by Sex")
            st.plotly_chart(fig_sex, use_container_width=True)
            
        with dem_col2:
            pdf['Age Group'] = pd.cut(pdf['age'], bins=[0, 10, 20, 30, 40, 50, 60, 100], labels=['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '60+'])
            age_res = pdf.groupby('Age Group', observed=False)['is_mutant'].mean().reset_index()
            age_res['Rate'] = age_res['is_mutant'] * 100
            fig_age = px.line(age_res, x='Age Group', y='Rate', title="HbAS Trait Rate by Age", markers=True)
            st.plotly_chart(fig_age, use_container_width=True)
            
        with dem_col3:
            if 'mean_temp_c' in county_df.columns and 'elevation_m' in county_df.columns:
                baseline = county_df.groupby('county')[['mean_temp_c', 'rain_mm', 'elevation_m']].mean().reset_index()
                res_climate = county_res.merge(baseline, on='county')
                fig_scatter = px.scatter(
                    res_climate, 
                    x='mean_temp_c', 
                    y='Prevalence Rate (%)', 
                    color='rain_mm', 
                    size='elevation_m',
                    hover_data=['county'], 
                    title="Temp, Rain & Elevation vs HbAS Prevalence"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Climate data missing")
                
        st.markdown("---")
        st.subheader("County Population & Resistance Breakdown")
        county_table = pdf.groupby('county').agg(
            Total_Patients=('patient_id', 'count'),
            HbAS_Cases=('is_mutant', 'sum')
        ).reset_index()
        county_table['HbAS_Prevalence_%'] = (county_table['HbAS_Cases'] / county_table['Total_Patients']) * 100
        
        st.dataframe(
            county_table.sort_values('Total_Patients', ascending=False),
            column_config={
                "county": "County",
                "Total_Patients": st.column_config.NumberColumn("Simulated Population", format="%d"),
                "HbAS_Cases": st.column_config.NumberColumn("HbAS Cases", format="%d"),
                "HbAS_Prevalence_%": st.column_config.NumberColumn("Prevalence (%)", format="%.2f%%")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.error("Simulated patient dataset not found. Please run simulate_patients.py first.")

def make_gauss_markov_diagnostics(
    fitted_vals: np.ndarray,
    residuals: np.ndarray,
    entity_labels: pd.Series | None = None,
    title_prefix: str = "Gauss-Markov Residual Diagnostics",
    max_scatter_points: int = 5000,
) -> go.Figure:
    n = len(residuals)
    std_residuals = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    sqrt_abs_std = np.sqrt(np.abs(std_residuals))

    if n > max_scatter_points:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n, size=max_scatter_points, replace=False)
    else:
        sample_indices = np.arange(n)

    fitted_s = fitted_vals[sample_indices]
    resid_s = residuals[sample_indices]
    std_resid_s = std_residuals[sample_indices]
    sqrt_abs_std_s = sqrt_abs_std[sample_indices]

    sorted_residuals = np.sort(std_resid_s)
    n_sample = len(sorted_residuals)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n_sample + 1) - 0.5) / n_sample)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. Residuals vs. Fitted (Linearity & Drift)",
            "2. Normal Q-Q Plot (Normality)",
            "3. Scale-Location (Homoscedasticity)",
            "4. Residuals by County (Spatial Heterogeneity)" if entity_labels is not None else "4. Residual Distribution",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    # 1. Residuals vs Fitted
    fig.add_trace(
        go.Scatter(
            x=fitted_s,
            y=resid_s,
            mode="markers",
            marker=dict(color="#1f77b4", size=4, opacity=0.35),
            name="Residuals",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[float(np.min(fitted_s)), float(np.max(fitted_s))],
            y=[0.0, 0.0],
            mode="lines",
            line=dict(color="black", dash="dash", width=1.5),
            name="Zero Reference",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    if len(fitted_s) > 10:
        poly = np.poly1d(np.polyfit(fitted_s, resid_s, 2))
        xs_curve = np.linspace(float(np.min(fitted_s)), float(np.max(fitted_s)), 100)
        fig.add_trace(
            go.Scatter(
                x=xs_curve,
                y=poly(xs_curve),
                mode="lines",
                line=dict(color="#d62728", width=2),
                name="Fitted Trend",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # 2. Normal Q-Q Plot
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode="markers",
            marker=dict(color="#1f77b4", size=4, opacity=0.35),
            name="Sample Quantiles",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    q_lim = max(abs(float(np.min(theoretical_quantiles))), abs(float(np.max(theoretical_quantiles))))
    fig.add_trace(
        go.Scatter(
            x=[-q_lim, q_lim],
            y=[-q_lim, q_lim],
            mode="lines",
            line=dict(color="#d62728", dash="dash", width=1.5),
            name="Normal 45-deg Line",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Scale-Location
    fig.add_trace(
        go.Scatter(
            x=fitted_s,
            y=sqrt_abs_std_s,
            mode="markers",
            marker=dict(color="#1f77b4", size=4, opacity=0.35),
            name="Scale-Location",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    if len(fitted_s) > 10:
        poly_sl = np.poly1d(np.polyfit(fitted_s, sqrt_abs_std_s, 2))
        fig.add_trace(
            go.Scatter(
                x=xs_curve,
                y=poly_sl(xs_curve),
                mode="lines",
                line=dict(color="#d62728", width=2),
                name="Spread Trend",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # 4. County Boxplot or Histogram
    if entity_labels is not None:
        plot_df = pd.DataFrame({"County": entity_labels, "Residual": residuals})
        for c in sorted(plot_df["County"].unique()):
            c_resid = plot_df[plot_df["County"] == c]["Residual"]
            fig.add_trace(
                go.Box(
                    y=c_resid,
                    name=c,
                    showlegend=False,
                    boxpoints=False,
                    marker=dict(size=2),
                ),
                row=2,
                col=2,
            )
    else:
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=40,
                marker_color="#1f77b4",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles N(0, 1)", row=1, col=2)
    fig.update_yaxes(title_text="Standardized Residuals", row=1, col=2)
    fig.update_xaxes(title_text="Fitted Values", row=2, col=1)
    fig.update_yaxes(title_text="sqrt(|Std Residuals|)", row=2, col=1)
    if entity_labels is not None:
        fig.update_xaxes(title_text="County", row=2, col=2, tickangle=45)
        fig.update_yaxes(title_text="Residuals", row=2, col=2)
    else:
        fig.update_xaxes(title_text="Residual Value", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

    fig.update_layout(
        title_text=title_prefix,
        height=720,
        margin=dict(l=40, r=20, t=50, b=50),
    )
    return fig


with tab_modeling:
    st.subheader("Statistical & Econometric Modeling Suite")
    st.caption("Longitudinal panel econometrics (N = 13,464 county-months) and cross-sectional spatial epidemiology (N = 45 counties).")

    domain_choice = st.radio(
        "Select Analytical Domain:",
        options=[
            "Mode 1: Longitudinal Ecohydrological Panel (13,464 County-Months)",
            "Mode 2: Spatial Epidemiology & Malaria Allele Resistance (45 Counties)",
        ],
        horizontal=True,
    )

    if domain_choice.startswith("Mode 1"):
        st.markdown("#### Mode 1: Longitudinal Ecohydrological Panel Econometrics")
        st.caption("Estimating dynamic interactions across 44 Kenyan counties over 306 monthly timesteps (2000–2026).")

        m1_col1, m1_col2, m1_col3 = st.columns([1, 1, 1])
        with m1_col1:
            target_var = st.selectbox(
                "Target Variable:",
                options=["ndvi", "evi", "soil_moisture_m3m3", "veg_cover_pct"],
                format_func=lambda v: f"{VARIABLES[v]['label']} ({v})",
            )
        with m1_col2:
            panel_model_choice = st.selectbox(
                "Econometric Specification:",
                options=[
                    "Pooled OLS (Benchmark)",
                    "Entity Fixed Effects (Within Estimator)",
                    "Two-Way Fixed Effects (TWFE: Entity + Month + Year)",
                    "Random Forest Regressor (Nonlinear Panel)",
                ],
            )
        with m1_col3:
            include_ar1 = st.checkbox("Include Autoregressive Lag-1 (y_{t-1})", value=True)

        candidate_covars = [v for v in VARIABLES.keys() if v != target_var and v not in ["elevation_m"]]
        default_covars = [v for v in ["rain_mm", "mean_temp_c", "soil_moisture_m3m3", "humidity_rh_pct", "wind_speed_ms"] if v in candidate_covars]
        selected_covars = st.multiselect("Predictor Covariates:", options=candidate_covars, default=default_covars)

        if not selected_covars and not include_ar1:
            st.warning("Please select at least one predictor covariate or enable the autoregressive lag.")
        else:
            panel_work = county_df.sort_values(["county", "month"]).reset_index(drop=True).copy()
            covars_used = list(selected_covars)
            if include_ar1:
                panel_work["target_lag1"] = panel_work.groupby("county")[target_var].shift(1)
                covars_used.append("target_lag1")

            panel_clean = panel_work.dropna(subset=[target_var] + covars_used).copy()

            if panel_model_choice == "Pooled OLS (Benchmark)":
                X = sm.add_constant(panel_clean[covars_used])
                y = panel_clean[target_var]
                res = sm.OLS(y, X).fit()
                y_pred = res.fittedvalues.values
                residuals = res.resid.values
                r2 = res.rsquared
                r2_adj = res.rsquared_adj
                r2_within = None
                param_df = pd.DataFrame({
                    "Covariate": res.params.index,
                    "Coefficient": res.params.values,
                    "Std Error": res.bse.values,
                    "t-statistic": res.tvalues.values,
                    "p-value": res.pvalues.values,
                    "CI Lower (95%)": res.conf_int()[0].values,
                    "CI Upper (95%)": res.conf_int()[1].values,
                })

            elif panel_model_choice == "Entity Fixed Effects (Within Estimator)":
                county_dummies = pd.get_dummies(panel_clean["county"], prefix="county", drop_first=True, dtype=float)
                X = pd.concat([sm.add_constant(panel_clean[covars_used]), county_dummies], axis=1)
                y = panel_clean[target_var]
                res = sm.OLS(y, X).fit()
                y_pred = res.fittedvalues.values
                residuals = res.resid.values
                r2 = res.rsquared
                r2_adj = res.rsquared_adj
                y_within_dev = panel_clean.groupby("county")[target_var].transform(lambda s: s - s.mean())
                ss_tot_within = np.sum(y_within_dev ** 2)
                ss_res_within = np.sum(residuals ** 2)
                r2_within = 1.0 - (ss_res_within / (ss_tot_within + 1e-12))
                structural_cols = ["const"] + covars_used
                param_df = pd.DataFrame({
                    "Covariate": structural_cols,
                    "Coefficient": res.params[structural_cols].values,
                    "Std Error": res.bse[structural_cols].values,
                    "t-statistic": res.tvalues[structural_cols].values,
                    "p-value": res.pvalues[structural_cols].values,
                    "CI Lower (95%)": res.conf_int().loc[structural_cols, 0].values,
                    "CI Upper (95%)": res.conf_int().loc[structural_cols, 1].values,
                })

            elif panel_model_choice == "Two-Way Fixed Effects (TWFE: Entity + Month + Year)":
                county_dummies = pd.get_dummies(panel_clean["county"], prefix="county", drop_first=True, dtype=float)
                month_dummies = pd.get_dummies(panel_clean["month"].dt.month, prefix="mo", drop_first=True, dtype=float)
                year_dummies = pd.get_dummies(panel_clean["month"].dt.year, prefix="yr", drop_first=True, dtype=float)
                X = pd.concat([sm.add_constant(panel_clean[covars_used]), county_dummies, month_dummies, year_dummies], axis=1)
                y = panel_clean[target_var]
                res = sm.OLS(y, X).fit()
                y_pred = res.fittedvalues.values
                residuals = res.resid.values
                r2 = res.rsquared
                r2_adj = res.rsquared_adj
                y_within_dev = panel_clean.groupby("county")[target_var].transform(lambda s: s - s.mean())
                ss_tot_within = np.sum(y_within_dev ** 2)
                ss_res_within = np.sum(residuals ** 2)
                r2_within = 1.0 - (ss_res_within / (ss_tot_within + 1e-12))
                structural_cols = ["const"] + covars_used
                param_df = pd.DataFrame({
                    "Covariate": structural_cols,
                    "Coefficient": res.params[structural_cols].values,
                    "Std Error": res.bse[structural_cols].values,
                    "t-statistic": res.tvalues[structural_cols].values,
                    "p-value": res.pvalues[structural_cols].values,
                    "CI Lower (95%)": res.conf_int().loc[structural_cols, 0].values,
                    "CI Upper (95%)": res.conf_int().loc[structural_cols, 1].values,
                })

            else:  # Random Forest Regressor
                X = panel_clean[covars_used]
                y = panel_clean[target_var]
                rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                y_pred = rf.predict(X)
                residuals = y.values - y_pred
                r2 = r2_score(y, y_pred)
                r2_adj = 1.0 - (1.0 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
                r2_within = None
                param_df = None
                imp_df = pd.DataFrame({"Predictor": covars_used, "Importance": rf.feature_importances_}).sort_values(by="Importance", ascending=True)

            st.markdown("### Model Performance Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Overall R-squared", f"{r2:.4f}")
            if r2_within is not None:
                m_col2.metric("Within R-squared", f"{r2_within:.4f}")
            else:
                m_col2.metric("Adjusted R-squared", f"{r2_adj:.4f}")
            resid_std = float(np.std(residuals))
            m_col3.metric("Residual Std Dev (σ_ε)", f"{resid_std:.4f}")
            dw_val = float(durbin_watson(residuals))
            m_col4.metric("Durbin-Watson", f"{dw_val:.3f}")

            jb_stat, jb_p, skew, kurt = jarque_bera(residuals)
            if panel_model_choice != "Random Forest Regressor (Nonlinear Panel)":
                bp_lm, bp_p, _, _ = het_breuschpagan(residuals, sm.add_constant(panel_clean[covars_used]))
                st.info(
                    f"**Formal Econometric Diagnostics:**  \n"
                    f"• **Durbin-Watson Stat:** DW = {dw_val:.3f} ({'Absence of first-order serial correlation' if dw_val >= 1.7 else 'Positive temporal persistence detected'})  \n"
                    f"• **Jarque-Bera Normality Test:** Stat = {jb_stat:.1f} (p = {jb_p:.2e}, Skewness = {skew:.2f}, Kurtosis = {kurt:.2f})  \n"
                    f"• **Breusch-Pagan Homoscedasticity Test:** LM Stat = {bp_lm:.1f} (p = {bp_p:.2e})"
                )
            else:
                st.info(
                    f"**Random Forest Residual Diagnostics:**  \n"
                    f"• **Durbin-Watson Stat:** DW = {dw_val:.3f}  \n"
                    f"• **Jarque-Bera Normality Test:** Stat = {jb_stat:.1f} (p = {jb_p:.2e}, Skewness = {skew:.2f}, Kurtosis = {kurt:.2f})"
                )

            if param_df is not None:
                st.markdown("### Structural Parameter Estimates")
                st.dataframe(
                    param_df.round(4),
                    column_config={
                        "Covariate": "Covariate",
                        "Coefficient": st.column_config.NumberColumn("Coefficient (β)", format="%.4f"),
                        "Std Error": st.column_config.NumberColumn("Std Error", format="%.4f"),
                        "t-statistic": st.column_config.NumberColumn("t-statistic", format="%.2f"),
                        "p-value": st.column_config.NumberColumn("p-value", format="%.4f"),
                        "CI Lower (95%)": st.column_config.NumberColumn("CI Lower (95%)", format="%.4f"),
                        "CI Upper (95%)": st.column_config.NumberColumn("CI Upper (95%)", format="%.4f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.markdown("### Random Forest Feature Importance")
                fig_rf = px.bar(
                    imp_df,
                    x="Importance",
                    y="Predictor",
                    orientation="h",
                    title="MDI Feature Importances",
                    color="Importance",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_rf, use_container_width=True)

            st.markdown("### Gauss-Markov Residual Diagnostics Suite")
            diag_fig = make_gauss_markov_diagnostics(
                fitted_vals=y_pred,
                residuals=residuals,
                entity_labels=panel_clean["county"],
                title_prefix=f"Residual Diagnostics: {panel_model_choice} (N = {len(panel_clean):,})",
            )
            st.plotly_chart(diag_fig, use_container_width=True)

    else:
        st.markdown("#### Mode 2: Spatial Epidemiology & Malaria Allele Resistance")
        st.caption("Cross-sectional ecological modeling of county-level HbAS sickle-cell trait prevalence (N = 45 counties).")

        epi_model_choice = st.radio(
            "Select Evaluation Model:",
            options=[
                "LASSO Regression (L1 Penalty)",
                "Ordinary Least Squares (OLS) with Diagnostics",
                "Random Forest (Exploratory)",
                "XGBoost (Exploratory)",
            ],
            horizontal=True,
        )

        if pdf is not None and not county_df.empty:
            county_table = pdf.groupby('county').agg(
                n_resistant=('is_mutant', 'sum'),
                n_tested=('patient_id', 'count')
            ).reset_index()

            predictors = ['mean_temp_c', 'max_temp_c', 'min_temp_c', 'rain_mm', 'humidity_rh_pct', 'soil_moisture_m3m3', 'ndvi', 'elevation_m', 'urban_pct']
            available_predictors = [p for p in predictors if p in county_df.columns]
            baseline = county_df.groupby('county')[available_predictors].mean().reset_index()
            model_df = county_table.merge(baseline, on='county')

            for p in available_predictors:
                model_df[f'z_{p}'] = (model_df[p] - model_df[p].mean()) / (model_df[p].std() + 1e-12)

            y = model_df['n_resistant'] / model_df['n_tested']
            X_base = model_df[[f'z_{p}' for p in available_predictors]]

            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_poly = poly.fit_transform(X_base)
            feature_names = poly.get_feature_names_out(X_base.columns)
            X = pd.DataFrame(X_poly, columns=feature_names)

            if epi_model_choice == "LASSO Regression (L1 Penalty)":
                model_df['n_susceptible'] = model_df['n_tested'] - model_df['n_resistant']
                endog = model_df[['n_resistant', 'n_susceptible']]
                exog = sm.add_constant(X)

                alphas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                best_alpha = alphas[0]
                best_mse = float('inf')

                for alpha in alphas:
                    fold_mses = []
                    for train_idx, test_idx in kf.split(X):
                        train_endog, test_endog = endog.iloc[train_idx], endog.iloc[test_idx]
                        train_exog, test_exog = exog.iloc[train_idx], exog.iloc[test_idx]
                        try:
                            glm = sm.GLM(train_endog, train_exog, family=sm.families.Binomial())
                            res = glm.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=1.0)
                            y_test_pred = res.predict(test_exog)
                            y_test_true = test_endog['n_resistant'] / (test_endog['n_resistant'] + test_endog['n_susceptible'])
                            fold_mses.append(mean_squared_error(y_test_true, y_test_pred))
                        except Exception:
                            fold_mses.append(float('inf'))
                    avg_mse = np.mean(fold_mses)
                    if avg_mse < best_mse:
                        best_mse = avg_mse
                        best_alpha = alpha

                glm = sm.GLM(endog, exog, family=sm.families.Binomial())
                res = glm.fit_regularized(method='elastic_net', alpha=best_alpha, L1_wt=1.0)
                y_pred = res.predict(exog)
                importances = res.params.drop('const', errors='ignore').values
                title_prefix = "LASSO Coefficients (Logit Link)"
                best_params_str = f"**Optimal Alpha:** {best_alpha}"

            elif epi_model_choice == "Ordinary Least Squares (OLS) with Diagnostics":
                X_ols = sm.add_constant(X_base)
                res_ols = sm.OLS(y, X_ols).fit()
                y_pred = res_ols.fittedvalues.values
                residuals_ols = res_ols.resid.values
                importances = res_ols.params.drop('const', errors='ignore').values
                feature_names = X_base.columns
                title_prefix = "OLS Standardized Coefficients"
                best_params_str = "Standard Unpenalized OLS"

            elif epi_model_choice == "Random Forest (Exploratory)":
                base_model = RandomForestRegressor(random_state=42)
                param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 3, 5]}
                grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X, y)
                model = grid_search.best_estimator_
                y_pred = model.predict(X)
                importances = model.feature_importances_
                title_prefix = "Relative Feature Importance"
                best_params_str = ", ".join([f"**{k}:** {v}" for k, v in grid_search.best_params_.items()])

            else:
                base_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
                param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]}
                grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X, y)
                model = grid_search.best_estimator_
                y_pred = model.predict(X)
                importances = model.feature_importances_
                title_prefix = "Relative Feature Importance"
                best_params_str = ", ".join([f"**{k}:** {v}" for k, v in grid_search.best_params_.items()])

            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            st.markdown(f"### {epi_model_choice.split(' ')[0]} Performance Metrics")
            st.info(f"**Hyperparameter Specification:** {best_params_str}")
            col1, col2 = st.columns(2)
            col1.metric("Model R-squared (R²)", f"{r2:.4f}")
            col2.metric("Mean Squared Error (MSE)", f"{mse:.6f}")

            imp_df = pd.DataFrame({'Predictor': feature_names, 'Importance': importances})
            imp_df['Abs_Importance'] = imp_df['Importance'].abs()
            imp_df = imp_df.sort_values(by='Abs_Importance', ascending=False).head(20)
            imp_df = imp_df.sort_values(by='Abs_Importance', ascending=True)

            fig_imp = px.bar(
                imp_df,
                x='Importance',
                y='Predictor',
                orientation='h',
                title=f"{title_prefix} (Top 20)",
                color='Importance',
                color_continuous_scale="RdBu" if "LASSO" in epi_model_choice or "OLS" in epi_model_choice else "Viridis",
            )
            if "LASSO" in epi_model_choice or "OLS" in epi_model_choice:
                fig_imp.add_vline(x=0.0, line_width=2, line_color="black")
            st.plotly_chart(fig_imp, use_container_width=True)

            if epi_model_choice == "Ordinary Least Squares (OLS) with Diagnostics":
                st.markdown("### Gauss-Markov Residual Diagnostics Suite")
                ols_diag_fig = make_gauss_markov_diagnostics(
                    fitted_vals=y_pred,
                    residuals=residuals_ols,
                    entity_labels=None,
                    title_prefix="OLS Gauss-Markov Residual Diagnostics (N = 45 Counties)",
                )
                st.plotly_chart(ols_diag_fig, use_container_width=True)

        else:
            st.error("Missing simulated patient or climate data for spatial modeling.")
