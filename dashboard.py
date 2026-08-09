"""Interactive dashboard: monthly Kenya climate & vegetation data on a map.

Run with:
    streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import json
import pathlib
import statsmodels.api as sm
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
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
    "ERA5-Land (humidity, soil moisture, wind speed), JRC (surface water) — 5 km monthly grid."
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
        fig = px.scatter_map(month_df, color=variable, **map_kwargs)
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
        st.caption("Every raw CSV plus source code, via GitHub.")
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

with tab_modeling:
    st.subheader("Ecological Machine Learning & Statistical Modeling")
    st.write("Modeling the association between standardized climatic variables and the proportion of simulated individuals carrying the HbAS malaria-protective allele.")
    
    model_choice = st.radio(
        "Select Evaluation Model:",
        options=["LASSO Regression (L1 Penalty)", "Random Forest (Exploratory)", "XGBoost (Exploratory)"],
        horizontal=True
    )
    
    if pdf is not None and not county_df.empty:
        # 1. Aggregate pdf to get outcomes
        county_table = pdf.groupby('county').agg(
            n_resistant=('is_mutant', 'sum'),
            n_tested=('patient_id', 'count')
        ).reset_index()
        
        # 2. Get baseline climate
        predictors = ['mean_temp_c', 'max_temp_c', 'min_temp_c', 'rain_mm', 'humidity_rh_pct', 'soil_moisture_m3m3', 'wind_u', 'wind_v', 'ndvi', 'elevation_m', 'urban_pct']
        
        # Check which predictors actually exist in the dataframe to avoid KeyErrors
        available_predictors = [p for p in predictors if p in county_df.columns]
        
        baseline = county_df.groupby('county')[available_predictors].mean().reset_index()
        
        # 3. Merge
        model_df = county_table.merge(baseline, on='county')
        
        # 4. Standardize predictors (z-scores)
        for p in available_predictors:
            model_df[f'z_{p}'] = (model_df[p] - model_df[p].mean()) / model_df[p].std()
            
        # 5. Prepare target and features
        y = model_df['n_resistant'] / model_df['n_tested']
        X_base = model_df[[f'z_{p}' for p in available_predictors]]
        
        # Add Interaction Terms (Degree 2)
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_base)
        feature_names = poly.get_feature_names_out(X_base.columns)
        X = pd.DataFrame(X_poly, columns=feature_names)
        
        if model_choice == "LASSO Regression (L1 Penalty)":
            model_df['n_susceptible'] = model_df['n_tested'] - model_df['n_resistant']
            endog = model_df[['n_resistant', 'n_susceptible']]
            exog = sm.add_constant(X)
            
            # Custom 5-Fold CV for statsmodels GLM
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
                        fold_mse = mean_squared_error(y_test_true, y_test_pred)
                        fold_mses.append(fold_mse)
                    except Exception:
                        fold_mses.append(float('inf'))
                
                avg_mse = np.mean(fold_mses)
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_alpha = alpha
            
            # Refit on all data with best alpha
            glm = sm.GLM(endog, exog, family=sm.families.Binomial())
            res = glm.fit_regularized(method='elastic_net', alpha=best_alpha, L1_wt=1.0)
            
            y_pred = res.predict(exog)
            importances = res.params.drop('const', errors='ignore').values
            title_prefix = "LASSO Coefficients (Logit Link)"
            best_params_str = f"**Optimal Alpha:** {best_alpha}"
        elif model_choice == "Random Forest (Exploratory)":
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
        
        st.markdown(f"### {model_choice.split(' ')[0]} Performance Metrics")
        st.info(f"**Optimal Hyperparameters (via 5-Fold CV):** {best_params_str}")
        col1, col2 = st.columns(2)
        col1.metric("Model R-squared ($R^2$)", f"{r2:.4f}")
        col2.metric("Mean Squared Error (MSE)", f"{mse:.6f}")
        
        if model_choice == "LASSO Regression (L1 Penalty)":
            st.markdown("### LASSO Coefficients (L1 Shrunk)")
            st.info("Variables with a coefficient of exactly **0.0** were automatically discarded by the algorithm due to multicollinearity.")
        else:
            st.markdown(f"### Feature Importances")
            
        imp_df = pd.DataFrame({
            'Predictor': feature_names,
            'Importance': importances
        })
        
        # For readability with 66+ interaction terms, keep only the top 20 by absolute magnitude
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
            color_continuous_scale="RdBu" if model_choice == "LASSO Regression (L1 Penalty)" else "Viridis"
        )
        if model_choice == "LASSO Regression (L1 Penalty)":
            fig_imp.add_vline(x=0.0, line_width=2, line_color="black")
            
        st.plotly_chart(fig_imp, use_container_width=True)
            
    else:
        st.error("Missing patient or climate data for modeling.")
