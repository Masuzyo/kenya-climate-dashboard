import pandas as pd
import numpy as np
import uuid
import os
import pathlib
import sys

# Ensure we can import from the main directory
sys.path.append(str(pathlib.Path(__file__).parent))
from dashboard_data import load_combined, county_monthly_averages

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_patients(n_patients=100000, out_csv="simulated_malaria_patients.csv"):
    print("Loading historical climate data...")
    df_raw = load_combined()
    df_county_month = county_monthly_averages(df_raw)
    
    # Calculate 20-year average for temperature, rainfall, elevation, and urban fraction
    county_baselines = df_county_month.groupby("county")[["mean_temp_c", "rain_mm", "elevation_m", "urban_pct"]].mean().reset_index()
    
    # Calculate physical area proxy (number of pixels)
    pixel_counts = df_raw.groupby("county").size().reset_index(name="pixel_count")
    # Divide by number of months to get unique pixels
    num_months = df_raw["month"].nunique()
    pixel_counts["pixel_count"] = pixel_counts["pixel_count"] / num_months
    
    county_baselines = county_baselines.merge(pixel_counts, on="county")
    
    print("Calculating resistance probabilities...")
    # Z-score normalization for logistic model
    temp_mean = county_baselines["mean_temp_c"].mean()
    temp_std = county_baselines["mean_temp_c"].std()
    rain_mean = county_baselines["rain_mm"].mean()
    rain_std = county_baselines["rain_mm"].std()
    elev_mean = county_baselines["elevation_m"].mean()
    elev_std = county_baselines["elevation_m"].std()
    
    county_baselines["z_temp"] = (county_baselines["mean_temp_c"] - temp_mean) / temp_std
    county_baselines["z_rain"] = (county_baselines["rain_mm"] - rain_mean) / rain_std
    county_baselines["z_elev"] = (county_baselines["elevation_m"] - elev_mean) / elev_std
    
    # Logistic Model for evolutionary allele frequency (q = frequency of HbS allele)
    beta_0 = -4.5
    beta_temp = 0.4
    beta_rain = 0.2
    beta_elev = -0.5
    
    logit = beta_0 + (beta_temp * county_baselines["z_temp"]) + (beta_rain * county_baselines["z_rain"]) + (beta_elev * county_baselines["z_elev"])
    county_baselines["hbs_allele_frequency"] = sigmoid(logit)
    
    print("\nHbS Allele Frequency Summary by County:")
    print(county_baselines[["county", "mean_temp_c", "rain_mm", "elevation_m", "hbs_allele_frequency"]].sort_values("hbs_allele_frequency", ascending=False).head(10))
    
    print(f"\nGenerating {n_patients} synthetic patients...")
    
    # --- Realistic Population Weighting ---
    # Proxy population = Area (pixels) * (Urban Percentage + 0.01 rural baseline)
    county_baselines["pop_weight"] = county_baselines["pixel_count"] * (county_baselines["urban_pct"] + 0.01)
    
    # Normalize
    county_baselines["pop_weight"] = county_baselines["pop_weight"] / county_baselines["pop_weight"].sum()
    
    # Apply a CAP for highly populated areas (e.g. max 10% per county)
    CAP = 0.10
    excess = 0
    for idx, row in county_baselines.iterrows():
        if row["pop_weight"] > CAP:
            excess += (row["pop_weight"] - CAP)
            county_baselines.at[idx, "pop_weight"] = CAP
            
    # Redistribute excess to non-capped counties
    non_capped_mask = county_baselines["pop_weight"] < CAP
    non_capped_sum = county_baselines.loc[non_capped_mask, "pop_weight"].sum()
    county_baselines.loc[non_capped_mask, "pop_weight"] += (county_baselines.loc[non_capped_mask, "pop_weight"] / non_capped_sum) * excess
    
    counties = county_baselines["county"].values
    weights = county_baselines["pop_weight"].values
    patient_counties = np.random.choice(counties, size=n_patients, p=weights)
    
    # 2. Demographics
    ages = np.random.gamma(shape=2.0, scale=12.0, size=n_patients).astype(int)
    ages = np.clip(ages, 0, 72)
    sexes = np.random.choice(["Male", "Female"], size=n_patients)
    
    patient_ids = [str(uuid.uuid4())[:8] for _ in range(n_patients)]
    
    patients_df = pd.DataFrame({
        "patient_id": patient_ids,
        "county": patient_counties,
        "age": ages,
        "sex": sexes
    })
    
    # 3. Join with allele frequencies and simulate genotypes using Hardy-Weinberg
    patients_df = patients_df.merge(county_baselines[["county", "hbs_allele_frequency"]], on="county", how="left")
    
    q = patients_df["hbs_allele_frequency"].values
    p = 1 - q
    
    # Probabilities for HbAA (Normal), HbAS (Sickle Trait / Malaria Resistant), HbSS (Sickle Disease)
    prob_AA = p**2
    prob_AS = 2 * p * q
    prob_SS = q**2
    
    random_draws = np.random.random(n_patients)
    
    genotypes = np.empty(n_patients, dtype=object)
    
    mask_AA = random_draws < prob_AA
    mask_AS = (random_draws >= prob_AA) & (random_draws < (prob_AA + prob_AS))
    mask_SS = random_draws >= (prob_AA + prob_AS)
    
    genotypes[mask_AA] = "HbAA (Normal)"
    genotypes[mask_AS] = "HbAS (Sickle Trait / Resistant)"
    genotypes[mask_SS] = "HbSS (Sickle Disease)"
    
    patients_df["human_genotype"] = genotypes
    patients_df["malaria_resistant"] = patients_df["human_genotype"] == "HbAS (Sickle Trait / Resistant)"
    
    # Drop internal column
    patients_df = patients_df.drop(columns=["hbs_allele_frequency"])
    
    print("\nGenotype Results:")
    print(patients_df["human_genotype"].value_counts(normalize=True))
    
    print(f"\nSaving to {out_csv}...")
    patients_df.to_csv(out_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    generate_patients()
