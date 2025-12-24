#!/usr/bin/env python3
"""
Generate large-scale clinical trial dataset for serious benchmarking.

This simulates a real-world Phase III trial or RWE study:
- 500,000 patients
- 50 predictors (demographics, biomarkers, comorbidities, etc.)
- Continuous outcome (e.g., change in disease score)

This will make R struggle while showing GPU's true power.
"""

import numpy as np
import pandas as pd

print("="*80)
print("GENERATING LARGE-SCALE CLINICAL DATASET")
print("="*80)
print()

# Set parameters
N_PATIENTS = 500_000  # Half a million patients
N_PREDICTORS = 50      # 50 covariates

print(f"Generating dataset:")
print(f"  Patients:    {N_PATIENTS:,}")
print(f"  Predictors:  {N_PREDICTORS}")
print(f"  Total cells: {N_PATIENTS * N_PREDICTORS:,}")
print()

# Set seed for reproducibility
np.random.seed(42)

print("Generating predictor matrix...")

# Create realistic predictor names
predictor_names = []
predictor_names += [f'biomarker_{i}' for i in range(1, 11)]      # 10 biomarkers
predictor_names += [f'comorbidity_{i}' for i in range(1, 11)]    # 10 comorbidities
predictor_names += [f'lab_value_{i}' for i in range(1, 11)]      # 10 lab values
predictor_names += [f'medication_{i}' for i in range(1, 11)]     # 10 medications
predictor_names += ['age', 'bmi', 'baseline_score']               # 3 demographics
predictor_names += [f'genetic_marker_{i}' for i in range(1, 8)]  # 7 genetic markers

# Total so far: 10+10+10+10+3+7 = 50 ✓

assert len(predictor_names) == N_PREDICTORS

# Generate data in chunks to avoid memory issues
print("Generating data in chunks (to avoid memory overflow)...")

# Pre-allocate
data = {}

for i, name in enumerate(predictor_names):
    if i % 10 == 0:
        print(f"  Progress: {i}/{N_PREDICTORS} variables")
    
    # Different distributions for realism
    if 'biomarker' in name or 'lab_value' in name:
        # Log-normal distribution (common for biomarkers)
        data[name] = np.random.lognormal(0, 1, N_PATIENTS)
    elif 'comorbidity' in name or 'medication' in name:
        # Binary (0/1)
        data[name] = np.random.binomial(1, 0.3, N_PATIENTS)
    elif name == 'age':
        # Age: normal around 60
        data[name] = np.random.normal(60, 15, N_PATIENTS)
    elif name == 'bmi':
        # BMI: normal around 28
        data[name] = np.random.normal(28, 5, N_PATIENTS)
    elif name == 'baseline_score':
        # Baseline disease score
        data[name] = np.random.normal(50, 10, N_PATIENTS)
    else:
        # Everything else: standard normal
        data[name] = np.random.randn(N_PATIENTS)

print("  ✓ All variables generated")
print()

# Create DataFrame
print("Creating DataFrame...")
df = pd.DataFrame(data)

# Generate outcome with realistic signal
print("Generating outcome variable...")

# True coefficients (sparse - only some predictors matter)
true_coef = np.zeros(N_PREDICTORS)
true_coef[0] = 0.5   # biomarker_1 has strong effect
true_coef[1] = 0.3   # biomarker_2 has moderate effect
true_coef[20] = -0.4  # medication_1 has negative effect
true_coef[30] = 0.2   # age has small effect
true_coef[40] = 0.15  # genetic_marker_3 has small effect

# Generate outcome
X_matrix = df.values
y = X_matrix @ true_coef + np.random.normal(0, 2, N_PATIENTS)

df['outcome'] = y

print("✓ Outcome generated")
print()

# Show summary
print("Dataset summary:")
print(f"  Shape: {df.shape}")
print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print()

print("First few rows:")
print(df.head())
print()

print("Data types:")
print(df.dtypes.value_counts())
print()

# Save to CSV
print("Saving to large_clinical_trial.csv...")
print("(This may take a minute - it's a large file...)")

df.to_csv('large_clinical_trial.csv', index=False)

import os
file_size_mb = os.path.getsize('large_clinical_trial.csv') / 1e6

print(f"✓ Data saved! File size: {file_size_mb:.1f} MB")
print()

print("="*80)
print("READY FOR BENCHMARKING")
print("="*80)
print()
print("This dataset will:")
print("  - Make R work hard (expect 30-120 seconds)")
print("  - Show GPU's true power (expect <5 seconds)")
print("  - Demonstrate real-world performance at scale")
print()
print("Now run:")
print("  1. Rscript benchmark_large_r.R")
print("  2. python benchmark_large_python.py")
print()