#!/usr/bin/env python3
"""
Python Benchmark - Large Clinical Trial (500K patients, 50 predictors)

This will show GPU's true power on realistic large-scale data.
"""

import numpy as np
import pandas as pd
import time
from pyregression import lm

print()
print("="*80)
print("PYTHON BENCHMARK - Large Clinical Trial Dataset")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading large_clinical_trial.csv...")
print("(This may take a moment - it's a large file)")
print()

start_load = time.time()
data = pd.read_csv('large_clinical_trial.csv')
load_time = time.time() - start_load

print(f"✓ Data loaded in {load_time:.2f} seconds")
print(f"  Observations: {len(data):,}")
print(f"  Variables: {len(data.columns)}")
print(f"  Memory: {data.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print()

# Get predictors
predictors = [col for col in data.columns if col != 'outcome']

print(f"Predictors: {len(predictors)}")
print(f"  First 10: {predictors[:10]}")
print(f"  ... and {len(predictors) - 10} more")
print()

# ============================================================================
# CPU BENCHMARK
# ============================================================================

print("="*80)
print("CPU BENCHMARK (PyRegression CPU Backend)")
print("="*80)
print()

print("Fitting model on CPU with 500,000 observations...")
print("(This will take a while - be patient!)")
print()

# Warm-up
print("Warming up CPU...")
# Use smaller subset for warmup to save time
warmup_data = data.sample(n=10000, random_state=42)
_ = lm(y='outcome', X=predictors, data=warmup_data, backend='cpu')
print("✓ Warm-up complete")
print()

# Full benchmark
print("Running CPU benchmark on full dataset...")
start = time.time()
model_cpu = lm(y='outcome', X=predictors, data=data, backend='cpu')
cpu_time = time.time() - start

print(f"✓ Model fitted in {cpu_time:.2f} seconds")
print()

# Summary (abbreviated for large output)
print("Model Summary (abbreviated for large model):")
print("-"*80)
print(f"Dependent variable: outcome")
print(f"Observations: {len(data):,}")
print(f"Predictors: {len(predictors)}")
print(f"R-squared: {model_cpu.r_squared:.6f}")
print(f"Adj R-squared: {model_cpu.adj_r_squared:.6f}")
print(f"Residual Std Error: {model_cpu.sigma:.6f}")
print(f"F-statistic: {model_cpu.f_statistic:.2f}")
print()

print("First 10 coefficients:")
print(model_cpu.coef.head(10))
print()

print("="*80)
print("CPU PERFORMANCE")
print("="*80)
print(f"Data load time:         {load_time:>20.2f} seconds")
print(f"Model fit time:         {cpu_time:>20.2f} seconds")
print(f"Total time:             {load_time + cpu_time:>20.2f} seconds")
print(f"Observations/second:    {len(data)/cpu_time:>20.1f}")
print(f"Backend:                {model_cpu.backend.name:>20s}")
print()

# ============================================================================
# GPU BENCHMARK
# ============================================================================

print()
print("="*80)
print("GPU BENCHMARK (PyRegression GPU Backend)")
print("="*80)
print()

from pyregression._backends import detect_gpu_capabilities

caps = detect_gpu_capabilities()

if not caps.has_gpu:
    print("⚠ No GPU detected. Skipping GPU benchmark.")
    print()
else:
    print(f"GPU detected: {caps.gpu_name}")
    print(f"GPU type: {caps.gpu_type}")
    print()
    
    print("Fitting model on GPU with 500,000 observations...")
    print("(Watch the GPU work!)")
    print()
    
    # Warm-up
    print("Warming up GPU...")
    warmup_data = data.sample(n=10000, random_state=42)
    _ = lm(y='outcome', X=predictors, data=warmup_data, backend='gpu')
    print("✓ GPU warm-up complete")
    print()
    
    # Full benchmark
    print("Running GPU benchmark on full dataset...")
    start = time.time()
    model_gpu = lm(y='outcome', X=predictors, data=data, backend='gpu')
    gpu_time = time.time() - start
    
    print(f"✓ Model fitted in {gpu_time:.2f} seconds")
    print()
    
    # Summary
    print("Model Summary (abbreviated):")
    print("-"*80)
    print(f"Dependent variable: outcome")
    print(f"Observations: {len(data):,}")
    print(f"Predictors: {len(predictors)}")
    print(f"R-squared: {model_gpu.r_squared:.6f}")
    print(f"Adj R-squared: {model_gpu.adj_r_squared:.6f}")
    print(f"Residual Std Error: {model_gpu.sigma:.6f}")
    print(f"F-statistic: {model_gpu.f_statistic:.2f}")
    print()
    
    print("First 10 coefficients:")
    print(model_gpu.coef.head(10))
    print()
    
    print("="*80)
    print("GPU PERFORMANCE")
    print("="*80)
    print(f"Data load time:         {load_time:>20.2f} seconds")
    print(f"Model fit time:         {gpu_time:>20.2f} seconds")
    print(f"Total time:             {load_time + gpu_time:>20.2f} seconds")
    print(f"Observations/second:    {len(data)/gpu_time:>20.1f}")
    print(f"Backend:                {model_gpu.backend.name:>20s}")
    print()
    
    # Comparison
    print("="*80)
    print("CPU vs GPU COMPARISON")
    print("="*80)
    print(f"CPU time:               {cpu_time:>20.2f} seconds")
    print(f"GPU time:               {gpu_time:>20.2f} seconds")
    print(f"Speedup:                {cpu_time/gpu_time:>20.2f}x")
    print()
    
    # Statistical comparison
    print("="*80)
    print("STATISTICAL COMPARISON (GPU vs CPU)")
    print("="*80)
    
    # Compare first 10 coefficients
    print()
    print("First 10 coefficients comparison:")
    print("-"*80)
    print(f"{'Variable':<20} {'CPU':>20} {'GPU':>20} {'Diff':>15}")
    print("-"*80)
    
    max_diff = 0
    for var in model_cpu.var_names[:10]:
        cpu_val = model_cpu.coef[var]
        gpu_val = model_gpu.coef[var]
        diff = abs(cpu_val - gpu_val)
        max_diff = max(max_diff, diff)
        
        print(f"{var:<20} {cpu_val:>20.10f} {gpu_val:>20.10f} {diff:>15.2e}")
    
    print("-"*80)
    
    # Overall statistics match
    print()
    print("Overall statistics comparison:")
    print("-"*80)
    print(f"{'Statistic':<30} {'CPU':>20} {'GPU':>20} {'Diff':>15}")
    print("-"*80)
    
    stats = [
        ('R-squared', model_cpu.r_squared, model_gpu.r_squared),
        ('Adj R-squared', model_cpu.adj_r_squared, model_gpu.adj_r_squared),
        ('Residual SE', model_cpu.sigma, model_gpu.sigma),
        ('F-statistic', model_cpu.f_statistic, model_gpu.f_statistic),
    ]
    
    for name, cpu_val, gpu_val in stats:
        diff = abs(cpu_val - gpu_val)
        print(f"{name:<30} {cpu_val:>20.10f} {gpu_val:>20.10f} {diff:>15.2e}")
    
    print("-"*80)
    print()
    
    if max_diff < 1e-4:
        print("✓ PASS: GPU matches CPU to statistical precision (< 1e-4)")
    else:
        print(f"⚠ WARNING: GPU differs from CPU by {max_diff:.2e}")
    print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print()
print("="*80)
print("BENCHMARK COMPLETE")
print("="*80)
print()
print("This dataset (500K patients, 50 predictors) shows:")
print("  - Real-world performance at scale")
print("  - Where GPU acceleration truly matters")
print("  - PyRegression's capability for large clinical trials")
print()