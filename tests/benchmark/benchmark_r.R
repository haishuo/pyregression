#!/usr/bin/env Rscript
# ==============================================================================
# R BENCHMARK - Large Clinical Trial (500K patients, 50 predictors)
# ==============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("R BENCHMARK - Large Clinical Trial Dataset\n")
cat(strrep("=", 80), "\n\n")

# Load data
cat("Loading large_clinical_trial.csv...\n")
cat("(This may take a moment - it's a large file)\n\n")

start_load <- Sys.time()
data <- read.csv('large_clinical_trial.csv')
end_load <- Sys.time()
load_time <- as.numeric(end_load - start_load, units = "secs")

cat(sprintf("✓ Data loaded in %.2f seconds\n", load_time))
cat(sprintf("  Observations: %s\n", format(nrow(data), big.mark=",")))
cat(sprintf("  Variables: %d\n", ncol(data)))
cat(sprintf("  Memory: %.2f GB\n", object.size(data) / 1e9))
cat("\n")

# Get predictor names (all except 'outcome')
predictors <- setdiff(names(data), 'outcome')

cat("Predictors:\n")
cat(paste("  ", head(predictors, 10)), "\n")
cat("  ... and", length(predictors) - 10, "more\n\n")

# Build formula
formula_str <- paste("outcome ~", paste(predictors, collapse=" + "))
cat("Formula:\n")
cat("  outcome ~ [50 predictors]\n\n")

# Fit model with timing
cat(strrep("=", 80), "\n")
cat("FITTING MODEL\n")
cat(strrep("=", 80), "\n\n")

cat("Running lm() on 500,000 observations...\n")
cat("(This will take a while - R is working hard!)\n\n")

# Force garbage collection before timing
gc()

# Time the fitting
start_time <- Sys.time()

model <- lm(as.formula(formula_str), data = data)

end_time <- Sys.time()
runtime <- as.numeric(end_time - start_time, units = "secs")

cat(sprintf("✓ Model fitted in %.2f seconds\n\n", runtime))

# Print summary (abbreviated for large output)
cat(strrep("=", 80), "\n")
cat("MODEL SUMMARY (abbreviated)\n")
cat(strrep("=", 80), "\n\n")

summ <- summary(model)

# Residuals
cat("Residuals:\n")
print(summary(residuals(model)))
cat("\n")

# Coefficients (just first 10 and last 5)
cat("Coefficients (showing first 10 and last 5 of 51):\n")
cat(strrep("-", 80), "\n")
coef_table <- coef(summ)
cat("First 10:\n")
print(coef_table[1:min(10, nrow(coef_table)), ], digits=4)
cat("\n...\n\nLast 5:\n")
print(coef_table[(nrow(coef_table)-4):nrow(coef_table), ], digits=4)
cat(strrep("-", 80), "\n")

# Key statistics
cat("\n")
cat(strrep("=", 80), "\n")
cat("KEY STATISTICS\n")
cat(strrep("=", 80), "\n")
cat(sprintf("R-squared:              %20.10f\n", summ$r.squared))
cat(sprintf("Adjusted R-squared:     %20.10f\n", summ$adj.r.squared))
cat(sprintf("Residual Std Error:     %20.10f\n", summ$sigma))
cat(sprintf("F-statistic:            %20.4f\n", summ$fstatistic[1]))
cat(sprintf("Degrees of Freedom:     %20s (residual)\n", format(summ$df[2], big.mark=",")))

cat("\n")
cat(strrep("=", 80), "\n")
cat("PERFORMANCE\n")
cat(strrep("=", 80), "\n")
cat(sprintf("Data load time:         %20.2f seconds\n", load_time))
cat(sprintf("Model fit time:         %20.2f seconds\n", runtime))
cat(sprintf("Total time:             %20.2f seconds\n", load_time + runtime))
cat(sprintf("Observations/second:    %20.1f\n", nrow(data) / runtime))
cat(sprintf("Memory used:            %20.2f GB\n", object.size(model) / 1e9))

cat("\n")
cat(strrep("=", 80), "\n")
cat("BENCHMARK COMPLETE\n")
cat(strrep("=", 80), "\n\n")

cat("R handled this large dataset, but it was slow.\n")
cat("Let's see how Python GPU performs...\n\n")