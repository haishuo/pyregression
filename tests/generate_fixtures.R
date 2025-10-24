#!/usr/bin/env Rscript
# Generate test fixtures for PyRegression validation
# 
# This script creates various test cases and saves R's lm() results
# as JSON files that Python can read for validation.

library(jsonlite)

# Create output directory
output_dir <- "tests/fixtures"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("Generating R validation fixtures...\n")
cat("Output directory:", output_dir, "\n\n")

# Helper function to extract lm results
extract_lm_results <- function(fit, X, y) {
  sum_fit <- summary(fit)
  
  list(
    # Input data
    X = unname(as.matrix(X)),
    y = as.numeric(y),
    
    # Coefficients
    coefficients = as.numeric(coef(fit)),
    
    # Residuals and fitted values
    residuals = as.numeric(residuals(fit)),
    fitted_values = as.numeric(fitted(fit)),
    
    # Standard errors
    std_errors = as.numeric(sum_fit$coefficients[, "Std. Error"]),
    
    # t-statistics and p-values
    t_values = as.numeric(sum_fit$coefficients[, "t value"]),
    p_values = as.numeric(sum_fit$coefficients[, "Pr(>|t|)"]),
    
    # Model statistics
    r_squared = sum_fit$r.squared,
    adj_r_squared = sum_fit$adj.r.squared,
    sigma = sum_fit$sigma,
    df_residual = sum_fit$df[2],
    
    # QR decomposition info
    rank = fit$rank,
    qr_pivot = fit$qr$pivot,
    
    # Variance-covariance matrix
    vcov = unname(vcov(fit)),
    
    # Model info
    n_obs = length(y),
    n_predictors = ncol(X) + 1  # +1 for intercept
  )
}

# Test Case 1: Simple linear regression (well-conditioned)
cat("[1/6] Simple linear regression...\n")
set.seed(42)
n <- 100
X1 <- matrix(rnorm(n * 3), n, 3)
colnames(X1) <- c("X1", "X2", "X3")
beta_true <- c(2.0, -1.5, 0.8)
y1 <- X1 %*% beta_true + rnorm(n, sd = 0.5)

fit1 <- lm(y1 ~ X1)
result1 <- extract_lm_results(fit1, X1, y1)
write_json(result1, file.path(output_dir, "simple_regression.json"), 
           pretty = TRUE, digits = 16)

# Test Case 2: Larger problem
cat("[2/6] Larger regression problem...\n")
set.seed(123)
n <- 500
p <- 10
X2 <- matrix(rnorm(n * p), n, p)
colnames(X2) <- paste0("X", 1:p)
beta_true2 <- rnorm(p)
y2 <- X2 %*% beta_true2 + rnorm(n, sd = 1.0)

fit2 <- lm(y2 ~ X2)
result2 <- extract_lm_results(fit2, X2, y2)
write_json(result2, file.path(output_dir, "large_regression.json"),
           pretty = TRUE, digits = 16)

# Test Case 3: Perfect collinearity (rank deficient)
cat("[3/6] Rank deficient case (perfect collinearity)...\n")
set.seed(456)
n <- 50
X3_base <- matrix(rnorm(n * 2), n, 2)
X3 <- cbind(X3_base, X3_base[, 1] + X3_base[, 2])  # Third column is sum of first two
colnames(X3) <- c("X1", "X2", "X3_redundant")
y3 <- X3_base %*% c(1, -1) + rnorm(n, sd = 0.3)

fit3 <- lm(y3 ~ X3)
result3 <- extract_lm_results(fit3, X3, y3)
result3$note <- "Rank deficient: X3 = X1 + X2"
write_json(result3, file.path(output_dir, "rank_deficient.json"),
           pretty = TRUE, digits = 16)

# Test Case 4: Near collinearity (ill-conditioned)
cat("[4/6] Ill-conditioned case (near collinearity)...\n")
set.seed(789)
n <- 100
X4_base <- matrix(rnorm(n * 2), n, 2)
X4 <- cbind(X4_base, X4_base[, 1] + X4_base[, 2] + rnorm(n, sd = 0.01))  # Nearly collinear
colnames(X4) <- c("X1", "X2", "X3_nearly_redundant")
y4 <- X4_base %*% c(2, -1) + rnorm(n, sd = 0.5)

fit4 <- lm(y4 ~ X4)
result4 <- extract_lm_results(fit4, X4, y4)
result4$note <- "Ill-conditioned: X3 ≈ X1 + X2"
# Compute condition number
X4_centered <- scale(X4, center = TRUE, scale = FALSE)
svd_X4 <- svd(X4_centered)
result4$condition_number <- max(svd_X4$d) / min(svd_X4$d)
write_json(result4, file.path(output_dir, "ill_conditioned.json"),
           pretty = TRUE, digits = 16)

# Test Case 5: Single predictor (simplest case)
cat("[5/6] Single predictor case...\n")
set.seed(321)
n <- 50
X5 <- matrix(rnorm(n), n, 1)
colnames(X5) <- "X1"
y5 <- 3 + 2 * X5 + rnorm(n, sd = 0.8)

fit5 <- lm(y5 ~ X5)
result5 <- extract_lm_results(fit5, X5, y5)
write_json(result5, file.path(output_dir, "single_predictor.json"),
           pretty = TRUE, digits = 16)

# Test Case 6: Intercept only (edge case)
cat("[6/6] Intercept-only model...\n")
set.seed(654)
n <- 30
y6 <- rnorm(n, mean = 5, sd = 2)
X6 <- matrix(numeric(0), n, 0)  # No predictors

fit6 <- lm(y6 ~ 1)
result6 <- list(
  X = X6,
  y = as.numeric(y6),
  coefficients = as.numeric(coef(fit6)),
  residuals = as.numeric(residuals(fit6)),
  fitted_values = as.numeric(fitted(fit6)),
  std_errors = as.numeric(summary(fit6)$coefficients[, "Std. Error"]),
  t_values = as.numeric(summary(fit6)$coefficients[, "t value"]),
  p_values = as.numeric(summary(fit6)$coefficients[, "Pr(>|t|)"]),
  r_squared = 0.0,  # No predictors
  adj_r_squared = 0.0,
  sigma = summary(fit6)$sigma,
  df_residual = summary(fit6)$df[2],
  rank = 1,
  n_obs = n,
  n_predictors = 1,
  note = "Intercept-only model"
)
write_json(result6, file.path(output_dir, "intercept_only.json"),
           pretty = TRUE, digits = 16)

# Create a summary file
cat("\nCreating summary...\n")
summary_info <- list(
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  r_platform = R.version$platform,
  timestamp = Sys.time(),
  test_cases = c(
    "simple_regression.json",
    "large_regression.json", 
    "rank_deficient.json",
    "ill_conditioned.json",
    "single_predictor.json",
    "intercept_only.json"
  ),
  description = list(
    simple_regression = "Well-conditioned, 100 obs, 3 predictors",
    large_regression = "Larger problem, 500 obs, 10 predictors",
    rank_deficient = "Perfect collinearity (rank deficient)",
    ill_conditioned = "Near collinearity (ill-conditioned)",
    single_predictor = "Simplest case, single predictor",
    intercept_only = "Edge case, no predictors"
  )
)

write_json(summary_info, file.path(output_dir, "summary.json"),
           pretty = TRUE)

cat("\n✓ All fixtures generated successfully!\n")
cat("Files created in:", output_dir, "\n")