# Test Fixtures

This directory will contain R-generated test data for validation.

Each test case will include:
- Input data (X, y, weights, etc.)
- Expected output from R (coefficients, standard errors, etc.)
- Edge cases (rank deficiency, perfect separation, etc.)

Fixtures will be generated using R scripts in `tests/generate_fixtures.R`.
