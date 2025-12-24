import torch

assert torch.backends.mps.is_available()

device = torch.device("mps")

X = torch.randn(50000, 64, device=device, dtype=torch.float32)
y = torch.randn(50000, 1, device=device, dtype=torch.float32)

# Normal equations
XtX = X.T @ X
Xty = X.T @ y

print("XtX device:", XtX.device)

# Cholesky
L = torch.linalg.cholesky(XtX)
print("L device:", L.device)

# First solve: L z = Xty
z = torch.linalg.solve_triangular(
    L,
    Xty,
    upper=False
)

print("z device:", z.device)

# Second solve: L.T beta = z
beta = torch.linalg.solve_triangular(
    L.transpose(-1, -2),
    z,
    upper=True
)

print("beta device:", beta.device)
