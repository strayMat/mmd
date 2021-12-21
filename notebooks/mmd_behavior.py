# %%
import pandas as pd
from mmd.demo_utils import generate_gaussian_blobs
from mmd.mmd import mmd_rbf, mmd_rbf_sigma_
import numpy as np
import matplotlib.pyplot as plt

## %% [markdown]
# Verify that the MMD is indeed converging toward zero when the two populations converge to the same distribution.

# Here the Overlap parameter controls the half distance between the two distributions means. The scale of the gaussian are identical for both population.
# %%
seed = 0
N = 1000

overlaps = [1e-4, 0.001, 0.01, 0.1, 1, 10]
mmds = []
mmds_sigma_1 = []

for overlap in overlaps:
    X_control, X_treated = generate_gaussian_blobs(N=N, overlap=overlap, seed=seed)
    mmds.append(mmd_rbf(X_control, X_treated))
    mmds_sigma_1.append(mmd_rbf_sigma_(X_control, X_treated, sigma=1))

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(overlaps, mmds, label="MMD with median heuristic")
ax.plot(overlaps, mmds_sigma_1, label="MMD with sigma=1")
ax.set_xscale("log")
ax.set_xlabel("Overlap (log scale)")
ax.set_ylabel("MMD value")
ax.set_title("MMD as a function of the overlap")
plt.legend(loc="upper left")

plt.plot()

# %%
results = pd.DataFrame({"overlap": overlaps, "mmd": mmds, "mmd_sigma_1": mmds_sigma_1})
print(results)

# %%

generate_gaussian_blobs()
