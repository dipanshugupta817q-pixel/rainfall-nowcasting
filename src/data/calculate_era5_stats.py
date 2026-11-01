from pathlib import Path
import numpy as np


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("data/processed/era5_aligned")
OUTPUT = Path("data/processed/era5_era5_stats.npz")

TRAIN_YEARS = [2021, 2022, 2023]

VARIABLES = [
    "t2m",
    "d2m",
    "u10",
    "v10",
    "sp",
    "msl",
    "tcc",
]


# ============================================================
# COLLECT TRAINING FILES
# ============================================================

files = []

for year in TRAIN_YEARS:

    for month in [6, 7, 8, 9]:

        pattern = f"{year}{month:02d}*.npz"

        files.extend(
            sorted(ROOT.glob(pattern))
        )


files = sorted(files)

print("=" * 60)
print("ERA5 — TRAINING NORMALIZATION STATISTICS")
print("=" * 60)

print("Training years :", TRAIN_YEARS)
print("Files          :", len(files))

if len(files) != 366:
    raise RuntimeError(
        f"Expected 366 training files, got {len(files)}"
    )


# ============================================================
# RUNNING STATISTICS
# ============================================================

sum_values = np.zeros(7, dtype=np.float64)
sum_squared = np.zeros(7, dtype=np.float64)
count_values = np.zeros(7, dtype=np.int64)

global_min = np.full(7, np.inf)
global_max = np.full(7, -np.inf)


# ============================================================
# PROCESS
# ============================================================

for i, file in enumerate(files, 1):

    with np.load(file) as z:

        data = z["data"]

        # data shape:
        # (48, 7, 20, 30)

        if data.shape != (48, 7, 20, 30):
            raise RuntimeError(
                f"Wrong shape in {file.name}: {data.shape}"
            )

        for c in range(7):

            x = data[:, c].astype(np.float64)

            sum_values[c] += x.sum()
            sum_squared[c] += np.square(x).sum()
            count_values[c] += x.size

            global_min[c] = min(
                global_min[c],
                x.min()
            )

            global_max[c] = max(
                global_max[c],
                x.max()
            )

    if i % 50 == 0:
        print(
            f"Processed: {i}/{len(files)}"
        )


# ============================================================
# CALCULATE MEAN / STD
# ============================================================

mean = sum_values / count_values

variance = (
    sum_squared / count_values
    - np.square(mean)
)

# Numerical safety
variance = np.maximum(variance, 0.0)

std = np.sqrt(variance)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 60)
print("ERA5 TRAINING STATISTICS")
print("=" * 60)

for i, var in enumerate(VARIABLES):

    print(f"\n{var}")

    print(f"  Mean : {mean[i]:.8f}")
    print(f"  Std  : {std[i]:.8f}")
    print(f"  Min  : {global_min[i]:.8f}")
    print(f"  Max  : {global_max[i]:.8f}")


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

np.savez(
    OUTPUT,
    variables=np.array(VARIABLES),
    mean=mean.astype(np.float32),
    std=std.astype(np.float32),
    minimum=global_min.astype(np.float32),
    maximum=global_max.astype(np.float32),
)


print()
print("=" * 60)
print(f"Saved: {OUTPUT}")
print("STATUS: ERA5 TRAINING STATISTICS CREATED")
print("=" * 60)