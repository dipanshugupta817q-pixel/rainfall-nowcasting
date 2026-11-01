from pathlib import Path
import numpy as np

DATA_DIR = Path("data/processed/imerg")

files = [
    f for f in DATA_DIR.glob("*.npz")
    if not f.name.startswith("2022")
]

# Sample approximately 2,000 frames
STEP = max(1, len(files) // 2000)
sampled = files[::STEP]

values = []

print("=" * 70)
print("          IMERG RAINFALL DISTRIBUTION ANALYSIS")
print("=" * 70)

print(f"Historical files : {len(files):,}")
print(f"Sampled files    : {len(sampled):,}")
print()

for i, file in enumerate(sampled, 1):

    data = np.load(
        file,
        allow_pickle=False
    )

    precip = data["precipitation"].astype(
        np.float32
    )

    finite = precip[
        np.isfinite(precip)
    ]

    values.append(finite.ravel())

    if i % 200 == 0:
        print(
            f"\rProcessed {i}/{len(sampled)}",
            end="",
            flush=True
        )

print()

values = np.concatenate(values)

# ---------------------------------------------------------
# Distribution
# ---------------------------------------------------------

percentiles = [
    0,
    25,
    50,
    75,
    90,
    95,
    99,
    99.5,
    99.9,
    99.99,
    100,
]

print()
print("=" * 70)
print("PERCENTILES")
print("=" * 70)

for p in percentiles:

    value = np.percentile(
        values,
        p
    )

    print(
        f"P{p:<6}: {value:.4f} mm/hr"
    )

print("=" * 70)

# ---------------------------------------------------------
# Threshold counts
# ---------------------------------------------------------

thresholds = [
    0,
    0.1,
    1,
    2.5,
    5,
    10,
    20,
    30,
    40,
    50,
]

print()
print("THRESHOLD COUNTS")
print("=" * 70)

total = len(values)

for threshold in thresholds:

    count = np.sum(
        values >= threshold
    )

    percentage = (
        count / total * 100
    )

    print(
        f">= {threshold:5.1f} mm/hr : "
        f"{count:10,} pixels "
        f"({percentage:.4f}%)"
    )

print("=" * 70)

# ---------------------------------------------------------
# Extreme values
# ---------------------------------------------------------

print()
print(
    f"Total finite pixels analysed: {total:,}"
)

print(
    f"Maximum observed: "
    f"{values.max():.4f} mm/hr"
)

print("=" * 70)