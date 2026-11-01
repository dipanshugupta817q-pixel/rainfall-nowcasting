from pathlib import Path
import numpy as np

DATA_DIR = Path("data/processed/imerg")

files = [
    f for f in DATA_DIR.glob("*.npz")
    if not f.name.startswith("2022")
]

print("=" * 70)
print("             IMERG MISSING-VALUE VALIDATION")
print("=" * 70)

print(f"Historical files: {len(files):,}")

# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

files_with_nan = 0
files_with_inf = 0
files_with_negative = 0

total_pixels = 0
nan_pixels = 0
inf_pixels = 0
negative_pixels = 0

global_min = np.inf
global_max = -np.inf

# Sample every Nth file to keep validation fast
STEP = max(1, len(files) // 1000)

sampled = files[::STEP]

print(f"Files sampled: {len(sampled):,}")
print()

for i, file in enumerate(sampled, 1):

    data = np.load(
        file,
        allow_pickle=False
    )

    precip = data["precipitation"].astype(
        np.float32
    )

    total_pixels += precip.size

    nan_mask = np.isnan(precip)
    inf_mask = np.isinf(precip)
    negative_mask = precip < 0

    n_nan = int(nan_mask.sum())
    n_inf = int(inf_mask.sum())
    n_negative = int(negative_mask.sum())

    nan_pixels += n_nan
    inf_pixels += n_inf
    negative_pixels += n_negative

    if n_nan:
        files_with_nan += 1

    if n_inf:
        files_with_inf += 1

    if n_negative:
        files_with_negative += 1

    finite = precip[np.isfinite(precip)]

    if finite.size:

        global_min = min(
            global_min,
            float(finite.min())
        )

        global_max = max(
            global_max,
            float(finite.max())
        )

    if i % 100 == 0:
        print(
            f"\rChecked {i}/{len(sampled)}",
            end="",
            flush=True
        )

print()
print()

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("=" * 70)

print(f"Total sampled pixels : {total_pixels:,}")

print(
    f"NaN pixels           : "
    f"{nan_pixels:,}"
)

print(
    f"Inf pixels           : "
    f"{inf_pixels:,}"
)

print(
    f"Negative pixels      : "
    f"{negative_pixels:,}"
)

print()

print(
    f"Files containing NaN : "
    f"{files_with_nan:,}"
)

print(
    f"Files containing Inf : "
    f"{files_with_inf:,}"
)

print(
    f"Files with negative  : "
    f"{files_with_negative:,}"
)

print()

print(
    f"Finite minimum       : "
    f"{global_min}"
)

print(
    f"Finite maximum       : "
    f"{global_max}"
)

print("=" * 70)

if (
    inf_pixels == 0
    and negative_pixels == 0
):

    print()
    print(
        "STATUS: NO INVALID INF/NEGATIVE RAINFALL VALUES"
    )

else:

    print()
    print(
        "STATUS: INVALID VALUES REQUIRE HANDLING"
    )

print("=" * 70)