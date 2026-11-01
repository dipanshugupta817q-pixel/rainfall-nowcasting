from pathlib import Path
import numpy as np

DATA_DIR = Path("data/processed/imerg")

files = [
    f for f in DATA_DIR.glob("*.npz")
    if not f.name.startswith("2022")
]

STEP = max(1, len(files) // 1000)
sampled = files[::STEP]

raw_values = []
log_values = []

print("=" * 70)
print("        IMERG NORMALIZATION VALIDATION")
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

    valid = precip[
        np.isfinite(precip)
    ]

    raw_values.append(valid.ravel())

    # Rainfall-aware transformation
    transformed = np.log1p(
        np.maximum(valid, 0)
    )

    log_values.append(
        transformed.ravel()
    )

    if i % 200 == 0:
        print(
            f"\rProcessed {i}/{len(sampled)}",
            end="",
            flush=True
        )

print()

raw = np.concatenate(raw_values)
log_rain = np.concatenate(log_values)

print()
print("=" * 70)
print("RAW RAINFALL")
print("=" * 70)

print(f"Min : {raw.min():.4f}")
print(f"Max : {raw.max():.4f}")
print(f"Mean: {raw.mean():.4f}")
print(f"Std : {raw.std():.4f}")

print()
print("=" * 70)
print("LOG1P TRANSFORMED")
print("=" * 70)

print(f"Min : {log_rain.min():.4f}")
print(f"Max : {log_rain.max():.4f}")
print(f"Mean: {log_rain.mean():.4f}")
print(f"Std : {log_rain.std():.4f}")

print()
print("=" * 70)
print("ROUND-TRIP CHECK")
print("=" * 70)

# Convert back to rainfall
recovered = np.expm1(log_rain)

max_error = np.max(
    np.abs(raw - recovered)
)

mean_error = np.mean(
    np.abs(raw - recovered)
)

print(
    f"Maximum reconstruction error : "
    f"{max_error:.10f}"
)

print(
    f"Mean reconstruction error    : "
    f"{mean_error:.10f}"
)

print("=" * 70)

if max_error < 1e-5:
    print("STATUS: LOG1P TRANSFORMATION VERIFIED")
else:
    print("STATUS: CHECK TRANSFORMATION")