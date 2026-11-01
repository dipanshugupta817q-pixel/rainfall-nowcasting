from pathlib import Path
import numpy as np

DATA_DIR = Path("data/processed/imerg")

files = sorted(
    f for f in DATA_DIR.glob("*.npz")
    if not f.name.startswith("2022")
)

print("=" * 70)
print("             IMERG GRID ALIGNMENT VALIDATION")
print("=" * 70)

print(f"Historical files found: {len(files):,}")

# ---------------------------------------------------------
# Select representative files
# ---------------------------------------------------------

sample_files = []

# First 10
sample_files.extend(files[:10])

# Middle 10
mid = len(files) // 2
sample_files.extend(files[mid:mid + 10])

# Last 10
sample_files.extend(files[-10:])

# Remove duplicates while preserving order
sample_files = list(dict.fromkeys(sample_files))

print(f"Files sampled: {len(sample_files)}")

# ---------------------------------------------------------
# Reference
# ---------------------------------------------------------

reference = np.load(
    sample_files[0],
    allow_pickle=False
)

ref_lat = reference["lat"].astype(np.float64)
ref_lon = reference["lon"].astype(np.float64)

print()
print(f"Reference: {sample_files[0].name}")
print(f"Precipitation shape: {reference['precipitation'].shape}")
print(f"Latitude shape: {ref_lat.shape}")
print(f"Longitude shape: {ref_lon.shape}")

print(
    f"Latitude range: "
    f"{ref_lat[0]:.5f} -> {ref_lat[-1]:.5f}"
)

print(
    f"Longitude range: "
    f"{ref_lon[0]:.5f} -> {ref_lon[-1]:.5f}"
)

shape_errors = []
lat_errors = []
lon_errors = []

# ---------------------------------------------------------
# Validate samples
# ---------------------------------------------------------

for i, file in enumerate(sample_files, 1):

    print(
        f"\rChecking {i}/{len(sample_files)}: "
        f"{file.name}",
        end="",
        flush=True
    )

    try:

        data = np.load(
            file,
            allow_pickle=False
        )

        precip = data["precipitation"]
        lat = data["lat"].astype(np.float64)
        lon = data["lon"].astype(np.float64)

        if precip.shape != (1, 20, 30):
            shape_errors.append(
                (file.name, precip.shape)
            )

        if lat.shape != (20,):
            shape_errors.append(
                (file.name, f"lat={lat.shape}")
            )

        if lon.shape != (30,):
            shape_errors.append(
                (file.name, f"lon={lon.shape}")
            )

        if not np.allclose(
            lat,
            ref_lat,
            rtol=0,
            atol=1e-5
        ):
            lat_errors.append(file.name)

        if not np.allclose(
            lon,
            ref_lon,
            rtol=0,
            atol=1e-5
        ):
            lon_errors.append(file.name)

    except Exception as e:

        shape_errors.append(
            (file.name, f"ERROR: {e}")
        )

print()
print()
print("=" * 70)

print(f"Shape errors     : {len(shape_errors)}")
print(f"Latitude errors  : {len(lat_errors)}")
print(f"Longitude errors : {len(lon_errors)}")

print("=" * 70)

if not shape_errors and not lat_errors and not lon_errors:

    print()
    print("STATUS: GRID ALIGNMENT VERIFIED")
    print()
    print(
        "Representative files across the historical dataset "
        "share the same 20 x 30 grid."
    )

else:

    print()
    print("STATUS: GRID ALIGNMENT ERRORS FOUND")

    if shape_errors:
        print("\nShape errors:")
        for x in shape_errors[:10]:
            print(x)

    if lat_errors:
        print("\nLatitude errors:")
        for x in lat_errors[:10]:
            print(x)

    if lon_errors:
        print("\nLongitude errors:")
        for x in lon_errors[:10]:
            print(x)

print("=" * 70)