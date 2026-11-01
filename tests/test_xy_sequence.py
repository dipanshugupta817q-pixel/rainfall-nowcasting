from pathlib import Path
import sys
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_imerg_file, crop_region


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATE = "20240815"

LAT_MIN = 28.0
LAT_MAX = 30.0

LON_MIN = 76.0
LON_MAX = 79.0

INPUT_FRAMES = 12
TARGET_FRAMES = 12

TOTAL_FRAMES = INPUT_FRAMES + TARGET_FRAMES


# ============================================================
# FIND ALL 24 FILES
# ============================================================

files = sorted(
    DATA_DIR.glob(
        f"3B-HHR.MS.MRG.3IMERG.{DATE}-*.V07B.HDF5"
    )
)

print("=" * 70)
print("IMERG 24-FRAME X → Y VALIDATION")
print("=" * 70)

print(f"\nFiles found : {len(files)}")
print(f"Expected    : {TOTAL_FRAMES}")

if len(files) != TOTAL_FRAMES:
    raise RuntimeError(
        f"Expected {TOTAL_FRAMES} files, "
        f"but found {len(files)}."
    )


# ============================================================
# LOAD ALL 24 FRAMES
# ============================================================

rainfall_frames = []
timestamps = []

reference_lat = None
reference_lon = None


for i, file_path in enumerate(files):

    print(
        f"Loading frame {i + 1:02d}/{TOTAL_FRAMES}: "
        f"{file_path.name}"
    )

    data = load_imerg_file(file_path)

    region = crop_region(
        data,
        LAT_MIN,
        LAT_MAX,
        LON_MIN,
        LON_MAX
    )

    rainfall = region["precipitation"][0]

    rainfall_frames.append(rainfall)

    timestamps.append(
        int(data["time"][0])
    )

    # --------------------------------------------------------
    # GRID CHECK
    # --------------------------------------------------------

    if reference_lat is None:

        reference_lat = region["lat"].copy()
        reference_lon = region["lon"].copy()

    else:

        if not np.array_equal(
            reference_lat,
            region["lat"]
        ):
            raise ValueError(
                f"Latitude mismatch at frame {i + 1}"
            )

        if not np.array_equal(
            reference_lon,
            region["lon"]
        ):
            raise ValueError(
                f"Longitude mismatch at frame {i + 1}"
            )


# ============================================================
# STACK ALL FRAMES
# ============================================================

all_frames = np.stack(
    rainfall_frames,
    axis=0
)

timestamps = np.array(timestamps)


# ============================================================
# TEMPORAL VALIDATION
# ============================================================

time_differences = np.diff(timestamps)

expected_interval = 30 * 60

temporal_ok = np.all(
    time_differences == expected_interval
)


# ============================================================
# CREATE X AND Y
# ============================================================

X = all_frames[
    :INPUT_FRAMES
]

Y = all_frames[
    INPUT_FRAMES:
]


X_times = timestamps[
    :INPUT_FRAMES
]

Y_times = timestamps[
    INPUT_FRAMES:
]


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nAll frames:")
print("Shape :", all_frames.shape)

print("\nInput X:")
print("Shape :", X.shape)

print("\nTarget Y:")
print("Shape :", Y.shape)


# ============================================================
# TIMESTAMP INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TIMESTAMP SPLIT")
print("=" * 70)

print("\nInput X timestamps:")

for i, timestamp in enumerate(X_times):

    print(
        f"  X[{i:02d}] : {timestamp}"
    )


print("\nTarget Y timestamps:")

for i, timestamp in enumerate(Y_times):

    print(
        f"  Y[{i:02d}] : {timestamp}"
    )


# ============================================================
# X / Y TEMPORAL SEPARATION
# ============================================================

xy_gap = Y_times[0] - X_times[-1]

print("\n" + "=" * 70)
print("X → Y TEMPORAL RELATIONSHIP")
print("=" * 70)

print(
    "Last X timestamp  :",
    X_times[-1]
)

print(
    "First Y timestamp :",
    Y_times[0]
)

print(
    "X → Y gap         :",
    xy_gap,
    "seconds"
)

print(
    "X → Y gap         :",
    xy_gap / 60,
    "minutes"
)


# ============================================================
# RAINFALL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("RAINFALL STATISTICS")
print("=" * 70)

print("\nX:")
print(
    "Min  :",
    np.nanmin(X)
)
print(
    "Max  :",
    np.nanmax(X)
)
print(
    "Mean :",
    np.nanmean(X)
)

print("\nY:")
print(
    "Min  :",
    np.nanmin(Y)
)
print(
    "Max  :",
    np.nanmax(Y)
)
print(
    "Mean :",
    np.nanmean(Y)
)

print("\nNaN values:")
print("X :", np.isnan(X).sum())
print("Y :", np.isnan(Y).sum())


# ============================================================
# SPATIAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL GRID")
print("=" * 70)

print(
    "Latitude cells  :",
    len(reference_lat)
)

print(
    "Longitude cells :",
    len(reference_lon)
)

print(
    "Grid shape      :",
    (
        len(reference_lat),
        len(reference_lon)
    )
)


# ============================================================
# FINAL CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


checks = {

    "24 files present":
        len(files) == 24,

    "All frames loaded":
        all_frames.shape == (24, 20, 30),

    "30-minute spacing":
        temporal_ok,

    "X shape":
        X.shape == (12, 20, 30),

    "Y shape":
        Y.shape == (12, 20, 30),

    "X/Y spatial dimensions match":
        X.shape[1:] == Y.shape[1:],

    "No X/Y temporal overlap":
        X_times[-1] < Y_times[0],

    "Correct X/Y temporal gap":
        xy_gap == expected_interval,

    "No NaN in X":
        not np.isnan(X).any(),

    "No NaN in Y":
        not np.isnan(Y).any(),
}


all_pass = True


for name, result in checks.items():

    status = "PASS" if result else "FAIL"

    print(
        f"{name:<35} : {status}"
    )

    if not result:
        all_pass = False


print("\n" + "=" * 70)

if all_pass:

    print("ALL CHECKS PASSED")
    print()
    print("X → Y SUPERVISED SEQUENCE IS VALID.")
    print()
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

else:

    print("VALIDATION FAILED")
    print("Check the failed item above.")

print("=" * 70)