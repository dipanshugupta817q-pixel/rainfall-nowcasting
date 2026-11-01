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

START_TIME = "20240815"

LAT_MIN = 28.0
LAT_MAX = 30.0

LON_MIN = 76.0
LON_MAX = 79.0

NUM_FRAMES = 12


# ============================================================
# CREATE FILE LIST
# ============================================================

files = sorted(
    DATA_DIR.glob(
        f"3B-HHR.MS.MRG.3IMERG.{START_TIME}-*.V07B.HDF5"
    )
)

# We expect at least the 12 downloaded frames.
files = files[:NUM_FRAMES]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("IMERG 12-FRAME SEQUENCE VALIDATION")
print("=" * 70)

print(f"\nFiles found : {len(files)}")
print(f"Expected    : {NUM_FRAMES}")


if len(files) != NUM_FRAMES:
    raise RuntimeError(
        f"Expected {NUM_FRAMES} files, "
        f"but found {len(files)}."
    )


# ============================================================
# LOAD FRAMES
# ============================================================

rainfall_frames = []
timestamps = []

reference_lat = None
reference_lon = None


for i, file_path in enumerate(files):

    print(
        f"\nLoading frame {i + 1}/{NUM_FRAMES}: "
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

    rainfall = region["precipitation"]

    # Remove the time dimension.
    # (1, 20, 30) -> (20, 30)
    rainfall = rainfall[0]

    rainfall_frames.append(rainfall)

    timestamps.append(int(data["time"][0]))

    # --------------------------------------------------------
    # Spatial reference
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
                f"Latitude grid mismatch at frame {i + 1}"
            )

        if not np.array_equal(
            reference_lon,
            region["lon"]
        ):
            raise ValueError(
                f"Longitude grid mismatch at frame {i + 1}"
            )


# ============================================================
# STACK
# ============================================================

sequence = np.stack(
    rainfall_frames,
    axis=0
)


# ============================================================
# TEMPORAL VALIDATION
# ============================================================

timestamps = np.array(timestamps)

time_differences = np.diff(timestamps)

print("\n" + "=" * 70)
print("TEMPORAL VALIDATION")
print("=" * 70)

print("Timestamps:")
print(timestamps)

print("\nTime differences (seconds):")
print(time_differences)

expected_difference = 30 * 60

temporal_ok = np.all(
    time_differences == expected_difference
)

print(
    "\n30-minute spacing :",
    "PASS" if temporal_ok else "FAIL"
)


# ============================================================
# SPATIAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL VALIDATION")
print("=" * 70)

print("Latitude shape  :", reference_lat.shape)
print("Longitude shape :", reference_lon.shape)

print(
    "Spatial grid    :",
    "PASS"
    if sequence.shape[1:] == (
        len(reference_lat),
        len(reference_lon)
    )
    else "FAIL"
)


# ============================================================
# SEQUENCE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE INFORMATION")
print("=" * 70)

print("Sequence shape :", sequence.shape)
print("Expected shape :", "(12, 20, 30)")

print("\nSequence dtype :", sequence.dtype)

print(
    "Minimum rainfall :",
    np.nanmin(sequence)
)

print(
    "Maximum rainfall :",
    np.nanmax(sequence)
)

print(
    "Mean rainfall    :",
    np.nanmean(sequence)
)

print(
    "NaN count        :",
    np.isnan(sequence).sum()
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

checks = {

    "12 files loaded":
        len(files) == 12,

    "Sequence shape":
        sequence.shape == (12, 20, 30),

    "30-minute spacing":
        temporal_ok,

    "Same latitude grid":
        all(
            np.array_equal(
                reference_lat,
                crop_region(
                    load_imerg_file(f),
                    LAT_MIN,
                    LAT_MAX,
                    LON_MIN,
                    LON_MAX
                )["lat"]
            )
            for f in files
        ),

    "Same longitude grid":
        all(
            np.array_equal(
                reference_lon,
                crop_region(
                    load_imerg_file(f),
                    LAT_MIN,
                    LAT_MAX,
                    LON_MIN,
                    LON_MAX
                )["lon"]
            )
            for f in files
        ),
}


all_pass = True

for name, result in checks.items():

    status = "PASS" if result else "FAIL"

    print(
        f"{name:<30} : {status}"
    )

    if not result:
        all_pass = False


print("\n" + "=" * 70)

if all_pass:

    print("ALL CHECKS PASSED")
    print("12-frame rainfall sequence is VALID.")

else:

    print("VALIDATION FAILED")
    print("Check the failed item above.")

print("=" * 70)