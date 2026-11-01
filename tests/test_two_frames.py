from pathlib import Path
import sys

# Project root ko Python path mein add karo
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_imerg_file, crop_region


# ============================================================
# FILES
# ============================================================

FILE_1 = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "3B-HHR.MS.MRG.3IMERG."
    "20240815-S000000-E002959.0000.V07B.HDF5"
)

FILE_2 = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "3B-HHR.MS.MRG.3IMERG."
    "20240815-S003000-E005959.0030.V07B.HDF5"
)


# ============================================================
# REGION
# ============================================================

LAT_MIN = 28.0
LAT_MAX = 30.0

LON_MIN = 76.0
LON_MAX = 79.0


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("IMERG TWO-FRAME VALIDATION")
print("=" * 70)

print("\nLoading frame 1...")
data1 = load_imerg_file(FILE_1)

print("Loading frame 2...")
data2 = load_imerg_file(FILE_2)


# ============================================================
# CROP
# ============================================================

print("\nCropping region...")

region1 = crop_region(
    data1,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX
)

region2 = crop_region(
    data2,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FRAME INFORMATION")
print("=" * 70)

print("\nFrame 1:")
print("Rainfall shape :", region1["precipitation"].shape)
print("Timestamp      :", data1["time"])

print("\nFrame 2:")
print("Rainfall shape :", region2["precipitation"].shape)
print("Timestamp      :", data2["time"])


# ============================================================
# SPATIAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL ALIGNMENT")
print("=" * 70)

same_lat = (
    region1["lat"].shape == region2["lat"].shape
    and (region1["lat"] == region2["lat"]).all()
)

same_lon = (
    region1["lon"].shape == region2["lon"].shape
    and (region1["lon"] == region2["lon"]).all()
)

print("Latitude identical  :", same_lat)
print("Longitude identical :", same_lon)

print("Latitude shape      :", region1["lat"].shape)
print("Longitude shape     :", region1["lon"].shape)


# ============================================================
# TEMPORAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL CHECK")
print("=" * 70)

t1 = int(data1["time"][0])
t2 = int(data2["time"][0])

difference = t2 - t1

print("Frame 1 timestamp :", t1)
print("Frame 2 timestamp :", t2)
print("Difference        :", difference, "seconds")
print("Difference        :", difference / 60, "minutes")

if difference == 1800:
    print("Temporal spacing  : PASS (30 minutes)")
else:
    print("Temporal spacing  : FAIL")


# ============================================================
# RAINFALL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("RAINFALL STATISTICS")
print("=" * 70)

rain1 = region1["precipitation"]
rain2 = region2["precipitation"]

print("\nFrame 1:")
print("Min  :", rain1.min())
print("Max  :", rain1.max())
print("Mean :", rain1.mean())

print("\nFrame 2:")
print("Min  :", rain2.min())
print("Max  :", rain2.max())
print("Mean :", rain2.mean())


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

checks = {
    "Frame 1 loaded": rain1.size > 0,
    "Frame 2 loaded": rain2.size > 0,
    "Same latitude grid": same_lat,
    "Same longitude grid": same_lon,
    "30-minute spacing": difference == 1800,
    "Same regional shape": rain1.shape == rain2.shape,
}

all_pass = True

for name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"{name:<30} : {status}")

    if not result:
        all_pass = False


print("\n" + "=" * 70)

if all_pass:
    print("ALL CHECKS PASSED")
    print("Two-frame IMERG pipeline is valid.")
else:
    print("VALIDATION FAILED")
    print("Check the failed item above.")

print("=" * 70)