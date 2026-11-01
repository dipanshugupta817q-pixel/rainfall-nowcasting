import earthaccess
import requests
import h5py
import numpy as np
from pathlib import Path

# ============================================================
# IMERG V07B - TRUE SERVER-SIDE SUBSET TEST
#
# One 30-minute granule
# Region: 28-30 N, 76-79 E
# Variable: precipitation
#
# IMPORTANT:
# Global HDF5 should NOT be downloaded.
# NASA server performs the spatial subset.
# ============================================================

OUT_DIR = Path("data/test_regional_imerg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "imerg_direct_subset_20220601_0000.nc4"

# ------------------------------------------------------------
# Exact IMERG V07B granule
# ------------------------------------------------------------

COLLECTION_ID = "C2723754847-GES_DISC"

GRANULE_NAME = (
    "GPM_3IMERGHH.07:"
    "3B-HHR.MS.MRG.3IMERG.20220601-S000000-E002959.0000.V07B.HDF5"
)

# NASA's command-line "encoded data URL" convention
# uses %253A for the colon in the granule identifier.
ENCODED_GRANULE = (
    "GPM_3IMERGHH.07%253A"
    "3B-HHR.MS.MRG.3IMERG.20220601-S000000-E002959.0000.V07B.HDF5"
)

OPENDAP_URL = (
    "https://opendap.earthdata.nasa.gov/"
    f"collections/{COLLECTION_ID}/granules/"
    f"{ENCODED_GRANULE}.dap.nc4"
)

# ------------------------------------------------------------
# IMERG grid
#
# Global grid:
# longitude: -179.95 ... 179.95
# latitude :  -89.95 ...  89.95
# spacing  : 0.1 degree
#
# Required region:
# longitude: 76.0 ... 79.0
# latitude : 28.0 ... 30.0
#
# Actual cell centers:
# longitude: 76.05 ... 78.95  -> 30 cells
# latitude : 28.05 ... 29.95  -> 20 cells
#
# IMERG HDF5 layout previously verified:
# precipitation = (time, longitude, latitude)
# = (1, 3600, 1800)
# ------------------------------------------------------------

LON_START = 2560
LON_END = 2589

LAT_START = 1180
LAT_END = 1199

# Request:
#   time
#   latitude
#   longitude
#   precipitation
#
# DAP4 constraint syntax follows NASA's documented pattern.
CONSTRAINT = (
    "/Grid/time[0:1:0];"
    f"/Grid/lon[{LON_START}:1:{LON_END}];"
    f"/Grid/lat[{LAT_START}:1:{LAT_END}];"
    f"/Grid/precipitation[0:1:0]"
    f"[{LON_START}:1:{LON_END}]"
    f"[{LAT_START}:1:{LAT_END}]"
)

print("=" * 70)
print("IMERG V07B TRUE SERVER-SIDE SUBSET TEST")
print("=" * 70)

print("\nDataset:")
print("GPM IMERG Final V07B Half-Hourly")

print("\nTime:")
print("2022-06-01 00:00 UTC")

print("\nRequested region:")
print("Latitude : 28.0 to 30.0 N")
print("Longitude: 76.0 to 79.0 E")

print("\nExpected precipitation shape:")
print("(1, 30, 20)")

print("\nServer:")
print("NASA GES DISC Cloud OPeNDAP")

# ------------------------------------------------------------
# Login
# ------------------------------------------------------------

print("\n[1/4] Logging into NASA Earthdata...")

earthaccess.login(
    persist=True,
    strategy="interactive"
)

session = earthaccess.get_requests_https_session()

print("[OK] Earthdata authentication successful")

# ------------------------------------------------------------
# Show request
# ------------------------------------------------------------

print("\n[2/4] OPeNDAP endpoint:")
print(OPENDAP_URL)

print("\n[3/4] Server-side constraint:")
print(CONSTRAINT)

print("\nRequesting subset...")
print("NO global HDF5 should be downloaded.")

# ------------------------------------------------------------
# Request
# ------------------------------------------------------------

try:

    response = session.get(
        OPENDAP_URL,
        params={
            "dap4.ce": CONSTRAINT
        },
        timeout=180
    )

except Exception as e:

    print("\n[ERROR] Network/request failure")
    print(type(e).__name__, ":", e)
    raise SystemExit(1)

print("\nHTTP Status:", response.status_code)

size_kb = len(response.content) / 1024

print(
    "Response size:",
    round(size_kb, 2),
    "KB"
)

# ------------------------------------------------------------
# Error handling
# ------------------------------------------------------------

if response.status_code != 200:

    print("\n[FAILED]")
    print("NASA server rejected the subset request.")
    print("\nServer response:")
    print(response.text[:5000])

    raise SystemExit(1)

# ------------------------------------------------------------
# Save result
# ------------------------------------------------------------

print("\n[4/4] Saving server-side subset...")

with open(OUTPUT_FILE, "wb") as f:
    f.write(response.content)

print("\n[SUCCESS]")
print("Subset file:")
print(OUTPUT_FILE)

print(
    "Downloaded size:",
    round(OUTPUT_FILE.stat().st_size / 1024, 2),
    "KB"
)

# ------------------------------------------------------------
# Validate NetCDF4/HDF5 result
# ------------------------------------------------------------

print("\nValidating returned file...")

try:

    with h5py.File(OUTPUT_FILE, "r") as f:

        print("\nTop-level groups:")
        print(list(f.keys()))

        if "Grid" not in f:
            print("\n[WARNING] Grid group not found.")
            print("Returned file structure may be different.")
            raise SystemExit(1)

        grid = f["Grid"]

        print("\nGrid variables:")
        print(list(grid.keys()))

        if "precipitation" not in grid:
            print("\n[WARNING] Grid/precipitation not found.")
            raise SystemExit(1)

        precipitation = grid["precipitation"][:]

        print("\nReturned precipitation shape:")
        print(precipitation.shape)

        if "lat" in grid:
            lat = grid["lat"][:]
            print(
                "Latitude:",
                float(np.min(lat)),
                "to",
                float(np.max(lat))
            )

        if "lon" in grid:
            lon = grid["lon"][:]
            print(
                "Longitude:",
                float(np.min(lon)),
                "to",
                float(np.max(lon))
            )

        print(
            "\nPrecipitation min:",
            float(np.nanmin(precipitation))
        )

        print(
            "Precipitation max:",
            float(np.nanmax(precipitation))
        )

except Exception as e:

    print("\n[VALIDATION ERROR]")
    print(type(e).__name__, ":", e)
    raise SystemExit(1)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print("\nIf the result shows:")
print("  HTTP Status: 200")
print("  Shape approximately: (1, 30, 20)")
print("  File size: KB rather than ~7.7 MB")
print("")
print("then TRUE server-side regional downloading works.")