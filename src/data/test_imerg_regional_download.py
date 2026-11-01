import earthaccess
import h5py
import numpy as np
from pathlib import Path

# ============================================================
# TEST ONLY — IMERG REGIONAL CROP
# One 30-minute frame
# ============================================================

OUT_DIR = Path("data/test_regional_imerg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2022-06-01T00:00:00"
END   = "2022-06-01T00:29:59"

print("=" * 60)
print("IMERG REGIONAL DOWNLOAD TEST")
print("=" * 60)

print("\nLogging into NASA Earthdata...")

auth = earthaccess.login(
    persist=True,
    strategy="interactive"
)

print("[OK] Earthdata authentication")

print("\nSearching for ONE IMERG V07 granule...")

results = earthaccess.search_data(
    short_name="GPM_3IMERGHH",
    version="07",
    temporal=(START, END)
)

print("Granules found:", len(results))

if len(results) == 0:
    raise RuntimeError("No IMERG granule found.")

granule = results[0]

print("\nGranule:")
print(granule)

print("\nDownloading ONE global granule temporarily...")

temp_dir = Path("data/test_regional_imerg/temp")
temp_dir.mkdir(parents=True, exist_ok=True)

files = earthaccess.download(
    [granule],
    local_path=str(temp_dir)
)

if not files:
    raise RuntimeError("Download failed.")

hdf_file = Path(files[0])

print("\nDownloaded global file:")
print(hdf_file)
print(
    "Global file size:",
    round(hdf_file.stat().st_size / (1024 * 1024), 2),
    "MB"
)

print("\nReading only Meerut region...")

LAT_MIN = 28.0
LAT_MAX = 30.0
LON_MIN = 76.0
LON_MAX = 79.0

with h5py.File(hdf_file, "r") as f:

    grid = f["Grid"]

    precip = grid["precipitation"][:]
    lat = grid["lat"][:]
    lon = grid["lon"][:]

    print("Global precipitation shape:", precip.shape)

    lat_idx = np.where(
        (lat >= LAT_MIN) &
        (lat <= LAT_MAX)
    )[0]

    lon_idx = np.where(
        (lon >= LON_MIN) &
        (lon <= LON_MAX)
    )[0]

    regional = precip[
        :,
        lon_idx,
        :
    ]

    regional = regional[
        :,
        :,
        lat_idx
    ]

    # Convert to time, latitude, longitude
    regional = np.transpose(
        regional,
        (0, 2, 1)
    )

    regional = regional.astype(np.float32)

    regional[regional <= -9990] = np.nan

    print("Regional shape:", regional.shape)
    print(
        "Latitude:",
        float(lat[lat_idx].min()),
        "to",
        float(lat[lat_idx].max())
    )
    print(
        "Longitude:",
        float(lon[lon_idx].min()),
        "to",
        float(lon[lon_idx].max())
    )

    output = OUT_DIR / "imerg_test_20220601_0000.npz"

    np.savez_compressed(
        output,
        precipitation=regional,
        lat=lat[lat_idx],
        lon=lon[lon_idx],
        time=grid["time"][:]
    )

print("\n[SUCCESS]")
print("Regional file:", output)
print(
    "Regional file size:",
    round(output.stat().st_size / 1024, 2),
    "KB"
)

print("\nDeleting temporary global HDF5...")

hdf_file.unlink()

print("[OK] Temporary global file deleted.")

print("\nTEST COMPLETE")