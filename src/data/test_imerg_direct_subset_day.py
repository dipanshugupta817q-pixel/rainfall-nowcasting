import earthaccess
import requests
import h5py
from pathlib import Path
from urllib.parse import quote
from datetime import datetime

# ============================================================
# IMERG V07B - FULL DAY SERVER-SIDE SUBSET TEST
#
# NASA discovers the REAL 48 granules.
# No manual filename construction.
#
# Date: 2022-06-01
# Region: 28-30 N, 76-79 E
# ============================================================

DATE = "2022-06-01"

OUT_DIR = Path("data/test_regional_imerg/day_20220601")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_ID = "C2723754847-GES_DISC"

# Verified from previous successful test
LON_START = 2560
LON_END = 2589

LAT_START = 1180
LAT_END = 1199

print("=" * 70)
print("IMERG V07B - 48 FRAME SERVER-SIDE SUBSET TEST")
print("=" * 70)

print("\nTarget:")
print("Date      :", DATE)
print("Latitude  : 28-30 N")
print("Longitude : 76-79 E")
print("Grid      : 30 longitude x 20 latitude")
print("Expected  : 48 frames")

# ------------------------------------------------------------
# Earthdata authentication
# ------------------------------------------------------------

print("\n[1] Logging into NASA Earthdata...")

earthaccess.login(
    persist=True,
    strategy="interactive"
)

session = earthaccess.get_requests_https_session()

print("[OK] Authentication successful")

# ------------------------------------------------------------
# SEARCH NASA FOR THE ACTUAL 48 GRANULES
# ------------------------------------------------------------

print("\n[2] Searching NASA for actual IMERG granules...")

results = earthaccess.search_data(
    short_name="GPM_3IMERGHH",
    version="07",
    temporal=(
        f"{DATE}T00:00:00",
        f"{DATE}T23:59:59"
    )
)

print("Granules found:", len(results))

if len(results) != 48:
    print("\n[WARNING]")
    print("Expected 48 granules.")
    print("NASA returned:", len(results))

    if len(results) == 0:
        raise SystemExit("No granules found.")

print("\nActual NASA granules:")
for i, r in enumerate(results[:48], 1):
    links = r.data_links()

    if links:
        print(
            f"{i:02d}/48:",
            Path(links[0]).name
        )

# ------------------------------------------------------------
# Process actual NASA URLs
# ------------------------------------------------------------

success = 0
failed = 0
total_kb = 0.0

for i, result in enumerate(results[:48], 1):

    try:

        links = result.data_links()

        if not links:
            print(f"\n[{i:02d}/48] FAILED - no data URL")
            failed += 1
            continue

        original_url = links[0]

        # ----------------------------------------------------
        # Extract the actual filename returned by NASA
        # ----------------------------------------------------

        filename = original_url.rstrip("/").split("/")[-1]

        # ----------------------------------------------------
        # Build the Cloud OPeNDAP granule URL
        #
        # NASA's encoded-data URL uses %253A for the colon
        # between collection/product and granule filename.
        # ----------------------------------------------------

        encoded_granule = (
            "GPM_3IMERGHH.07%253A"
            + filename
        )

        opendap_url = (
            "https://opendap.earthdata.nasa.gov/"
            f"collections/{COLLECTION_ID}/granules/"
            f"{encoded_granule}.dap.nc4"
        )

        # ----------------------------------------------------
        # Server-side subset
        # ----------------------------------------------------

        constraint = (
            "/Grid/time[0:1:0];"
            f"/Grid/lon[{LON_START}:1:{LON_END}];"
            f"/Grid/lat[{LAT_START}:1:{LAT_END}];"
            f"/Grid/precipitation[0:1:0]"
            f"[{LON_START}:1:{LON_END}]"
            f"[{LAT_START}:1:{LAT_END}]"
        )

        output_name = (
            filename.replace(".HDF5", "") +
            "_regional.nc4"
        )

        output = OUT_DIR / output_name

        if output.exists():

            with h5py.File(output, "r") as f:
                shape = f["Grid"]["precipitation"].shape

            if shape == (1, 30, 20):
                size_kb = output.stat().st_size / 1024

                print(
                    f"[{i:02d}/48] EXISTS | "
                    f"{size_kb:.2f} KB"
                )

                success += 1
                total_kb += size_kb
                continue

            else:
                output.unlink()

        # ----------------------------------------------------
        # Request
        # ----------------------------------------------------

        print(
            f"[{i:02d}/48] "
            f"{filename} -> requesting..."
        )

        response = session.get(
            opendap_url,
            params={
                "dap4.ce": constraint
            },
            timeout=180
        )

        if response.status_code != 200:

            print(
                f"    FAILED HTTP {response.status_code}"
            )

            failed += 1
            continue

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        with open(output, "wb") as f:
            f.write(response.content)

        size_kb = output.stat().st_size / 1024

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        with h5py.File(output, "r") as f:

            if "Grid" not in f:
                raise RuntimeError("Grid group missing")

            if "precipitation" not in f["Grid"]:
                raise RuntimeError(
                    "Grid/precipitation missing"
                )

            shape = f["Grid"]["precipitation"].shape

        if shape != (1, 30, 20):

            print(
                f"    FAILED shape={shape}"
            )

            output.unlink()
            failed += 1
            continue

        success += 1
        total_kb += size_kb

        print(
            f"    OK | "
            f"{size_kb:.2f} KB | "
            f"shape={shape}"
        )

    except Exception as e:

        print(
            f"    ERROR: "
            f"{type(e).__name__}: {e}"
        )

        if output.exists():
            output.unlink()

        failed += 1

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DAY TEST SUMMARY")
print("=" * 70)

print(f"NASA granules found : {len(results)}")
print(f"Successful subsets  : {success}")
print(f"Failed subsets      : {failed}")
print(
    f"Regional data size  : "
    f"{total_kb / 1024:.2f} MB"
)

if success == 48:

    print("\n[SUCCESS]")
    print("48/48 server-side regional subsets downloaded.")
    print("NO global HDF5 files were downloaded.")
    print("")
    print("This method is ready for bulk acquisition testing.")

else:

    print("\n[NOT READY]")
    print(
        "Do NOT replace the main downloader yet."
    )