from pathlib import Path
import re
import csv
import numpy as np

from load_data import load_imerg_file, crop_region


# ============================================================
# CONFIG
# ============================================================

RAW_DIR = Path("data/raw")

OUTPUT_FILE = Path(
    "data/processed/rainfall_event_screening.csv"
)

LAT_MIN = 28.0
LAT_MAX = 30.0

LON_MIN = 76.0
LON_MAX = 79.0


# ============================================================
# EVENT THRESHOLDS
# ============================================================

# These are screening thresholds, not final scientific
# warning thresholds.

LIGHT_MAX = 2.5
MODERATE_MAX = 7.5
HEAVY_MAX = 15.0


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_rainfall(max_rain):

    if max_rain < 0.1:
        return "dry"

    elif max_rain < LIGHT_MAX:
        return "light"

    elif max_rain < MODERATE_MAX:
        return "moderate"

    elif max_rain < HEAVY_MAX:
        return "heavy"

    else:
        return "very_heavy"


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path):

    data = load_imerg_file(file_path)

    regional = crop_region(
        data,
        LAT_MIN,
        LAT_MAX,
        LON_MIN,
        LON_MAX
    )

    rainfall = regional["precipitation"]

    # Expected shape:
    # (time, lat, lon)

    rainfall = np.asarray(
        rainfall,
        dtype=np.float32
    )

    valid = rainfall[
        np.isfinite(rainfall)
    ]

    if valid.size == 0:
        return None

    return {
        "file": file_path.name,
        "max_rain": float(np.max(valid)),
        "mean_rain": float(np.mean(valid)),
        "p95_rain": float(np.percentile(valid, 95)),
        "valid_pixels": int(valid.size),
        "category": classify_rainfall(
            float(np.max(valid))
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        RAW_DIR.glob(
            "3B-HHR.MS.MRG.3IMERG.*.V07B.HDF5"
        )
    )

    print("=" * 70)
    print("IMERG RAINFALL EVENT SCREENING")
    print("=" * 70)

    print(f"Files found : {len(files)}")

    if not files:
        print("\nNo IMERG Final V07B files found.")
        return

    results = []

    for i, file_path in enumerate(files, start=1):

        print(
            f"[{i}/{len(files)}] "
            f"{file_path.name}"
        )

        try:

            result = process_file(
                file_path
            )

            if result is not None:

                results.append(result)

        except Exception as e:

            print(
                f"[ERROR] "
                f"{file_path.name}: {e}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "max_rain",
                "mean_rain",
                "p95_rain",
                "valid_pixels",
                "category"
            ]
        )

        writer.writeheader()

        writer.writerows(results)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    categories = {}

    for row in results:

        category = row["category"]

        categories[category] = (
            categories.get(category, 0) + 1
        )

    print("\n" + "=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)

    print(f"Processed : {len(results)}")
    print(f"Output    : {OUTPUT_FILE}")

    print("\nCategories:")

    for category, count in categories.items():

        print(
            f"{category:12s}: {count}"
        )


if __name__ == "__main__":
    main()