from pathlib import Path
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "era5"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "era5_aligned"

VARIABLES = [
    "t2m",
    "d2m",
    "u10",
    "v10",
    "sp",
    "msl",
    "tcc",
]

TARGET_LAT = np.arange(28.05, 29.95 + 0.001, 0.1, dtype=np.float32)
TARGET_LON = np.arange(76.05, 78.95 + 0.001, 0.1, dtype=np.float32)

EXPECTED_SHAPE = (48, 7, 20, 30)

START_YEAR = 2021
END_YEAR = 2025

MONTHS = [6, 7, 8, 9]


# ============================================================
# HELPERS
# ============================================================

def get_daily_file(date_obj):
    year = date_obj.year
    date_str = date_obj.strftime("%Y%m%d")

    return INPUT_ROOT / str(year) / f"{date_str}.nc"


def process_one_day(date_obj):

    date_str = date_obj.strftime("%Y%m%d")

    current_file = get_daily_file(date_obj)
    next_file = get_daily_file(date_obj + timedelta(days=1))

    output_file = OUTPUT_ROOT / f"{date_str}.npz"

    # --------------------------------------------------------
    # Skip if already processed
    # --------------------------------------------------------

    if output_file.exists():
        print(f"[SKIP] {date_str}")
        return True

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not current_file.exists():
        print(f"[MISSING CURRENT] {current_file}")
        return False

    if not next_file.exists():
        print(f"[MISSING NEXT] {next_file}")
        return False

    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {date_str}")
    print(f"{'=' * 60}")

    # --------------------------------------------------------
    # Open current + next day
    # --------------------------------------------------------

    ds1 = xr.open_dataset(current_file)
    ds2 = xr.open_dataset(next_file)

    try:

        ds1 = ds1[VARIABLES]
        ds2 = ds2[VARIABLES]

        # ----------------------------------------------------
        # Concatenate time
        # ----------------------------------------------------

        ds = xr.concat([ds1, ds2], dim="valid_time")

        # Remove any duplicate timestamps
        _, unique_indices = np.unique(
            ds.valid_time.values,
            return_index=True
        )

        unique_indices = np.sort(unique_indices)

        ds = ds.isel(valid_time=unique_indices)

        # ----------------------------------------------------
        # Target 30-minute timestamps
        # ----------------------------------------------------

        day_start = np.datetime64(
            date_obj.strftime("%Y-%m-%dT00:00:00")
        )

        target_times = (
            day_start
            + np.arange(48).astype("timedelta64[m]") * 30
        )

        # ----------------------------------------------------
        # Temporal interpolation
        #
        # ERA5 = hourly
        # Target = 30-minute
        # ----------------------------------------------------

        ds = ds.interp(
            valid_time=target_times,
            method="linear"
        )

        # ----------------------------------------------------
        # Spatial interpolation
        #
        # ERA5 0.25°
        # Target 0.1°
        # ----------------------------------------------------

        ds = ds.interp(
            latitude=TARGET_LAT,
            longitude=TARGET_LON,
            method="linear"
        )

        # ----------------------------------------------------
        # Extract array
        # ----------------------------------------------------

        arrays = []

        for var in VARIABLES:

            arr = ds[var].values.astype(np.float32)

            if arr.shape != (48, 20, 30):
                raise RuntimeError(
                    f"{var} has wrong shape: {arr.shape}"
                )

            arrays.append(arr)

        data = np.stack(arrays, axis=1)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print("Output shape:", data.shape)

        if data.shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"Wrong final shape: {data.shape}"
            )

        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()

        print("NaN:", nan_count)
        print("Inf:", inf_count)

        if nan_count > 0:
            raise RuntimeError("NaN detected")

        if inf_count > 0:
            raise RuntimeError("Inf detected")

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez_compressed(
            output_file,
            data=data,
            time=target_times.astype(str),
            latitude=TARGET_LAT,
            longitude=TARGET_LON,
            variables=np.array(VARIABLES)
        )

        print(f"[SAVED] {output_file}")

        return True

    finally:

        ds1.close()
        ds2.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ERA5 — FULL ALIGNMENT")
    print("=" * 60)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    total_days = 0
    success = 0
    failed = 0

    for year in range(START_YEAR, END_YEAR + 1):

        for month in MONTHS:

            if month == 6:
                start_day = 1
            else:
                start_day = 1

            # June 1 → September 30
            current = datetime(year, month, 1)

            if month == 9:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)

            while current < next_month:

                # Only June–September
                if current.month not in MONTHS:
                    break

                total_days += 1

                try:

                    if process_one_day(current):

                        success += 1

                    else:

                        failed += 1

                except Exception as e:

                    failed += 1

                    print(
                        f"[ERROR] {current.strftime('%Y%m%d')}: {e}"
                    )

                current += timedelta(days=1)

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("ERA5 ALIGNMENT COMPLETE")
    print("=" * 60)

    print(f"Total days : {total_days}")
    print(f"Success    : {success}")
    print(f"Failed     : {failed}")

    print("=" * 60)

    if failed == 0:
        print("STATUS: ALL ERA5 DAYS ALIGNED SUCCESSFULLY")
    else:
        print("STATUS: SOME ERA5 DAYS FAILED")


if __name__ == "__main__":
    main()