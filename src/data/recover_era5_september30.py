from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr


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

TARGET_LAT = np.arange(
    28.05,
    29.95 + 0.001,
    0.1,
    dtype=np.float32
)

TARGET_LON = np.arange(
    76.05,
    78.95 + 0.001,
    0.1,
    dtype=np.float32
)

TARGET_SHAPE = (48, 7, 20, 30)

YEARS = [2021, 2022, 2023, 2024, 2025]


# ============================================================
# RECOVER ONE SEPTEMBER 30
# ============================================================

def recover_day(year):

    date_str = f"{year}0930"

    current_file = (
        INPUT_ROOT
        / str(year)
        / f"{date_str}.nc"
    )

    output_file = (
        OUTPUT_ROOT
        / f"{date_str}.npz"
    )

    print("\n" + "=" * 60)
    print(f"RECOVERING: {date_str}")
    print("=" * 60)

    # --------------------------------------------------------
    # Safety: never overwrite existing file
    # --------------------------------------------------------

    if output_file.exists():
        print(f"[SKIP] Already exists: {output_file}")
        return True

    if not current_file.exists():
        print(f"[ERROR] Missing source:")
        print(current_file)
        return False

    print(f"Source: {current_file}")

    # --------------------------------------------------------
    # Open September 30 only
    # --------------------------------------------------------

    ds = xr.open_dataset(current_file)

    try:

        ds = ds[VARIABLES]

        print(
            "Original dimensions:",
            dict(ds.sizes)
        )

        # ----------------------------------------------------
        # Target = 48 frames
        #
        # 00:00 ... 23:00 are actual/interpolated hourly data
        # 23:30 has no Oct 1 00:00 observation available.
        #
        # Therefore:
        # 23:30 = 23:00 carry-forward
        # ----------------------------------------------------

        hourly_times = ds.valid_time.values

        if len(hourly_times) != 24:
            raise RuntimeError(
                f"Expected 24 hourly frames, got {len(hourly_times)}"
            )

        day_start = np.datetime64(
            f"{year}-09-30T00:00:00"
        )

        target_times = (
            day_start
            + np.arange(48).astype("timedelta64[m]") * 30
        )

        # ----------------------------------------------------
        # First 47 frames:
        # 00:00 -> 23:00
        #
        # Linear interpolation between hourly observations
        # ----------------------------------------------------

        first_47_times = target_times[:47]

        aligned = ds.interp(
            valid_time=first_47_times,
            method="linear"
        )

        # ----------------------------------------------------
        # Spatial interpolation
        # ----------------------------------------------------

        aligned = aligned.interp(
            latitude=TARGET_LAT,
            longitude=TARGET_LON,
            method="linear"
        )

        arrays = []

        for var in VARIABLES:

            arr = aligned[var].values.astype(
                np.float32
            )

            if arr.shape != (47, 20, 30):
                raise RuntimeError(
                    f"{var} wrong shape: {arr.shape}"
                )

            arrays.append(arr)

        data_47 = np.stack(
            arrays,
            axis=1
        )

        # ----------------------------------------------------
        # 48th frame = 23:00 carry-forward
        # ----------------------------------------------------

        last_frame = data_47[-1:, :, :, :]

        data = np.concatenate(
            [data_47, last_frame],
            axis=0
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print("Final shape:", data.shape)

        if data.shape != TARGET_SHAPE:
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
        # Verify final two frames
        # ----------------------------------------------------

        same_count = np.sum(
            data[-1] == data[-2]
        )

        print(
            "23:30 carry-forward pixels:",
            same_count,
            "/",
            7 * 20 * 30
        )

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

        ds.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ERA5 — SEPTEMBER 30 BOUNDARY RECOVERY")
    print("=" * 60)

    success = 0
    failed = 0

    for year in YEARS:

        try:

            if recover_day(year):
                success += 1
            else:
                failed += 1

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] {year}0930: {e}"
            )

    print("\n")
    print("=" * 60)
    print("RECOVERY COMPLETE")
    print("=" * 60)

    print("Expected : 5")
    print("Success  :", success)
    print("Failed   :", failed)

    if failed == 0:
        print(
            "STATUS: ALL 5 BOUNDARY FILES RECOVERED"
        )
    else:
        print(
            "STATUS: RECOVERY INCOMPLETE"
        )


if __name__ == "__main__":
    main()