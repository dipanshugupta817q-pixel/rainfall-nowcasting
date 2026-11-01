"""
ERA5 FAST MONTHLY-BATCH DOWNLOADER
==================================

Strategy:
    Monthly CDS request
    -> local monthly NetCDF
    -> extract ONLY missing daily files

Period:
    2021-06-01 -> 2025-09-30

Months:
    June, July, August, September

Region:
    28-30 N
    76-79 E

Variables:
    7 ERA5 single-level variables

Important:
    - Never re-download a complete month
    - Never overwrite existing daily files
    - One CDS request at a time
    - Resumable
    - Live dashboard
    - ETA
"""

import os
import sys
import time
import calendar
from pathlib import Path
from datetime import date
from threading import Lock

import cdsapi
import xarray as xr


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\projects\rainfall-nowcasting"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "era5"
)

CACHE_DIR = (
    OUTPUT_DIR
    / "_monthly_cache"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATE RANGE
# ============================================================

START_DATE = date(
    2021,
    6,
    1
)

END_DATE = date(
    2025,
    9,
    30
)

TARGET_MONTHS = {
    6,
    7,
    8,
    9
}


# ============================================================
# CDS
# ============================================================

DATASET = (
    "reanalysis-era5-single-levels"
)

MAX_RETRIES = 4

RETRY_WAIT = 120


VARIABLES = [

    "2m_temperature",

    "2m_dewpoint_temperature",

    "10m_u_component_of_wind",

    "10m_v_component_of_wind",

    "surface_pressure",

    "mean_sea_level_pressure",

    "total_cloud_cover",

]


# ============================================================
# REGION
# ============================================================

AREA = [
    30.0,   # North
    76.0,   # West
    28.0,   # South
    79.0,   # East
]


# ============================================================
# HOURS
# ============================================================

TIMES = [
    f"{hour:02d}:00"
    for hour in range(24)
]


# ============================================================
# PRINT LOCK
# ============================================================

print_lock = Lock()


def safe_print(text):

    with print_lock:

        print(
            text,
            flush=True
        )


# ============================================================
# FORMAT
# ============================================================

def format_bytes(value):

    if value < 1024:

        return (
            f"{value:.0f} B"
        )

    if value < 1024 ** 2:

        return (
            f"{value / 1024:.1f} KB"
        )

    if value < 1024 ** 3:

        return (
            f"{value / (1024 ** 2):.2f} MB"
        )

    return (
        f"{value / (1024 ** 3):.2f} GB"
    )


def format_speed(
    bytes_per_second
):

    if bytes_per_second <= 0:

        return "0 KB/s"

    kb = (
        bytes_per_second
        / 1024
    )

    if kb < 1024:

        return (
            f"{kb:.1f} KB/s"
        )

    return (
        f"{kb / 1024:.2f} MB/s"
    )


def format_eta(seconds):

    if (
        seconds is None
        or seconds <= 0
    ):

        return "--:--:--"

    seconds = int(seconds)

    hours = (
        seconds // 3600
    )

    minutes = (
        (seconds % 3600)
        // 60
    )

    secs = (
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# MONTH GENERATOR
# ============================================================

def generate_months():

    year = START_DATE.year
    month = START_DATE.month

    while True:

        current = date(
            year,
            month,
            1
        )

        if current > END_DATE:

            break

        if month in TARGET_MONTHS:

            yield (
                year,
                month
            )

        month += 1

        if month > 12:

            month = 1
            year += 1


# ============================================================
# MONTH RANGE
# ============================================================

def month_range(
    year,
    month
):

    first = date(
        year,
        month,
        1
    )

    last_day = calendar.monthrange(
        year,
        month
    )[1]

    last = date(
        year,
        month,
        last_day
    )


    if first < START_DATE:

        first = START_DATE


    if last > END_DATE:

        last = END_DATE


    return (
        first,
        last
    )


# ============================================================
# DAYS IN MONTH
# ============================================================

def days_for_month(
    year,
    month
):

    first, last = month_range(
        year,
        month
    )

    return [
        date.fromordinal(
            first.toordinal() + i
        )
        for i in range(
            (last - first).days + 1
        )
    ]


# ============================================================
# DAILY FILE
# ============================================================

def daily_path(day):

    year_dir = (
        OUTPUT_DIR
        / str(day.year)
    )

    year_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        year_dir
        / f"{day:%Y%m%d}.nc"
    )


# ============================================================
# CHECK DAILY FILE
# ============================================================

def daily_exists(day):

    path = daily_path(day)

    return (
        path.exists()
        and path.stat().st_size > 1000
    )


# ============================================================
# MONTHLY CACHE
# ============================================================

def monthly_path(
    year,
    month
):

    return (
        CACHE_DIR
        / f"{year}{month:02d}.nc"
    )


# ============================================================
# MONTH STATUS
# ============================================================

def month_status(
    year,
    month
):

    days = days_for_month(
        year,
        month
    )

    existing = [
        d
        for d in days
        if daily_exists(d)
    ]

    missing = [
        d
        for d in days
        if not daily_exists(d)
    ]

    return (
        existing,
        missing
    )


# ============================================================
# DOWNLOAD MONTH
# ============================================================

def download_month(
    year,
    month
):

    label = (
        f"{year}-{month:02d}"
    )

    cache_file = monthly_path(
        year,
        month
    )

    temp_file = (
        cache_file
        .with_suffix(".tmp.nc")
    )


    # --------------------------------------------------------
    # CACHE ALREADY EXISTS
    # --------------------------------------------------------

    if (
        cache_file.exists()
        and cache_file.stat().st_size > 1000
    ):

        safe_print(
            f"CACHE HIT {label} | "
            f"{format_bytes(cache_file.stat().st_size)}"
        )

        return {
            "status": "cache",
            "size": cache_file.stat().st_size
        }


    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    total_days = calendar.monthrange(
        year,
        month
    )[1]


    request = {

        "product_type": "reanalysis",

        "variable": VARIABLES,

        "year": str(year),

        "month": f"{month:02d}",

        "day": [
            f"{d:02d}"
            for d in range(
                1,
                total_days + 1
            )
        ],

        "time": TIMES,

        "area": AREA,

        "data_format": "netcdf",

        "download_format": "unarchived",
    }


    # --------------------------------------------------------
    # RETRY
    # --------------------------------------------------------

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            safe_print(
                f"DOWNLOAD {label} | "
                f"attempt {attempt}/{MAX_RETRIES}"
            )


            client = cdsapi.Client(
                quiet=False,
                progress=True
            )


            if temp_file.exists():

                try:
                    temp_file.unlink()
                except Exception:
                    pass


            client.retrieve(
                DATASET,
                request,
                str(temp_file)
            )


            if not temp_file.exists():

                raise RuntimeError(
                    "Monthly file was not created."
                )


            size = (
                temp_file.stat().st_size
            )


            if size < 1000:

                raise RuntimeError(
                    "Monthly file is too small."
                )


            os.replace(
                temp_file,
                cache_file
            )


            safe_print(
                f"MONTH SUCCESS {label} | "
                f"{format_bytes(size)}"
            )


            return {
                "status": "downloaded",
                "size": size
            }


        except KeyboardInterrupt:

            raise


        except Exception as exc:

            safe_print(
                f"FAILED {label} | "
                f"{exc}"
            )


            if temp_file.exists():

                try:
                    temp_file.unlink()
                except Exception:
                    pass


            if attempt < MAX_RETRIES:

                safe_print(
                    f"RETRY {label} | "
                    f"waiting {RETRY_WAIT}s"
                )

                time.sleep(
                    RETRY_WAIT
                )


            else:

                return {
                    "status": "failed",
                    "size": 0,
                    "error": str(exc)
                }


# ============================================================
# SPLIT ONLY MISSING DAYS
# ============================================================

def split_missing_days(
    year,
    month
):

    cache_file = monthly_path(
        year,
        month
    )

    if not cache_file.exists():

        raise FileNotFoundError(
            cache_file
        )


    existing, missing = (
        month_status(
            year,
            month
        )
    )


    # --------------------------------------------------------
    # NOTHING TO SPLIT
    # --------------------------------------------------------

    if not missing:

        safe_print(
            f"SKIP SPLIT {year}-{month:02d} | "
            f"all {len(existing)} days already exist"
        )

        return 0


    safe_print(
        f"SPLIT {year}-{month:02d} | "
        f"missing days={len(missing)}"
    )


    # --------------------------------------------------------
    # Open monthly dataset
    # --------------------------------------------------------

    ds = xr.open_dataset(
        cache_file
    )


    try:

        if "valid_time" in ds.coords:

            time_name = "valid_time"

        elif "time" in ds.coords:

            time_name = "time"

        else:

            raise RuntimeError(
                "ERA5 time coordinate not found."
            )


        created = 0


        for day in missing:

            output_file = (
                daily_path(day)
            )


            # Re-check in case another
            # process created it.

            if daily_exists(day):

                continue


            # ------------------------------------------------
            # Select exact day
            # ------------------------------------------------

            daily_ds = ds.sel(
                {
                    time_name:
                    str(day)
                }
            )


            temp_daily = (
                output_file
                .with_suffix(".tmp.nc")
            )


            if temp_daily.exists():

                try:
                    temp_daily.unlink()
                except Exception:
                    pass


            daily_ds.to_netcdf(
                temp_daily
            )


            os.replace(
                temp_daily,
                output_file
            )


            created += 1


        safe_print(
            f"SPLIT DONE {year}-{month:02d} | "
            f"created={created} | "
            f"already={len(existing)}"
        )


        return created


    finally:

        ds.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "                 ERA5 FAST MONTHLY DOWNLOADER"
    )
    print("=" * 78)

    print(
        f"Period        : "
        f"{START_DATE} -> {END_DATE}"
    )

    print(
        "Months        : June, July, August, September"
    )

    print(
        "Resolution     : Hourly / 0.25 degree"
    )

    print(
        "Region         : 28-30 N / 76-79 E"
    )

    print(
        f"Variables      : {len(VARIABLES)}"
    )

    print(
        "CDS workers    : 1"
    )

    print(
        f"Output         : {OUTPUT_DIR}"
    )

    print("=" * 78)
    print()


    # ========================================================
    # MONTHS
    # ========================================================

    months = list(
        generate_months()
    )

    total_months = len(months)


    # ========================================================
    # TOTAL DAYS
    # ========================================================

    total_days = sum(
        len(
            days_for_month(
                y,
                m
            )
        )
        for y, m in months
    )


    # ========================================================
    # EXISTING DATA
    # ========================================================

    existing_days = 0
    existing_bytes = 0
    complete_months = 0


    for year, month in months:

        existing, missing = (
            month_status(
                year,
                month
            )
        )

        existing_days += len(
            existing
        )


        for day in existing:

            try:

                existing_bytes += (
                    daily_path(day)
                    .stat()
                    .st_size
                )

            except Exception:
                pass


        if not missing:

            complete_months += 1


    remaining_days = (
        total_days
        - existing_days
    )


    print(
        f"Total months   : {total_months}"
    )

    print(
        f"Complete months: {complete_months}"
    )

    print(
        f"Total days     : {total_days}"
    )

    print(
        f"Already done   : {existing_days}"
    )

    print(
        f"Remaining days : {remaining_days}"
    )

    print(
        f"Existing data  : "
        f"{format_bytes(existing_bytes)}"
    )

    print()


    # ========================================================
    # NOTHING LEFT
    # ========================================================

    if remaining_days == 0:

        print("=" * 78)

        print(
            "ALL ERA5 DAILY DATA ALREADY EXISTS."
        )

        print("=" * 78)

        return


    # ========================================================
    # TIMING
    # ========================================================

    start_time = time.time()

    months_processed = 0
    months_downloaded = 0
    months_skipped = 0
    months_failed = 0

    downloaded_bytes = 0


    # ========================================================
    # MONTH LOOP
    # ========================================================

    try:

        for index, (
            year,
            month
        ) in enumerate(
            months,
            start=1
        ):


            existing, missing = (
                month_status(
                    year,
                    month
                )
            )


            label = (
                f"{year}-{month:02d}"
            )


            # =================================================
            # COMPLETE MONTH = INSTANT SKIP
            # =================================================

            if not missing:

                months_skipped += 1

                months_processed += 1

                safe_print(
                    f"SKIP MONTH {label} | "
                    f"all {len(existing)} days already downloaded"
                )

                continue


            # =================================================
            # MONTH NEEDS DATA
            # =================================================

            print()
            print("=" * 78)

            print(
                f"MONTH {index}/{total_months} | "
                f"{label}"
            )

            print(
                f"Existing days : {len(existing)}"
            )

            print(
                f"Missing days  : {len(missing)}"
            )

            print("=" * 78)


            # -------------------------------------------------
            # Download monthly dataset
            # -------------------------------------------------

            result = download_month(
                year,
                month
            )


            if result["status"] == "failed":

                months_failed += 1

                safe_print(
                    f"MONTH FAILED {label}"
                )

                continue


            if result["status"] == "downloaded":

                months_downloaded += 1

                downloaded_bytes += (
                    result["size"]
                )


            else:

                months_skipped += 1


            # -------------------------------------------------
            # Extract only missing days
            # -------------------------------------------------

            try:

                split_missing_days(
                    year,
                    month
                )

            except Exception as exc:

                months_failed += 1

                safe_print(
                    f"SPLIT FAILED {label} | "
                    f"{exc}"
                )

                continue


            months_processed += 1


            # =================================================
            # DASHBOARD
            # =================================================

            current_days = 0
            current_bytes = 0


            for y, m in months:

                ex, _ = month_status(
                    y,
                    m
                )

                current_days += len(ex)


                for d in ex:

                    try:

                        current_bytes += (
                            daily_path(d)
                            .stat()
                            .st_size
                        )

                    except Exception:
                        pass


            elapsed = (
                time.time()
                - start_time
            )


            remaining = (
                total_days
                - current_days
            )


            # -----------------------------------------------
            # ETA based on completed DAYS
            # -----------------------------------------------

            days_done_this_run = (
                current_days
                - existing_days
            )


            if (
                days_done_this_run > 0
                and elapsed > 0
            ):

                days_per_second = (
                    days_done_this_run
                    / elapsed
                )

                eta_seconds = (
                    remaining
                    / days_per_second
                )

            else:

                eta_seconds = None


            speed = (
                downloaded_bytes
                / elapsed
                if elapsed > 0
                else 0
            )


            print()

            print(
                "--------------------------------------------------------------------------"
            )

            print(
                f"MONTH PROGRESS : "
                f"{months_processed}/{total_months}"
            )

            print(
                f"MONTH DOWNLOADED: "
                f"{months_downloaded}"
            )

            print(
                f"MONTH SKIPPED  : "
                f"{months_skipped}"
            )

            print(
                f"MONTH FAILED   : "
                f"{months_failed}"
            )

            print(
                f"DAYS ON DISK   : "
                f"{current_days}/{total_days}"
            )

            print(
                f"REMAINING      : "
                f"{remaining} days"
            )

            print(
                f"DOWNLOADED     : "
                f"{format_bytes(downloaded_bytes)}"
            )

            print(
                f"TOTAL ON DISK  : "
                f"{format_bytes(current_bytes)}"
            )

            print(
                f"SPEED          : "
                f"{format_speed(speed)}"
            )

            print(
                f"ETA            : "
                f"{format_eta(eta_seconds)}"
            )

            print(
                "--------------------------------------------------------------------------"
            )


    # ========================================================
    # CTRL+C
    # ========================================================

    except KeyboardInterrupt:

        print()
        print()
        print("=" * 78)

        print(
            "DOWNLOAD INTERRUPTED"
        )

        print(
            "Completed daily files are preserved."
        )

        print(
            "Run the same script again to continue."
        )

        print("=" * 78)

        sys.exit(0)


    # ========================================================
    # FINAL
    # ========================================================

    final_days = 0
    final_bytes = 0


    for year, month in months:

        existing, missing = (
            month_status(
                year,
                month
            )
        )

        final_days += len(
            existing
        )


        for day in existing:

            try:

                final_bytes += (
                    daily_path(day)
                    .stat()
                    .st_size
                )

            except Exception:
                pass


    final_remaining = (
        total_days
        - final_days
    )


    elapsed = (
        time.time()
        - start_time
    )


    speed = (
        downloaded_bytes / elapsed
        if elapsed > 0
        else 0
    )


    print()
    print("=" * 78)
    print(
        "                    FINAL REPORT"
    )
    print("=" * 78)

    print(
        f"Days on disk     : "
        f"{final_days}/{total_days}"
    )

    print(
        f"Days remaining   : "
        f"{final_remaining}"
    )

    print(
        f"Months downloaded: "
        f"{months_downloaded}"
    )

    print(
        f"Months skipped   : "
        f"{months_skipped}"
    )

    print(
        f"Months failed    : "
        f"{months_failed}"
    )

    print(
        f"Downloaded       : "
        f"{format_bytes(downloaded_bytes)}"
    )

    print(
        f"Total on disk    : "
        f"{format_bytes(final_bytes)}"
    )

    print(
        f"Average speed    : "
        f"{format_speed(speed)}"
    )

    print(
        f"Elapsed          : "
        f"{elapsed / 60:.1f} minutes"
    )

    print("=" * 78)


    if final_remaining == 0:

        print()
        print(
            "ALL ERA5 DATA COMPLETE."
        )

    elif months_failed > 0:

        print()
        print(
            "Some months failed."
        )

        print(
            "Run the same script again; "
            "completed data will be skipped."
        )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()