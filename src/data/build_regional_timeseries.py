from pathlib import Path
from datetime import datetime, timezone
import json
import time
import traceback

import earthaccess
import h5py
import numpy as np
from tqdm import tqdm


# ============================================================
# IMERG FINAL V07B
# HISTORICAL BULK REGIONAL ACQUISITION PIPELINE
# ============================================================
#
# DATASET
# -------
# NASA GPM IMERG Final Run V07
#
# PERIOD
# ------
# 1998 - 2020
#
# MONTHS
# -------
# June, July, August, September
#
# TEMPORAL RESOLUTION
# -------------------
# 30 minutes
#
# SPATIAL RESOLUTION
# ------------------
# 0.1 degree
#
# REGION
# ------
# Latitude  : 28.0 - 30.0 N
# Longitude : 76.0 - 79.0 E
#
# EXPECTED REGIONAL GRID
# ----------------------
# 20 x 30
#
# WORKFLOW
# --------
#
# Earthdata / GES DISC
#        ↓
# IMERG global HDF5
#        ↓
# HDF5 validation
#        ↓
# Regional crop
#        ↓
# Quality checks
#        ↓
# Compressed NPZ
#        ↓
# Global HDF5 deleted
#
# IMPORTANT
# ---------
# Download is performed in DAY-SIZED chunks.
#
# This prevents hundreds/thousands of global HDF5 files
# from accumulating in the temporary directory.
#
# Resume-safe:
# Existing regional NPZ files are skipped.
#
# Failed HDF5 files are NOT deleted.
#
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

SHORT_NAME = "GPM_3IMERGHH"
VERSION = "07"

# ------------------------------------------------------------
# TRAINING PERIOD
# ------------------------------------------------------------

START_YEAR = 1998
END_YEAR = 2020

# ------------------------------------------------------------
# MONSOON MONTHS
# ------------------------------------------------------------

MONTHS = [6, 7, 8, 9]

# ------------------------------------------------------------
# REGIONAL DOMAIN
# ------------------------------------------------------------

LAT_MIN = 28.0
LAT_MAX = 30.0

LON_MIN = 76.0
LON_MAX = 79.0

# ------------------------------------------------------------
# EXPECTED GRID
# ------------------------------------------------------------

EXPECTED_LAT_POINTS = 20
EXPECTED_LON_POINTS = 30

# ------------------------------------------------------------
# IMERG temporal resolution
# ------------------------------------------------------------

FRAMES_PER_DAY = 48

# ------------------------------------------------------------
# Download threads
# ------------------------------------------------------------

DOWNLOAD_THREADS = 4

# ------------------------------------------------------------
# Temporary download chunk
#
# 1 day = 48 IMERG files
# ------------------------------------------------------------

CHUNK_SIZE = 48


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

TEMP_DIR = (
    PROJECT_ROOT
    / "data"
    / "temp_imerg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imerg"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)

LOG_FILE = (
    LOG_DIR
    / "imerg_bulk_acquisition.jsonl"
)

INVENTORY_FILE = (
    LOG_DIR
    / "imerg_bulk_inventory.json"
)


# ============================================================
# 3. CREATE DIRECTORIES
# ============================================================

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. LOGGING
# ============================================================

def log_event(
    event_type,
    **kwargs
):

    record = {
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "event":
            event_type,

        **kwargs,
    }

    with LOG_FILE.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                record
            )
            + "\n"
        )


# ============================================================
# 5. DATE UTILITIES
# ============================================================

def month_days(
    year,
    month
):

    if month == 12:

        next_month = datetime(
            year + 1,
            1,
            1
        )

    else:

        next_month = datetime(
            year,
            month + 1,
            1
        )

    current = datetime(
        year,
        month,
        1
    )

    return (
        next_month - current
    ).days


def expected_frames():

    total = 0

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        for month in MONTHS:

            total += (
                month_days(
                    year,
                    month
                )
                * FRAMES_PER_DAY
            )

    return total


def expected_frames_for_month(
    year,
    month
):

    return (
        month_days(
            year,
            month
        )
        * FRAMES_PER_DAY
    )


# ============================================================
# 6. BUILD ALL PROCESSING DAYS
# ============================================================

def build_day_list():

    days = []

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        for month in MONTHS:

            number_of_days = (
                month_days(
                    year,
                    month
                )
            )

            for day in range(
                1,
                number_of_days + 1
            ):

                days.append(
                    (
                        year,
                        month,
                        day
                    )
                )

    return days


# ============================================================
# 7. EARTHDATA SEARCH
# ============================================================

def search_day(
    year,
    month,
    day
):

    start_date = (
        f"{year:04d}-"
        f"{month:02d}-"
        f"{day:02d}"
        "T00:00:00"
    )

    if day < month_days(
        year,
        month
    ):

        end_date = (
            f"{year:04d}-"
            f"{month:02d}-"
            f"{day + 1:02d}"
            "T00:00:00"
        )

    else:

        if month == 12:

            next_year = year + 1
            next_month = 1

        else:

            next_year = year
            next_month = month + 1

        end_date = (
            f"{next_year:04d}-"
            f"{next_month:02d}-01"
            "T00:00:00"
        )

    results = (
        earthaccess.search_data(
            short_name=SHORT_NAME,
            version=VERSION,
            temporal=(
                start_date,
                end_date
            )
        )
    )

    return results


# ============================================================
# 8. EXTRACT NASA FILE URL
# ============================================================

def get_result_url(
    result
):

    umm = result.get(
        "umm",
        {}
    )

    related_urls = (
        umm.get(
            "RelatedUrls",
            []
        )
    )

    # Prefer actual HDF5 data URL
    for item in related_urls:

        url = item.get(
            "URL",
            ""
        )

        if (
            "GPM_3IMERGHH" in url
            and (
                ".HDF5" in url
                or ".h5" in url
            )
        ):

            return url

    # Fallback
    for item in related_urls:

        url = item.get(
            "URL",
            ""
        )

        if url:

            return url

    return ""


# ============================================================
# 9. OUTPUT FILENAME
# ============================================================

def output_filename(
    result
):

    url = get_result_url(
        result
    )

    if not url:

        raise ValueError(
            "Could not determine "
            "NASA granule URL."
        )

    original = Path(
        url
    ).name

    stem = original

    if stem.endswith(
        ".HDF5"
    ):

        stem = stem[:-5]

    elif stem.endswith(
        ".h5"
    ):

        stem = stem[:-3]

    return (
        f"{stem}_regional.npz"
    )


# ============================================================
# 10. HDF5 VALIDATION
# ============================================================

def validate_hdf5(
    hdf5_path
):

    required = [
        "Grid/precipitation",
        "Grid/lat",
        "Grid/lon",
    ]

    with h5py.File(
        hdf5_path,
        "r"
    ) as f:

        # ----------------------------------------------------
        # Dataset existence
        # ----------------------------------------------------

        for item in required:

            if item not in f:

                raise ValueError(
                    "Missing HDF5 dataset: "
                    f"{item}"
                )

        # ----------------------------------------------------
        # Shapes
        # ----------------------------------------------------

        precip_shape = (
            f[
                "Grid/precipitation"
            ].shape
        )

        lat_shape = (
            f[
                "Grid/lat"
            ].shape
        )

        lon_shape = (
            f[
                "Grid/lon"
            ].shape
        )

        if len(
            precip_shape
        ) != 3:

            raise ValueError(
                "Unexpected precipitation "
                f"shape: {precip_shape}"
            )

        if len(
            lat_shape
        ) != 1:

            raise ValueError(
                "Unexpected latitude "
                f"shape: {lat_shape}"
            )

        if len(
            lon_shape
        ) != 1:

            raise ValueError(
                "Unexpected longitude "
                f"shape: {lon_shape}"
            )


# ============================================================
# 11. PROCESS HDF5 → REGIONAL NPZ
# ============================================================

def process_hdf5_to_regional_npz(
    hdf5_path,
    output_path
):

    # --------------------------------------------------------
    # Read global HDF5
    # --------------------------------------------------------

    with h5py.File(
        hdf5_path,
        "r"
    ) as f:

        precip = (
            f[
                "Grid/precipitation"
            ][:]
        )

        lat = (
            f[
                "Grid/lat"
            ][:]
        )

        lon = (
            f[
                "Grid/lon"
            ][:]
        )

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        if (
            "Grid/time"
            in f
        ):

            time_value = (
                f[
                    "Grid/time"
                ][:]
            )

        else:

            time_value = np.array(
                [],
                dtype=np.int64
            )

    # --------------------------------------------------------
    # Convert dtype
    # --------------------------------------------------------

    precip = np.asarray(
        precip,
        dtype=np.float32
    )

    lat = np.asarray(
        lat
    )

    lon = np.asarray(
        lon
    )

    # --------------------------------------------------------
    # Invalid values
    # --------------------------------------------------------

    precip[
        ~np.isfinite(
            precip
        )
    ] = np.nan

    precip[
        precip <= -9990
    ] = np.nan

    # --------------------------------------------------------
    # Determine orientation
    # --------------------------------------------------------
    #
    # IMERG commonly stores:
    #
    # (time, lon, lat)
    #
    # We want:
    #
    # (time, lat, lon)
    #
    # --------------------------------------------------------

    if (
        precip.shape[1]
        == len(lon)

        and

        precip.shape[2]
        == len(lat)
    ):

        precip = np.transpose(
            precip,
            (
                0,
                2,
                1
            )
        )

    elif (
        precip.shape[1]
        == len(lat)

        and

        precip.shape[2]
        == len(lon)
    ):

        # Already correct
        pass

    else:

        raise ValueError(
            "Cannot determine precipitation "
            "spatial dimensions. "

            f"precip={precip.shape}, "

            f"lat={len(lat)}, "

            f"lon={len(lon)}"
        )

    # --------------------------------------------------------
    # Regional mask
    # --------------------------------------------------------

    lat_mask = (
        (lat >= LAT_MIN)
        &
        (lat < LAT_MAX)
    )

    lon_mask = (
        (lon >= LON_MIN)
        &
        (lon < LON_MAX)
    )

    if not np.any(
        lat_mask
    ):

        raise ValueError(
            "No latitude points "
            "inside target region."
        )

    if not np.any(
        lon_mask
    ):

        raise ValueError(
            "No longitude points "
            "inside target region."
        )

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    regional_lat = (
        lat[
            lat_mask
        ]
    )

    regional_lon = (
        lon[
            lon_mask
        ]
    )

    regional_precip = (
        precip[
            :,
            lat_mask,
            :
        ][:, :, lon_mask]
    )

    # --------------------------------------------------------
    # Grid validation
    # --------------------------------------------------------

    if len(
        regional_lat
    ) != EXPECTED_LAT_POINTS:

        raise ValueError(
            "Unexpected regional "
            "latitude count: "

            f"{len(regional_lat)} "
            f"(expected "
            f"{EXPECTED_LAT_POINTS})"
        )

    if len(
        regional_lon
    ) != EXPECTED_LON_POINTS:

        raise ValueError(
            "Unexpected regional "
            "longitude count: "

            f"{len(regional_lon)} "
            f"(expected "
            f"{EXPECTED_LON_POINTS})"
        )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    expected_shape = (
        len(regional_lat),
        len(regional_lon)
    )

    if (
        regional_precip.shape[1:]
        != expected_shape
    ):

        raise ValueError(
            "Regional precipitation "
            "shape mismatch."
        )

    # --------------------------------------------------------
    # Infinity check
    # --------------------------------------------------------

    if np.isinf(
        regional_precip
    ).any():

        raise ValueError(
            "Infinite values detected."
        )

    # --------------------------------------------------------
    # Save compressed NPZ
    # --------------------------------------------------------

    np.savez_compressed(

        output_path,

        precipitation=(
            regional_precip
        ),

        lat=(
            regional_lat
        ),

        lon=(
            regional_lon
        ),

        time=(
            time_value
        ),
    )

    # --------------------------------------------------------
    # Verify saved NPZ
    # --------------------------------------------------------

    with np.load(
        output_path
    ) as check:

        saved_precip = (
            check[
                "precipitation"
            ]
        )

        saved_lat = (
            check[
                "lat"
            ]
        )

        saved_lon = (
            check[
                "lon"
            ]
        )

        if (
            saved_precip.shape
            != regional_precip.shape
        ):

            raise ValueError(
                "Saved precipitation "
                "verification failed."
            )

        if (
            len(saved_lat)
            != EXPECTED_LAT_POINTS
        ):

            raise ValueError(
                "Saved latitude "
                "verification failed."
            )

        if (
            len(saved_lon)
            != EXPECTED_LON_POINTS
        ):

            raise ValueError(
                "Saved longitude "
                "verification failed."
            )


# ============================================================
# 12. CHECK WHETHER RESULT ALREADY EXISTS
# ============================================================

def regional_file_exists(
    result
):

    try:

        filename = (
            output_filename(
                result
            )
        )

    except Exception:

        return False

    return (
        OUTPUT_DIR
        / filename
    ).exists()


# ============================================================
# 13. PROCESS ONE DOWNLOADED HDF5
# ============================================================

def process_one_file(
    hdf5_path,
    result,
    year,
    month,
    day
):

    hdf5_path = Path(
        hdf5_path
    )

    # --------------------------------------------------------
    # Determine output
    # --------------------------------------------------------

    filename = (
        output_filename(
            result
        )
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    # --------------------------------------------------------
    # Already processed
    # --------------------------------------------------------

    if output_path.exists():

        hdf5_path.unlink(
            missing_ok=True
        )

        return "skipped"

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_hdf5(
        hdf5_path
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    process_hdf5_to_regional_npz(

        hdf5_path,

        output_path
    )

    # --------------------------------------------------------
    # Delete global HDF5
    # ONLY AFTER successful processing
    # --------------------------------------------------------

    hdf5_path.unlink(
        missing_ok=True
    )

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    log_event(

        "processed",

        file=filename,

        year=year,

        month=month,

        day=day,
    )

    return "processed"


# ============================================================
# 14. MAIN
# ============================================================

def main():

    global_start = time.time()

    total_expected = (
        expected_frames()
    )

    day_list = (
        build_day_list()
    )

    total_days = len(
        day_list
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 80)
    print(
        "IMERG FINAL V07B"
    )
    print(
        "HISTORICAL BULK REGIONAL ACQUISITION"
    )
    print("=" * 80)

    print(
        f"Dataset          : "
        f"{SHORT_NAME}"
    )

    print(
        f"Version          : "
        f"{VERSION}"
    )

    print(
        f"Training period  : "
        f"{START_YEAR}–{END_YEAR}"
    )

    print(
        f"Months           : "
        f"{MONTHS}"
    )

    print(
        f"Total days       : "
        f"{total_days:,}"
    )

    print(
        f"Expected frames  : "
        f"{total_expected:,}"
    )

    print(
        f"Region           : "
        f"{LAT_MIN}–{LAT_MAX} N, "
        f"{LON_MIN}–{LON_MAX} E"
    )

    print(
        f"Regional grid    : "
        f"{EXPECTED_LAT_POINTS} × "
        f"{EXPECTED_LON_POINTS}"
    )

    print(
        f"Resolution       : "
        f"0.1°"
    )

    print(
        f"Temporal         : "
        f"30 minutes"
    )

    print(
        f"Threads          : "
        f"{DOWNLOAD_THREADS}"
    )

    print(
        f"Download chunk   : "
        f"{CHUNK_SIZE} files/day"
    )

    print("=" * 80)
    print()

    # ========================================================
    # EARTHDATA LOGIN
    # ========================================================

    print(
        "[1/3] "
        "NASA Earthdata authentication..."
    )

    auth = earthaccess.login(
        persist=True,
        strategy="interactive"
    )

    if not auth.authenticated:

        raise RuntimeError(
            "Earthdata authentication failed."
        )

    print(
        "[OK] "
        "Earthdata authentication successful."
    )

    print()

    # ========================================================
    # INVENTORY
    # ========================================================

    inventory = {

        "dataset":
            SHORT_NAME,

        "version":
            VERSION,

        "start_year":
            START_YEAR,

        "end_year":
            END_YEAR,

        "months":
            MONTHS,

        "region":
        {
            "lat_min":
                LAT_MIN,

            "lat_max":
                LAT_MAX,

            "lon_min":
                LON_MIN,

            "lon_max":
                LON_MAX,
        },

        "expected_frames":
            total_expected,

        "processed":
            0,

        "skipped":
            0,

        "failed":
            0,

        "days_completed":
            [],

        "last_processed_day":
            None,

        "runtime_minutes":
            0,
    }

    # ========================================================
    # GLOBAL PROGRESS BAR
    # ========================================================

    overall_bar = tqdm(

        total=total_expected,

        desc="OVERALL",

        unit="frame",

        dynamic_ncols=True,

        mininterval=1.0,

        smoothing=0.1,

    )

    # ========================================================
    # DAY LOOP
    # ========================================================

    for day_index, (
        year,
        month,
        day
    ) in enumerate(
        day_list,
        start=1
    ):

        day_start = time.time()

        print()
        print("-" * 80)

        print(
            f"DAY "
            f"[{day_index}/{total_days}]  "
            f"{year}-{month:02d}-{day:02d}"
        )

        print("-" * 80)

        try:

            # =================================================
            # SEARCH
            # =================================================

            search_start = time.time()

            results = (
                search_day(
                    year,
                    month,
                    day
                )
            )

            search_seconds = (
                time.time()
                - search_start
            )

            expected_today = (
                FRAMES_PER_DAY
            )

            print(
                f"Granules found   : "
                f"{len(results):,}"
            )

            print(
                f"Expected         : "
                f"{expected_today:,}"
            )

            print(
                f"Search time      : "
                f"{search_seconds:.2f} sec"
            )

            # =================================================
            # SEARCH VALIDATION
            # =================================================

            if len(results) != (
                expected_today
            ):

                print(
                    "[WARNING] "
                    "This day does not contain "
                    "the expected 48 granules."
                )

                log_event(

                    "daily_granule_count_warning",

                    year=year,

                    month=month,

                    day=day,

                    found=len(results),

                    expected=expected_today,
                )

            # =================================================
            # DETERMINE PENDING RESULTS
            # =================================================

            pending_results = []

            already_done = 0

            for result in results:

                try:

                    if regional_file_exists(
                        result
                    ):

                        already_done += 1

                    else:

                        pending_results.append(
                            result
                        )

                except Exception as exc:

                    print(
                        "[WARNING] "
                        "Could not determine "
                        "existing output: "
                        f"{exc}"
                    )

                    pending_results.append(
                        result
                    )

            # =================================================
            # UPDATE SKIPPED
            # =================================================

            if already_done > 0:

                overall_bar.update(
                    already_done
                )

                inventory[
                    "skipped"
                ] += already_done

            print(
                f"Already processed: "
                f"{already_done:,}"
            )

            print(
                f"To download      : "
                f"{len(pending_results):,}"
            )

            # =================================================
            # DOWNLOAD
            # =================================================

            downloaded_paths = []

            if pending_results:

                print()

                print(
                    "[DOWNLOAD] "
                    "Downloading this day's "
                    "global HDF5 files..."
                )

                download_start = (
                    time.time()
                )

                downloaded_paths = (
                    earthaccess.download(

                        pending_results,

                        local_path=TEMP_DIR,

                        threads=(
                            DOWNLOAD_THREADS
                        ),
                    )
                )

                download_seconds = (
                    time.time()
                    - download_start
                )

                print()

                print(
                    f"[DOWNLOAD] "
                    f"Returned "
                    f"{len(downloaded_paths):,} "
                    f"files."
                )

                print(
                    f"[DOWNLOAD] "
                    f"Time: "
                    f"{download_seconds / 60:.2f} min"
                )

            else:

                print(
                    "[DOWNLOAD] "
                    "Nothing to download."
                )

            # =================================================
            # PROCESS
            # =================================================

            day_processed = 0
            day_failed = 0
            day_skipped = already_done

            if downloaded_paths:

                print()

                print(
                    "[PROCESS] "
                    "Cropping global files "
                    "to regional 20×30 grid..."
                )

            # -------------------------------------------------
            # Map downloaded paths to results.
            #
            # Earthaccess generally returns paths in the same
            # order as the requested results, but we use
            # filename matching as a safer mechanism.
            # -------------------------------------------------

            result_by_name = {}

            for result in pending_results:

                try:

                    expected_name = (
                        output_filename(
                            result
                        )
                    )

                    global_stem = (
                        expected_name
                        .replace(
                            "_regional.npz",
                            ""
                        )
                    )

                    result_by_name[
                        global_stem
                    ] = result

                except Exception:

                    pass

            for file_index, hdf5_path in enumerate(
                downloaded_paths,
                start=1
            ):

                hdf5_path = Path(
                    hdf5_path
                )

                # ------------------------------------------------
                # Find corresponding result
                # ------------------------------------------------

                result = None

                stem = (
                    hdf5_path.stem
                )

                if stem in result_by_name:

                    result = (
                        result_by_name[
                            stem
                        ]
                    )

                else:

                    # Try normalized stem
                    for key, candidate in (
                        result_by_name.items()
                    ):

                        if (
                            key in stem
                            or stem in key
                        ):

                            result = candidate
                            break

                try:

                    if result is None:

                        raise ValueError(
                            "Could not match "
                            "downloaded HDF5 to "
                            "Earthdata result."
                        )

                    status = (
                        process_one_file(

                            hdf5_path,

                            result,

                            year,

                            month,

                            day,
                        )
                    )

                    if status == (
                        "processed"
                    ):

                        day_processed += 1

                        inventory[
                            "processed"
                        ] += 1

                    elif status == (
                        "skipped"
                    ):

                        day_skipped += 1

                        inventory[
                            "skipped"
                        ] += 1

                    # ------------------------------------------------
                    # UPDATE OVERALL PROGRESS
                    # ------------------------------------------------

                    overall_bar.update(
                        1
                    )

                except Exception as exc:

                    day_failed += 1

                    inventory[
                        "failed"
                    ] += 1

                    print()

                    print(
                        "[ERROR] "
                        f"{hdf5_path.name}"
                    )

                    print(
                        f"Reason: {exc}"
                    )

                    log_event(

                        "file_processing_failed",

                        file=(
                            hdf5_path.name
                        ),

                        year=year,

                        month=month,

                        day=day,

                        error=str(exc),

                        traceback=(
                            traceback.format_exc()
                        ),
                    )

                    # IMPORTANT:
                    #
                    # Failed HDF5 remains in
                    # temp_imerg for inspection
                    # / retry.
                    #
                    # We do NOT delete it.

            # =================================================
            # DAY SUMMARY
            # =================================================

            day_seconds = (
                time.time()
                - day_start
            )

            print()

            print(
                f"{year}-{month:02d}-{day:02d} "
                "COMPLETE"
            )

            print(
                f"Processed : "
                f"{day_processed:,}"
            )

            print(
                f"Skipped   : "
                f"{day_skipped:,}"
            )

            print(
                f"Failed    : "
                f"{day_failed:,}"
            )

            print(
                f"Runtime   : "
                f"{day_seconds / 60:.2f} min"
            )

            # =================================================
            # DAY COMPLETE
            # =================================================

            if day_failed == 0:

                inventory[
                    "days_completed"
                ].append(
                    f"{year:04d}-"
                    f"{month:02d}-"
                    f"{day:02d}"
                )

            inventory[
                "last_processed_day"
            ] = (
                f"{year:04d}-"
                f"{month:02d}-"
                f"{day:02d}"
            )

            # =================================================
            # SAVE INVENTORY AFTER EVERY DAY
            # =================================================

            inventory[
                "runtime_minutes"
            ] = (
                time.time()
                - global_start
            ) / 60

            with INVENTORY_FILE.open(
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    inventory,
                    f,
                    indent=2
                )

            # =================================================
            # LOG DAY
            # =================================================

            log_event(

                "day_completed",

                year=year,

                month=month,

                day=day,

                processed=day_processed,

                skipped=day_skipped,

                failed=day_failed,

                runtime_minutes=(
                    day_seconds / 60
                ),
            )

        # =====================================================
        # CTRL+C
        # =====================================================

        except KeyboardInterrupt:

            print()
            print()
            print(
                "=" * 80
            )

            print(
                "[STOPPED BY USER]"
            )

            print(
                "Downloader interrupted."
            )

            print(
                "Processed regional NPZ files "
                "are safe."
            )

            print(
                "Temporary HDF5 files are "
                "NOT deleted unless processing "
                "was successful."
            )

            print(
                "=" * 80
            )

            inventory[
                "runtime_minutes"
            ] = (
                time.time()
                - global_start
            ) / 60

            with INVENTORY_FILE.open(
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    inventory,
                    f,
                    indent=2
                )

            overall_bar.close()

            return

        # =====================================================
        # DAY FAILURE
        # =====================================================

        except Exception as exc:

            inventory[
                "failed"
            ] += 1

            print()
            print(
                "[DAY FAILED]"
            )

            print(
                f"{year}-{month:02d}-{day:02d}"
            )

            print(
                f"Reason: {exc}"
            )

            log_event(

                "day_failed",

                year=year,

                month=month,

                day=day,

                error=str(exc),

                traceback=(
                    traceback.format_exc()
                ),
            )

            # ------------------------------------------------
            # Save progress even after failure
            # ------------------------------------------------

            inventory[
                "runtime_minutes"
            ] = (
                time.time()
                - global_start
            ) / 60

            with INVENTORY_FILE.open(
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    inventory,
                    f,
                    indent=2
                )

            continue

    # ========================================================
    # CLOSE PROGRESS BAR
    # ========================================================

    overall_bar.close()

    # ========================================================
    # FINAL RUNTIME
    # ========================================================

    total_runtime_minutes = (
        time.time()
        - global_start
    ) / 60

    inventory[
        "runtime_minutes"
    ] = total_runtime_minutes

    inventory[
        "completed_at_utc"
    ] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    # ========================================================
    # FINAL INVENTORY
    # ========================================================

    with INVENTORY_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            inventory,
            f,
            indent=2
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)

    print(
        "FINAL ACQUISITION SUMMARY"
    )

    print("=" * 80)

    print(
        f"Dataset          : "
        f"{SHORT_NAME}"
    )

    print(
        f"Version          : "
        f"{VERSION}"
    )

    print(
        f"Training period  : "
        f"{START_YEAR}–{END_YEAR}"
    )

    print(
        f"Months           : "
        f"{MONTHS}"
    )

    print(
        f"Expected frames  : "
        f"{total_expected:,}"
    )

    print(
        f"Processed         : "
        f"{inventory['processed']:,}"
    )

    print(
        f"Skipped           : "
        f"{inventory['skipped']:,}"
    )

    print(
        f"Failed            : "
        f"{inventory['failed']:,}"
    )

    print(
        f"Runtime           : "
        f"{total_runtime_minutes:.2f} min"
    )

    print()

    print(
        f"Regional data     : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Temporary data    : "
        f"{TEMP_DIR}"
    )

    print(
        f"Log               : "
        f"{LOG_FILE}"
    )

    print(
        f"Inventory         : "
        f"{INVENTORY_FILE}"
    )

    print("=" * 80)

    if (
        inventory["failed"]
        == 0
    ):

        print(
            "[SUCCESS]"
        )

        print(
            "1998–2020 historical "
            "regional acquisition completed "
            "without recorded failures."
        )

    else:

        print(
            "[WARNING]"
        )

        print(
            "Acquisition completed with "
            f"{inventory['failed']} "
            "recorded failures."
        )

        print(
            "Failed files/days are logged "
            "for later retry."
        )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()