import re
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import earthaccess


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(r"C:\projects\rainfall-nowcasting")
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "imerg"
TEMP_DIR = PROJECT_DIR / "data" / "temp_opendap"

START_DATE = date(2008, 6, 1)
END_DATE = date(2020, 9, 30)

ALLOWED_MONTHS = {6, 7, 8, 9}

SHORT_NAME = "GPM_3IMERGHH"
VERSION = "07"

LAT_MIN = 28.0
LAT_MAX = 30.0
LON_MIN = 76.0
LON_MAX = 79.0

GRID_RESOLUTION = 0.1

EXPECTED_LAT_SIZE = 20
EXPECTED_LON_SIZE = 30

MAX_RETRIES = 3
RETRY_DELAY = 5

AUTH_RETRIES = 3
AUTH_RETRY_DELAY = 10

OPENDAP_BASE = (
    "https://opendap.earthdata.nasa.gov/"
    "collections/C2723754847-GES_DISC/granules/"
)


# =============================================================================
# PROJECT SAFETY
# =============================================================================

def safety_check():

    print("=" * 80)
    print("PROJECT SAFETY CHECK")
    print("=" * 80)

    detected_project = Path.cwd().resolve()
    allowed_project = PROJECT_DIR.resolve()

    print(f"Detected project : {detected_project}")
    print(f"Allowed project  : {allowed_project}")

    if detected_project != allowed_project:

        raise RuntimeError(
            "\nWRONG PROJECT DIRECTORY.\n"
            f"Expected: {allowed_project}\n"
            f"Found   : {detected_project}"
        )

    print("[OK] Correct project directory.")
    print()


# =============================================================================
# TARGET DATES
# =============================================================================

def get_target_dates():

    dates = []

    current = START_DATE

    while current <= END_DATE:

        if current.month in ALLOWED_MONTHS:
            dates.append(current)

        current += timedelta(days=1)

    return dates


# =============================================================================
# NPZ VALIDATION
# =============================================================================

def is_valid_npz(file_path):

    try:

        with np.load(
            file_path,
            allow_pickle=False
        ) as data:

            required_keys = {
                "precipitation",
                "lat",
                "lon",
                "time",
            }

            if not required_keys.issubset(
                set(data.files)
            ):
                return False

            precipitation = data["precipitation"]
            lat = data["lat"]
            lon = data["lon"]

            if precipitation.shape != (
                1,
                EXPECTED_LAT_SIZE,
                EXPECTED_LON_SIZE,
            ):
                return False

            if lat.shape != (
                EXPECTED_LAT_SIZE,
            ):
                return False

            if lon.shape != (
                EXPECTED_LON_SIZE,
            ):
                return False

            if not np.isfinite(lat).all():
                return False

            if not np.isfinite(lon).all():
                return False

            return True

    except Exception:

        return False


# =============================================================================
# PARSE EXISTING FILENAMES
# =============================================================================

def parse_existing_filename(filename):

    """
    Supports:

    NEW:
        20080601_0000.npz

    OLD:
        3B-HHR.MS.MRG.3IMERG.20090713-S163000-E165959.0990.V07B_regional.npz
    """

    # -------------------------------------------------------------------------
    # NEW FORMAT
    # -------------------------------------------------------------------------

    new_match = re.match(
        r"^(\d{8})_(\d{4})\.npz$",
        filename
    )

    if new_match:

        date_str = new_match.group(1)
        time_str = new_match.group(2)

        try:

            dt = datetime.strptime(
                date_str + time_str,
                "%Y%m%d%H%M"
            )

            return (
                dt.date(),
                dt.strftime("%H%M")
            )

        except ValueError:

            return None

    # -------------------------------------------------------------------------
    # OLD FORMAT
    # -------------------------------------------------------------------------

    old_match = re.search(
        r"3IMERG\.(\d{8})-S(\d{6})-E",
        filename
    )

    if old_match:

        date_str = old_match.group(1)
        start_time = old_match.group(2)

        try:

            dt = datetime.strptime(
                date_str + start_time,
                "%Y%m%d%H%M%S"
            )

            return (
                dt.date(),
                dt.strftime("%H%M")
            )

        except ValueError:

            return None

    return None


# =============================================================================
# LOCAL INVENTORY
# =============================================================================

def build_local_inventory(target_dates):

    print("=" * 80)
    print("BUILDING LOCAL DATA INVENTORY")
    print("=" * 80)
    print("Scanning existing NPZ files...")
    print()

    target_date_set = set(
        target_dates
    )

    expected_times = {
        f"{hour:02d}{minute:02d}"
        for hour in range(24)
        for minute in (0, 30)
    }

    existing_frames = set()

    files = list(
        OUTPUT_DIR.glob("*.npz")
    )

    for file_path in files:

        parsed = parse_existing_filename(
            file_path.name
        )

        if parsed is None:
            continue

        frame_date, frame_time = parsed

        if frame_date not in target_date_set:
            continue

        if frame_time not in expected_times:
            continue

        if not is_valid_npz(file_path):
            continue

        existing_frames.add(
            (
                frame_date,
                frame_time
            )
        )

    total_frames = (
        len(target_dates) * 48
    )

    complete_days = 0
    incomplete_days = 0
    missing_frames = 0

    for target_date in target_dates:

        day_count = sum(
            1
            for time_part in expected_times
            if (
                target_date,
                time_part
            ) in existing_frames
        )

        if day_count == 48:

            complete_days += 1

        else:

            incomplete_days += 1
            missing_frames += (
                48 - day_count
            )

    existing_count = len(
        existing_frames
    )

    coverage = (
        existing_count
        / total_frames
        * 100
        if total_frames > 0
        else 0
    )

    print()
    print(
        f"Target days       : "
        f"{len(target_dates):,}"
    )

    print(
        f"Target frames     : "
        f"{total_frames:,}"
    )

    print(
        f"Existing frames   : "
        f"{existing_count:,}"
    )

    print(
        f"Frame coverage    : "
        f"{coverage:.2f}%"
    )

    print(
        f"Complete days     : "
        f"{complete_days:,}"
    )

    print(
        f"Incomplete days   : "
        f"{incomplete_days:,}"
    )

    print(
        f"Missing frames    : "
        f"{missing_frames:,}"
    )

    print("=" * 80)

    return (
        existing_frames,
        complete_days,
        incomplete_days,
        missing_frames,
    )


# =============================================================================
# GRID INDICES
# =============================================================================

def calculate_grid_indices():

    lon_start_index = int(
        round(
            (LON_MIN + 180.0)
            / GRID_RESOLUTION
        )
    )

    lon_end_index = int(
        round(
            (LON_MAX + 180.0)
            / GRID_RESOLUTION
        )
    ) - 1

    lat_start_index = int(
        round(
            (LAT_MIN + 90.0)
            / GRID_RESOLUTION
        )
    )

    lat_end_index = int(
        round(
            (LAT_MAX + 90.0)
            / GRID_RESOLUTION
        )
    ) - 1

    return (
        lon_start_index,
        lon_end_index,
        lat_start_index,
        lat_end_index,
    )


# =============================================================================
# OPENDAP URL
# =============================================================================

def build_opendap_url(filename):

    granule_id = (
        f"{SHORT_NAME}.{VERSION}:{filename}"
    )

    encoded_granule_id = (
        granule_id.replace(
            ":",
            "%253A"
        )
    )

    return (
        OPENDAP_BASE
        + encoded_granule_id
        + ".dap.nc4"
    )


# =============================================================================
# OPENDAP CONSTRAINT
# =============================================================================

def build_constraint():

    (
        lon_start,
        lon_end,
        lat_start,
        lat_end,
    ) = calculate_grid_indices()

    return (
        f"/Grid/time[0:1:0];"
        f"/Grid/lon[{lon_start}:1:{lon_end}];"
        f"/Grid/lat[{lat_start}:1:{lat_end}];"
        f"/Grid/precipitation"
        f"[0:1:0]"
        f"[{lon_start}:1:{lon_end}]"
        f"[{lat_start}:1:{lat_end}]"
    )


# =============================================================================
# NASA SEARCH
# =============================================================================

def search_day(target_date):

    start_dt = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        0,
        0,
        0,
    )

    end_dt = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        23,
        59,
        59,
    )

    print(
        f"Searching NASA granules for "
        f"{target_date.isoformat()}..."
    )

    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=VERSION,
        temporal=(
            start_dt,
            end_dt
        ),
    )

    return results


# =============================================================================
# GRANULE FILENAME
# =============================================================================

def get_granule_filename(granule):

    links = granule.data_links()

    if not links:
        return None

    for link in links:

        filename = str(link).split("/")[-1]

        if filename.endswith(".HDF5"):

            return filename

    return str(
        links[0]
    ).split("/")[-1]


# =============================================================================
# FRAME DATETIME
# =============================================================================

def get_frame_datetime(filename):

    match = re.search(
        r"3IMERG\.(\d{8})-S(\d{6})-E",
        filename
    )

    if not match:
        return None

    date_str = match.group(1)
    time_str = match.group(2)

    try:

        return datetime.strptime(
            date_str + time_str,
            "%Y%m%d%H%M%S"
        )

    except ValueError:

        return None


# =============================================================================
# PARSE OPENDAP RESPONSE
# =============================================================================

def parse_response(
    url,
    constraint
):

    full_url = (
        url
        + "?"
        + constraint
    )

    dataset = xr.open_dataset(
        full_url,
        engine="h5netcdf",
        group="Grid",
        decode_times=False,
    )

    try:

        precipitation = dataset[
            "precipitation"
        ].values

        lat = dataset[
            "lat"
        ].values

        lon = dataset[
            "lon"
        ].values

        time_value = dataset[
            "time"
        ].values

    finally:

        dataset.close()

    precipitation = np.asarray(
        precipitation,
        dtype=np.float32
    )

    lat = np.asarray(
        lat,
        dtype=np.float32
    )

    lon = np.asarray(
        lon,
        dtype=np.float32
    )

    time_value = np.asarray(
        time_value
    )

    # NASA OPeNDAP:
    # time, lon, lat
    #
    # Project storage:
    # time, lat, lon

    if precipitation.ndim != 3:

        raise ValueError(
            f"Unexpected precipitation shape: "
            f"{precipitation.shape}"
        )

    precipitation = np.transpose(
        precipitation,
        (0, 2, 1)
    )

    if precipitation.shape != (
        1,
        EXPECTED_LAT_SIZE,
        EXPECTED_LON_SIZE,
    ):

        raise ValueError(
            f"Unexpected regional shape: "
            f"{precipitation.shape}"
        )

    return (
        precipitation,
        lat,
        lon,
        time_value,
    )


# =============================================================================
# DOWNLOAD ONE FRAME
# =============================================================================

def download_frame(
    filename,
    target_date,
    existing_frames,
):

    frame_datetime = get_frame_datetime(
        filename
    )

    if frame_datetime is None:

        print(
            f"[SKIP] Cannot parse datetime: "
            f"{filename}"
        )

        return False

    frame_time = (
        frame_datetime.strftime("%H%M")
    )

    frame_key = (
        target_date,
        frame_time
    )

    if frame_key in existing_frames:

        return False

    output_file = (
        OUTPUT_DIR
        / (
            f"{target_date.strftime('%Y%m%d')}_"
            f"{frame_time}.npz"
        )
    )

    if output_file.exists():

        if is_valid_npz(
            output_file
        ):

            existing_frames.add(
                frame_key
            )

            return False

    url = build_opendap_url(
        filename
    )

    constraint = build_constraint()

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            (
                precipitation,
                lat,
                lon,
                time_value,
            ) = parse_response(
                url,
                constraint
            )

            np.savez_compressed(
                output_file,
                precipitation=precipitation,
                lat=lat,
                lon=lon,
                time=time_value,
            )

            if not is_valid_npz(
                output_file
            ):

                try:
                    output_file.unlink()
                except Exception:
                    pass

                raise ValueError(
                    "Saved NPZ failed validation."
                )

            existing_frames.add(
                frame_key
            )

            print(
                f"[OK] {target_date} "
                f"{frame_time}"
            )

            return True

        except Exception as exc:

            print(
                f"[RETRY "
                f"{attempt}/{MAX_RETRIES}] "
                f"{target_date} "
                f"{frame_time}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

    print(
        f"[FAILED] "
        f"{target_date} "
        f"{frame_time}"
    )

    return False


# =============================================================================
# PROCESS ONE DAY
# =============================================================================

def process_day(
    target_date,
    existing_frames,
):

    expected_times = [
        f"{hour:02d}{minute:02d}"
        for hour in range(24)
        for minute in (0, 30)
    ]

    existing_count = sum(
        1
        for time_part in expected_times
        if (
            target_date,
            time_part
        ) in existing_frames
    )

    if existing_count == 48:

        print(
            f"[SKIP COMPLETE] "
            f"{target_date} "
            f"(48/48)"
        )

        return 0

    print()
    print("-" * 80)
    print(
        f"PROCESSING DAY: "
        f"{target_date}"
    )
    print(
        f"Existing frames: "
        f"{existing_count}/48"
    )
    print("-" * 80)

    try:

        granules = search_day(
            target_date
        )

    except Exception as exc:

        print(
            f"[DAY SEARCH FAILED] "
            f"{target_date}: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return 0

    if not granules:

        print(
            f"[NO GRANULES] "
            f"{target_date}"
        )

        return 0

    print(
        f"NASA granules found: "
        f"{len(granules)}"
    )

    downloaded_today = 0

    for granule in granules:

        filename = get_granule_filename(
            granule
        )

        if filename is None:
            continue

        frame_datetime = get_frame_datetime(
            filename
        )

        if frame_datetime is None:
            continue

        if (
            frame_datetime.date()
            != target_date
        ):
            continue

        frame_time = (
            frame_datetime.strftime(
                "%H%M"
            )
        )

        frame_key = (
            target_date,
            frame_time
        )

        if frame_key in existing_frames:
            continue

        success = download_frame(
            filename,
            target_date,
            existing_frames,
        )

        if success:
            downloaded_today += 1

    final_count = sum(
        1
        for time_part in expected_times
        if (
            target_date,
            time_part
        ) in existing_frames
    )

    print(
        f"[DAY COMPLETE] "
        f"{target_date} | "
        f"Now: {final_count}/48 | "
        f"Downloaded this run: "
        f"{downloaded_today}"
    )

    return downloaded_today


# =============================================================================
# NASA AUTHENTICATION
# =============================================================================

def authenticate():

    print("=" * 80)
    print("[1/3] NASA Earthdata authentication...")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # We first try the normal persisted authentication.
    # If the NASA authentication connection gets reset, retry.
    #
    # On final failure, use explicit interactive authentication.
    # -------------------------------------------------------------------------

    last_error = None

    for attempt in range(
        1,
        AUTH_RETRIES + 1
    ):

        try:

            print(
                f"Authentication attempt "
                f"{attempt}/{AUTH_RETRIES}..."
            )

            auth = earthaccess.login(
                persist=True
            )

            if auth:

                print(
                    "[OK] NASA Earthdata "
                    "authentication successful."
                )

                print()

                return auth

        except Exception as exc:

            last_error = exc

            print(
                f"[AUTH RETRY] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            if attempt < AUTH_RETRIES:

                print(
                    f"Waiting "
                    f"{AUTH_RETRY_DELAY} seconds "
                    f"before retry..."
                )

                time.sleep(
                    AUTH_RETRY_DELAY
                )

    # -------------------------------------------------------------------------
    # Interactive fallback
    # -------------------------------------------------------------------------

    print()
    print(
        "[INFO] Normal authentication attempts "
        "failed."
    )

    print(
        "[INFO] Starting explicit interactive "
        "Earthdata login..."
    )

    try:

        auth = earthaccess.login(
            strategy="interactive",
            persist=True,
        )

        if auth:

            print(
                "[OK] Interactive NASA Earthdata "
                "authentication successful."
            )

            print()

            return auth

    except Exception as exc:

        last_error = exc

    raise RuntimeError(
        "NASA Earthdata authentication failed "
        "after retries and interactive fallback. "
        f"Last error: {last_error}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    safety_check()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("IMERG DIRECT REGIONAL DOWNLOADER")
    print("SERVER-SIDE REGIONAL OPeNDAP SUBSET")
    print("=" * 80)

    print(
        f"Date range       : "
        f"{START_DATE} → {END_DATE}"
    )

    print(
        "Months           : "
        "June, July, August, September"
    )

    print(
        f"Region           : "
        f"{LAT_MIN}-{LAT_MAX} N, "
        f"{LON_MIN}-{LON_MAX} E"
    )

    print(
        "Download method  : "
        "SERVER-SIDE OPeNDAP SUBSET"
    )

    print(
        "Global HDF5      : "
        "NEVER DOWNLOADED"
    )

    print(
        f"Output directory : "
        f"{OUTPUT_DIR}"
    )

    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # TARGET DATES
    # -------------------------------------------------------------------------

    target_dates = get_target_dates()

    total_frames = (
        len(target_dates) * 48
    )

    print(
        f"Total target days   : "
        f"{len(target_dates):,}"
    )

    print(
        f"Total target frames : "
        f"{total_frames:,}"
    )

    print()

    # -------------------------------------------------------------------------
    # INVENTORY
    # -------------------------------------------------------------------------

    (
        existing_frames,
        complete_days,
        incomplete_days,
        missing_frames,
    ) = build_local_inventory(
        target_dates
    )

    # -------------------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------------------

    existing_count = len(
        existing_frames
    )

    coverage = (
        existing_count
        / total_frames
        * 100
    )

    remaining_coverage = (
        100 - coverage
    )

    print()
    print("=" * 80)
    print("ACQUISITION STATUS")
    print("=" * 80)

    print(
        f"Already downloaded : "
        f"{existing_count:,}/"
        f"{total_frames:,}"
    )

    print(
        f"Coverage           : "
        f"{coverage:.2f}%"
    )

    print(
        f"Complete days      : "
        f"{complete_days:,}/"
        f"{len(target_dates):,}"
    )

    print(
        f"Incomplete days    : "
        f"{incomplete_days:,}"
    )

    print(
        f"Missing frames     : "
        f"{missing_frames:,}"
    )

    print(
        f"Remaining coverage : "
        f"{remaining_coverage:.2f}%"
    )

    print("=" * 80)

    # -------------------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------------------

    if missing_frames == 0:

        print()
        print(
            "[COMPLETE] "
            "All target frames already exist."
        )

        return

    # -------------------------------------------------------------------------
    # AUTHENTICATION
    # -------------------------------------------------------------------------

    authenticate()

    # -------------------------------------------------------------------------
    # DOWNLOAD
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("[2/3] DOWNLOADING MISSING FRAMES")
    print("=" * 80)
    print()

    run_start = time.time()

    downloaded_total = 0

    processed_incomplete_days = 0

    for index, target_date in enumerate(
        target_dates,
        start=1
    ):

        day_count = sum(
            1
            for hour in range(24)
            for minute in (0, 30)
            if (
                target_date,
                f"{hour:02d}{minute:02d}"
            ) in existing_frames
        )

        # ---------------------------------------------------------------------
        # COMPLETE DAY — NO NASA SEARCH
        # ---------------------------------------------------------------------

        if day_count == 48:
            continue

        processed_incomplete_days += 1

        print()
        print(
            f"[INCOMPLETE DAY "
            f"{processed_incomplete_days}/"
            f"{incomplete_days}] "
            f"Calendar position "
            f"{index}/"
            f"{len(target_dates)}"
        )

        downloaded_today = process_day(
            target_date,
            existing_frames,
        )

        downloaded_total += (
            downloaded_today
        )

        # ---------------------------------------------------------------------
        # OVERALL PROGRESS
        # ---------------------------------------------------------------------

        current_existing = len(
            existing_frames
        )

        current_missing = (
            total_frames
            - current_existing
        )

        current_coverage = (
            current_existing
            / total_frames
            * 100
        )

        elapsed_minutes = (
            time.time()
            - run_start
        ) / 60

        print()
        print(
            f"OVERALL PROGRESS | "
            f"Existing: "
            f"{current_existing:,}/"
            f"{total_frames:,} | "
            f"Coverage: "
            f"{current_coverage:.2f}% | "
            f"Remaining: "
            f"{current_missing:,} | "
            f"Downloaded this run: "
            f"{downloaded_total:,} | "
            f"Elapsed: "
            f"{elapsed_minutes:.1f} min"
        )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    final_existing = len(
        existing_frames
    )

    final_missing = (
        total_frames
        - final_existing
    )

    final_coverage = (
        final_existing
        / total_frames
        * 100
    )

    elapsed_minutes = (
        time.time()
        - run_start
    ) / 60

    print()
    print("=" * 80)
    print("[3/3] FINAL ACQUISITION STATUS")
    print("=" * 80)

    print(
        f"Target frames     : "
        f"{total_frames:,}"
    )

    print(
        f"Existing frames   : "
        f"{final_existing:,}"
    )

    print(
        f"Downloaded run    : "
        f"{downloaded_total:,}"
    )

    print(
        f"Missing frames    : "
        f"{final_missing:,}"
    )

    print(
        f"Final coverage    : "
        f"{final_coverage:.2f}%"
    )

    print(
        f"Runtime           : "
        f"{elapsed_minutes:.2f} minutes"
    )

    print("=" * 80)

    if final_missing == 0:

        print(
            "SUCCESS: "
            "All target IMERG frames acquired."
        )

    else:

        print(
            f"PARTIAL COMPLETE: "
            f"{final_missing:,} frames remain."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()