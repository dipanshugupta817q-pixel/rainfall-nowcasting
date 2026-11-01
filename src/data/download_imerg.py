"""
NASA IMERG Regional Downloader
--------------------------------
Parallel, resumable, server-side cropped downloader.

Target:
    2021-06-01 -> 2025-09-30
    June-September only
    Meerut region:
        LAT: 28.05 -> 29.95
        LON: 76.05 -> 78.95

Existing historical files from 2008-2020 are preserved.

Output:
    data/processed/imerg/YYYYMMDD_HHMM.npz

The NASA OPeNDAP server performs the spatial crop BEFORE downloading.
"""

import os
import re
import time
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
)

import numpy as np
import requests
import earthaccess
import h5py


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\projects\rainfall-nowcasting")

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "imerg"
LOG_DIR = PROJECT_ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATE RANGE
# ============================================================

# Additional data required:
# 2021-2025
START_DATE = datetime(2021, 6, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 9, 30, tzinfo=timezone.utc)

TARGET_MONTHS = {6, 7, 8, 9}


# ============================================================
# NASA COLLECTION
# ============================================================

COLLECTION_ID = "C2723754847-GES_DISC"

SHORT_NAME = "GPM_3IMERGHH"
VERSION = "07"


# ============================================================
# REGIONAL CROP
# ============================================================

LAT_START = 28.05
LAT_END = 29.95

LON_START = 76.05
LON_END = 78.95

# Verified IMERG grid indices
LAT_I0 = 1180
LAT_I1 = 1199

LON_I0 = 2560
LON_I1 = 2589


# ============================================================
# PARALLEL DOWNLOAD SETTINGS
# ============================================================

WORKERS = 64

QUEUE_MULTIPLIER = 4

CONNECT_TIMEOUT = 20
READ_TIMEOUT = 90

# Individual HTTP request retries
REQUEST_RETRIES = 4

# Complete failed-download retry rounds
GLOBAL_RETRY_ROUNDS = 2


# ============================================================
# METADATA RETRY SETTINGS
# ============================================================

# NASA Earthdata metadata lookup can occasionally fail
# because of transient network/authentication/API issues.
METADATA_RETRIES = 4

# Exponential backoff:
# 3s -> 6s -> 12s
METADATA_BACKOFF = 3


# ============================================================
# SERVER-SIDE DAP4 CONSTRAINT
# ============================================================

CONSTRAINT = (
    "/Grid/lat[1180:1:1199];"
    "/Grid/lon[2560:1:2589];"
    "/Grid/precipitation[0][2560:1:2589][1180:1:1199]"
)


# ============================================================
# THREAD LOCAL SESSION
# ============================================================

_thread_local = threading.local()


def get_session(auth):
    """
    Give each worker thread its own HTTP session.
    """

    if not hasattr(_thread_local, "session"):
        _thread_local.session = auth.get_session()

    return _thread_local.session


# ============================================================
# FILE / INVENTORY HELPERS
# ============================================================

def timestamp_from_filename(name):
    """
    Supports:

        YYYYMMDD_HHMM.npz

    and old:

        3B-HHR.MS.MRG.3IMERG.YYYYMMDD-SHHMMSS-....regional.npz
    """

    # New format
    m = re.match(
        r"^(\d{8})_(\d{4})\.npz$",
        name
    )

    if m:
        return datetime.strptime(
            m.group(1) + m.group(2),
            "%Y%m%d%H%M"
        ).replace(tzinfo=timezone.utc)

    # Old IMERG format
    m = re.search(
        r"\.3IMERG\.(\d{8})-S"
        r"(\d{2})(\d{2})(\d{2})-",
        name
    )

    if m:
        date_str = m.group(1)
        hh = m.group(2)
        mm = m.group(3)

        return datetime.strptime(
            date_str + hh + mm,
            "%Y%m%d%H%M"
        ).replace(tzinfo=timezone.utc)

    return None


def build_inventory():
    """
    Scan existing NPZ files once.

    Existing files are NEVER deleted.
    """

    existing = set()

    files = list(
        OUTPUT_DIR.glob("*.npz")
    )

    print(
        f"Scanning existing NPZ files: "
        f"{len(files):,}"
    )

    for file in files:

        ts = timestamp_from_filename(
            file.name
        )

        if ts is not None:
            existing.add(ts)

    print(
        f"Recognized existing timestamps: "
        f"{len(existing):,}"
    )

    return existing


# ============================================================
# EXPECTED FRAME GENERATOR
# ============================================================

def generate_expected_frames():
    """
    Generate all expected 30-minute timestamps
    for June-September 2021-2025.
    """

    current = START_DATE

    while current <= END_DATE:

        if current.month in TARGET_MONTHS:

            day_start = current.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0
            )

            for i in range(48):

                yield (
                    day_start
                    + timedelta(minutes=30 * i)
                )

        current += timedelta(days=1)


# ============================================================
# NASA METADATA
# ============================================================

def get_day_granules(auth, day):
    """
    Find all IMERG granules for one UTC day.

    Includes retry/backoff because NASA Earthdata
    metadata search can occasionally fail due to
    transient network/API/authentication issues.
    """

    day_start = day.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    day_end = day_start + timedelta(days=1)

    last_error = None

    for attempt in range(
        1,
        METADATA_RETRIES + 1
    ):

        try:

            results = earthaccess.search_data(
                short_name=SHORT_NAME,
                version=VERSION,
                temporal=(
                    day_start,
                    day_end
                ),
            )

            # Explicitly handle unexpected None
            if results is None:

                raise RuntimeError(
                    "NASA Earthdata search returned None"
                )

            granules = {}

            for result in results:

                # Some metadata responses may have
                # missing/invalid link information.
                try:
                    links = (
                        result.data_links()
                        or []
                    )

                except Exception as link_error:

                    print(
                        f"\nWarning: data_links() failed "
                        f"for one granule on "
                        f"{day_start:%Y-%m-%d}: "
                        f"{link_error}",
                        flush=True
                    )

                    continue

                for link in links:

                    if not link:
                        continue

                    # Remove query string if present
                    filename = os.path.basename(
                        link.split("?")[0]
                    )

                    match = re.search(
                        r"\.3IMERG\."
                        r"(\d{8})-S"
                        r"(\d{2})(\d{2})(\d{2})-",
                        filename
                    )

                    if not match:
                        continue

                    date_str = match.group(1)
                    hh = match.group(2)
                    mm = match.group(3)

                    if (
                        date_str
                        != day_start.strftime("%Y%m%d")
                    ):
                        continue

                    ts = datetime.strptime(
                        date_str + hh + mm,
                        "%Y%m%d%H%M"
                    ).replace(
                        tzinfo=timezone.utc
                    )

                    granules[ts] = filename

            # IMERG should provide 48 half-hour
            # granules per complete day.
            if len(granules) == 0:

                raise RuntimeError(
                    f"No IMERG granules found for "
                    f"{day_start:%Y-%m-%d}"
                )

            if len(granules) < 48:

                raise RuntimeError(
                    f"Only {len(granules)}/48 "
                    f"granules found for "
                    f"{day_start:%Y-%m-%d}"
                )

            return granules

        except Exception as exc:

            last_error = exc

            if attempt < METADATA_RETRIES:

                wait_seconds = (
                    METADATA_BACKOFF
                    * (2 ** (attempt - 1))
                )

                print(
                    f"\nMetadata retry "
                    f"{attempt}/"
                    f"{METADATA_RETRIES - 1} "
                    f"for "
                    f"{day_start:%Y-%m-%d} "
                    f"| error: {exc} "
                    f"| waiting "
                    f"{wait_seconds}s...",
                    flush=True
                )

                time.sleep(
                    wait_seconds
                )

            else:

                raise RuntimeError(
                    f"Metadata lookup failed after "
                    f"{METADATA_RETRIES} attempts "
                    f"for "
                    f"{day_start:%Y-%m-%d}: "
                    f"{last_error}"
                )


# ============================================================
# OPENDAP URL
# ============================================================

def build_opendap_url(filename):
    """
    Build verified NASA OPeNDAP server-side subset URL.

    The ':' between collection version and
    granule name requires double URL encoding.
    """

    granule_id = (
        f"GPM_3IMERGHH.07:{filename}"
    )

    encoded_granule = quote(
        quote(
            granule_id,
            safe=""
        ),
        safe=""
    )

    encoded_constraint = quote(
        CONSTRAINT,
        safe="/:;"
    )

    url = (
        f"https://opendap.earthdata.nasa.gov/"
        f"collections/{COLLECTION_ID}/"
        f"granules/{encoded_granule}.dap.nc4"
        f"?dap4.ce={encoded_constraint}"
    )

    return url


# ============================================================
# HDF5 DECODER
# ============================================================

def read_subset_hdf5(path):
    """
    Read downloaded DAP4 HDF5 subset.

    Returns:

        precipitation -> (1, 20, 30)
        lat            -> (20,)
        lon            -> (30,)
    """

    with h5py.File(
        path,
        "r"
    ) as f:

        grid = f["Grid"]

        precip_ds = grid[
            "precipitation"
        ]

        lat_ds = grid["lat"]
        lon_ds = grid["lon"]

        precip = precip_ds[...]

        lat = lat_ds[...]
        lon = lon_ds[...]

        # ----------------------------------------------------
        # Apply scale/offset
        # ----------------------------------------------------

        if "scale_factor" in precip_ds.attrs:

            scale = precip_ds.attrs[
                "scale_factor"
            ]

            precip = (
                precip * scale
            )

        if "add_offset" in precip_ds.attrs:

            offset = precip_ds.attrs[
                "add_offset"
            ]

            precip = (
                precip + offset
            )

        # ----------------------------------------------------
        # Fill value -> NaN
        # ----------------------------------------------------

        fill_value = precip_ds.attrs.get(
            "_FillValue",
            None
        )

        if fill_value is not None:

            precip = precip.astype(
                np.float32
            )

            precip[
                np.isclose(
                    precip,
                    fill_value,
                    equal_nan=False
                )
            ] = np.nan

        # ----------------------------------------------------
        # Shape correction
        # ----------------------------------------------------

        if precip.ndim == 3:

            # Server returns:
            # (time, lon, lat)

            if precip.shape == (
                1,
                30,
                20
            ):

                precip = np.transpose(
                    precip,
                    (0, 2, 1)
                )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if precip.shape != (
            1,
            20,
            30
        ):

            raise ValueError(
                f"Unexpected precipitation "
                f"shape: {precip.shape}"
            )

        if lat.shape != (20,):

            raise ValueError(
                f"Unexpected latitude "
                f"shape: {lat.shape}"
            )

        if lon.shape != (30,):

            raise ValueError(
                f"Unexpected longitude "
                f"shape: {lon.shape}"
            )

        return (
            precip.astype(np.float32),
            lat.astype(np.float32),
            lon.astype(np.float32),
        )


# ============================================================
# DOWNLOAD ONE FRAME
# ============================================================

def download_one(task, auth):
    """
    Download and save one IMERG frame.
    """

    ts, source_filename = task

    output_filename = (
        f"{ts.strftime('%Y%m%d_%H%M')}.npz"
    )

    output_path = (
        OUTPUT_DIR / output_filename
    )

    # Already completed
    if output_path.exists():

        return {
            "status": "skipped",
            "timestamp": ts,
            "filename": output_filename,
            "bytes": output_path.stat().st_size,
            "elapsed": 0,
            "error": "",
        }

    url = build_opendap_url(
        source_filename
    )

    start = time.time()

    last_error = ""

    for attempt in range(
        1,
        REQUEST_RETRIES + 1
    ):

        temp_path = None

        try:

            session = get_session(
                auth
            )

            response = session.get(
                url,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT
                )
            )

            response.raise_for_status()

            data = response.content

            if len(data) < 1000:

                raise ValueError(
                    f"Response suspiciously "
                    f"small: {len(data)} bytes"
                )

            # ------------------------------------------------
            # Temporary HDF5
            # ------------------------------------------------

            fd, temp_name = tempfile.mkstemp(
                suffix=".h5",
                dir=str(OUTPUT_DIR)
            )

            os.close(fd)

            temp_path = Path(
                temp_name
            )

            temp_path.write_bytes(
                data
            )

            # ------------------------------------------------
            # Decode
            # ------------------------------------------------

            precip, lat, lon = (
                read_subset_hdf5(
                    temp_path
                )
            )

            # ------------------------------------------------
            # Atomic NPZ save
            # ------------------------------------------------

            tmp_npz = (
                output_path.with_suffix(
                    ".npz.tmp"
                )
            )

            np.savez_compressed(
                tmp_npz,
                precipitation=precip,
                lat=lat,
                lon=lon,
            )

            actual_tmp = tmp_npz

            # numpy may append .npz
            if not actual_tmp.exists():

                candidate = Path(
                    str(tmp_npz) + ".npz"
                )

                if candidate.exists():

                    actual_tmp = candidate

            actual_tmp.replace(
                output_path
            )

            elapsed = (
                time.time() - start
            )

            return {
                "status": "success",
                "timestamp": ts,
                "filename": output_filename,
                "bytes": len(data),
                "elapsed": elapsed,
                "error": "",
            }

        except Exception as e:

            last_error = (
                f"attempt "
                f"{attempt}/{REQUEST_RETRIES}: "
                f"{type(e).__name__}: {e}"
            )

            time.sleep(
                min(
                    2 ** attempt,
                    10
                )
            )

        finally:

            if temp_path is not None:

                try:

                    temp_path.unlink(
                        missing_ok=True
                    )

                except Exception:
                    pass

    return {
        "status": "failed",
        "timestamp": ts,
        "filename": output_filename,
        "bytes": 0,
        "elapsed": (
            time.time() - start
        ),
        "error": last_error,
    }


# ============================================================
# SPEED TRACKER
# ============================================================

class SpeedTracker:

    def __init__(self):

        self.start_time = (
            time.time()
        )

        self.total_bytes = 0

        self.history = []

        self.lock = threading.Lock()

    def add(self, nbytes):

        now = time.time()

        with self.lock:

            self.total_bytes += (
                nbytes
            )

            self.history.append(
                (
                    now,
                    self.total_bytes
                )
            )

            cutoff = (
                now - 30
            )

            while (
                len(self.history) > 2
                and self.history[0][0]
                < cutoff
            ):

                self.history.pop(0)

    def speed(self):

        with self.lock:

            if len(self.history) < 2:
                return 0.0

            t0, b0 = (
                self.history[0]
            )

            t1, b1 = (
                self.history[-1]
            )

            dt = t1 - t0

            if dt <= 0:
                return 0.0

            return (
                (b1 - b0) / dt
            )

    def average_speed(self):

        elapsed = (
            time.time()
            - self.start_time
        )

        if elapsed <= 0:
            return 0

        return (
            self.total_bytes
            / elapsed
        )


# ============================================================
# FORMATTERS
# ============================================================

def format_bytes(n):

    if n < 1024:
        return f"{n:.0f} B"

    if n < 1024 ** 2:
        return (
            f"{n / 1024:.1f} KB"
        )

    if n < 1024 ** 3:
        return (
            f"{n / 1024 ** 2:.2f} MB"
        )

    return (
        f"{n / 1024 ** 3:.2f} GB"
    )


def format_speed(n):

    if n <= 0:
        return "---"

    if n < 1024:
        return f"{n:.1f} B/s"

    if n < 1024 ** 2:
        return (
            f"{n / 1024:.1f} KB/s"
        )

    return (
        f"{n / 1024 ** 2:.2f} MB/s"
    )


def format_time(seconds):

    if (
        seconds is None
        or seconds <= 0
    ):
        return "--"

    seconds = int(seconds)

    h = seconds // 3600
    m = (
        seconds % 3600
    ) // 60
    s = seconds % 60

    if h:
        return (
            f"{h}h "
            f"{m:02d}m "
            f"{s:02d}s"
        )

    if m:
        return (
            f"{m}m "
            f"{s:02d}s"
        )

    return f"{s}s"


# ============================================================
# PROGRESS BAR
# ============================================================

def progress_bar(
    current,
    total,
    width=55
):

    if total <= 0:

        return (
            "["
            + "#" * width
            + "]"
        )

    ratio = min(
        max(
            current / total,
            0
        ),
        1
    )

    filled = int(
        width * ratio
    )

    return (
        "["
        + "#" * filled
        + "-"
        * (width - filled)
        + "] "
        + f"{ratio * 100:6.2f}%"
    )


# ============================================================
# DASHBOARD
# ============================================================

def show_dashboard(
    total,
    success,
    failed,
    active,
    tracker,
    start_time,
):

    completed = success

    remaining = max(
        total - completed,
        0
    )

    speed = tracker.speed()

    avg_speed = (
        tracker.average_speed()
    )

    if speed > 0:

        avg_bytes = (
            tracker.total_bytes
            / max(success, 1)
        )

        remaining_bytes = (
            remaining * avg_bytes
        )

        eta = (
            remaining_bytes
            / speed
        )

    else:

        eta = None

    elapsed = (
        time.time()
        - start_time
    )

    print(
        "\033[2J\033[H",
        end=""
    )

    print("=" * 76)

    print(
        "                 "
        "NASA IMERG DOWNLOAD MANAGER"
    )

    print("=" * 76)

    print()

    print(
        f"Overall     : "
        f"{completed:,} / "
        f"{total:,} frames"
    )

    print(
        progress_bar(
            completed,
            total
        )
    )

    print()

    print(
        f"Workers     : "
        f"{active}/{WORKERS}"
    )

    print(
        f"Speed       : "
        f"{format_speed(speed)} "
        f"(rolling 30s)"
    )

    print(
    f"Avg speed  : "
    f"{format_speed(tracker.average_speed())}"
)

    print(
    f"Downloaded  : "
    f"{format_bytes(tracker.total_bytes)}"
)

    print(
        f"Remaining   : "
        f"{remaining:,} frames"
    )

    print(
        f"ETA         : "
        f"{format_time(eta)}"
    )

    print(
        f"Elapsed     : "
        f"{format_time(elapsed)}"
    )

    print()

    print(
        f"Successful  : "
        f"{success:,}"
    )

    print(
        f"Failed      : "
        f"{failed:,}"
    )

    print()

    print(
        "Existing files are preserved."
    )

    print(
        "Ctrl+C = safe stop; "
        "completed files remain saved."
    )

    print("=" * 76)


# ============================================================
# METADATA DISCOVERY
# ============================================================

def build_download_tasks(
    auth,
    existing
):

    tasks = []

    missing_frames = [
        ts
        for ts in generate_expected_frames()
        if ts not in existing
    ]

    print()

    print(
        f"Missing frames: "
        f"{len(missing_frames):,}"
    )

    # Group missing timestamps by day
    days = {}

    for ts in missing_frames:

        day = ts.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        days.setdefault(
            day,
            []
        ).append(ts)

    print(
        f"Days requiring metadata lookup: "
        f"{len(days):,}"
    )

    print()
    print(
        "Discovering NASA granules..."
    )
    print()

    failed_days = []

    for index, (
        day,
        timestamps
    ) in enumerate(
        sorted(days.items()),
        start=1
    ):

        print(
            f"\rMetadata "
            f"{index}/"
            f"{len(days)} | "
            f"{day.strftime('%Y-%m-%d')} "
            f"| tasks="
            f"{len(tasks):,}",
            end="",
            flush=True
        )

        try:

            granules = (
                get_day_granules(
                    auth,
                    day
                )
            )

            for ts in timestamps:

                filename = (
                    granules.get(ts)
                )

                if filename is None:

                    failed_days.append(
                        (
                            day,
                            ts,
                            "NASA granule not found"
                        )
                    )

                    continue

                tasks.append(
                    (
                        ts,
                        filename
                    )
                )

        except Exception as e:

            failed_days.append(
                (
                    day,
                    None,
                    str(e)
                )
            )

    print()
    print()

    if failed_days:

        log = (
            LOG_DIR
            / "metadata_failures.txt"
        )

        with log.open(
            "a",
            encoding="utf-8"
        ) as f:

            for (
                day,
                ts,
                error
            ) in failed_days:

                f.write(
                    f"{day.isoformat()} | "
                    f"{ts} | "
                    f"{error}\n"
                )

        print(
            f"Metadata failures: "
            f"{len(failed_days):,}"
        )

    print(
        f"Download tasks ready: "
        f"{len(tasks):,}"
    )

    return tasks


# ============================================================
# PARALLEL DOWNLOAD ENGINE
# ============================================================

def run_parallel_download(
    tasks,
    auth
):

    total = len(tasks)

    if total == 0:

        print()
        print(
            "Nothing to download."
        )

        return

    tracker = SpeedTracker()

    success = 0
    failed = 0

    start_time = time.time()

    final_failed = []

    task_iterator = iter(tasks)

    executor = ThreadPoolExecutor(
        max_workers=WORKERS,
        thread_name_prefix="IMERG"
    )

    futures = {}

    try:

        # ----------------------------------------------------
        # Initial queue
        # ----------------------------------------------------

        queue_size = (
            WORKERS
            * QUEUE_MULTIPLIER
        )

        for _ in range(
            queue_size
        ):

            try:

                task = next(
                    task_iterator
                )

            except StopIteration:

                break

            future = executor.submit(
                download_one,
                task,
                auth
            )

            futures[future] = task

        # ----------------------------------------------------
        # Process completed tasks
        # ----------------------------------------------------

        while futures:

            done, _ = wait(
                futures,
                return_when=FIRST_COMPLETED
            )

            for future in done:

                task = futures.pop(
                    future
                )

                try:

                    result = (
                        future.result()
                    )

                except Exception as e:

                    result = {
                        "status": "failed",
                        "timestamp": task[0],
                        "filename": "",
                        "bytes": 0,
                        "elapsed": 0,
                        "error": str(e),
                    }

                status = (
                    result["status"]
                )

                if status == "success":

                    success += 1

                    tracker.add(
                        result["bytes"]
                    )

                elif status == "skipped":

                    success += 1

                else:

                    failed += 1

                    final_failed.append(
                        (
                            task,
                            result["error"]
                        )
                    )

                active = len(
                    futures
                )

                show_dashboard(
                    total,
                    success,
                    failed,
                    active,
                    tracker,
                    start_time,
                )

                # ------------------------------------------------
                # Refill queue
                # ------------------------------------------------

                try:

                    next_task = next(
                        task_iterator
                    )

                    new_future = (
                        executor.submit(
                            download_one,
                            next_task,
                            auth
                        )
                    )

                    futures[
                        new_future
                    ] = next_task

                except StopIteration:

                    pass

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Stopping safely..."
        )

        for future in futures:

            future.cancel()

        executor.shutdown(
            wait=False,
            cancel_futures=True
        )

        print(
            "Downloaded files are preserved."
        )

        return

    finally:

        executor.shutdown(
            wait=True
        )

    # ========================================================
    # GLOBAL RETRIES
    # ========================================================

    for retry_round in range(
        1,
        GLOBAL_RETRY_ROUNDS + 1
    ):

        if not final_failed:
            break

        retry_tasks = [
            item[0]
            for item in final_failed
        ]

        print()
        print(
            f"Global retry round "
            f"{retry_round}/"
            f"{GLOBAL_RETRY_ROUNDS}: "
            f"{len(retry_tasks)} "
            f"failed frames"
        )

        final_failed = []

        retry_executor = (
            ThreadPoolExecutor(
                max_workers=WORKERS,
                thread_name_prefix="IMERG-RETRY"
            )
        )

        retry_futures = {}

        try:

            for task in retry_tasks:

                future = (
                    retry_executor.submit(
                        download_one,
                        task,
                        auth
                    )
                )

                retry_futures[
                    future
                ] = task

            while retry_futures:

                done, _ = wait(
                    retry_futures,
                    return_when=FIRST_COMPLETED
                )

                for future in done:

                    task = (
                        retry_futures.pop(
                            future
                        )
                    )

                    try:

                        result = (
                            future.result()
                        )

                    except Exception as e:

                        result = {
                            "status": "failed",
                            "bytes": 0,
                            "error": str(e),
                        }

                    if result["status"] in (
                        "success",
                        "skipped"
                    ):

                        success += 1

                        if (
                            result["status"]
                            == "success"
                        ):

                            tracker.add(
                                result["bytes"]
                            )

                    else:

                        failed += 1

                        final_failed.append(
                            (
                                task,
                                result.get(
                                    "error",
                                    "unknown error"
                                )
                            )
                        )

                    show_dashboard(
                        total,
                        success,
                        failed,
                        len(
                            retry_futures
                        ),
                        tracker,
                        start_time,
                    )

        finally:

            retry_executor.shutdown(
                wait=True
            )

    # ========================================================
    # FINAL FAILED LOG
    # ========================================================

    if final_failed:

        failed_log = (
            LOG_DIR
            / "imerg_failed_frames.txt"
        )

        with failed_log.open(
            "a",
            encoding="utf-8"
        ) as f:

            for task, error in (
                final_failed
            ):

                ts, filename = task

                f.write(
                    f"{ts.isoformat()} | "
                    f"{filename} | "
                    f"{error}\n"
                )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print()

    print("=" * 76)

    print(
        "                    "
        "DOWNLOAD COMPLETE"
    )

    print("=" * 76)

    print(
        f"Successful : "
        f"{success:,}"
    )

    print(
        f"Failed     : "
        f"{len(final_failed):,}"
    )

    print(
      f"Data       : "
      f"{format_bytes(tracker.total_bytes)}"
)

    print(
    f"Avg speed  : "
    f"{format_speed(tracker.average_speed())}"
)

    print(
        f"Elapsed    : "
        f"{format_time(elapsed)}"
    )

    print("=" * 76)

    if final_failed:

        print()
        print(
            "Some frames failed."
        )

        print(
            f"See: "
            f"{LOG_DIR / 'imerg_failed_frames.txt'}"
        )

    else:

        print()
        print(
            "ALL DOWNLOAD TASKS SUCCESSFUL."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 76)

    print(
        "                 "
        "NASA IMERG PARALLEL DOWNLOADER"
    )

    print("=" * 76)

    print()

    print(
        f"Workers       : "
        f"{WORKERS}"
    )

    print(
        f"Output        : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Date range    : "
        f"{START_DATE.date()} "
        f"-> "
        f"{END_DATE.date()}"
    )

    print(
        "Months        : "
        "June, July, August, September"
    )

    print(
        "Resolution     : "
        "30 minutes"
    )

    print(
        "Spatial crop   : "
        "20 lat x 30 lon"
    )

    print(
        "Download mode  : "
        "Server-side OPeNDAP subset"
    )

    print(
        "Metadata retry : "
        f"{METADATA_RETRIES} attempts"
    )

    print()

    # --------------------------------------------------------
    # NASA AUTH
    # --------------------------------------------------------

    print(
        "Authenticating with NASA Earthdata..."
    )

    auth = earthaccess.login(
        strategy="netrc",
        persist=True
    )

    if not auth:

        raise RuntimeError(
            "NASA Earthdata authentication failed."
        )

    print(
        "NASA authentication: OK"
    )

    # --------------------------------------------------------
    # EXISTING FILES
    # --------------------------------------------------------

    existing = build_inventory()

    # --------------------------------------------------------
    # TASK DISCOVERY
    # --------------------------------------------------------

    tasks = build_download_tasks(
        auth,
        existing
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    run_parallel_download(
        tasks,
        auth
    )


if __name__ == "__main__":
    main()