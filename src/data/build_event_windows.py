from pathlib import Path
from datetime import datetime, timedelta
import csv
import re


RAW_DIR = Path("data/raw")

SCREENING_FILE = Path(
    "data/processed/rainfall_event_screening.csv"
)

OUTPUT_FILE = Path(
    "data/processed/event_windows.csv"
)

INPUT_FRAMES = 12
TARGET_FRAMES = 12
TOTAL_FRAMES = INPUT_FRAMES + TARGET_FRAMES

FRAME_INTERVAL = timedelta(minutes=30)


def extract_datetime(filename):
    """
    Extract start datetime from IMERG filename.

    Example:
    3B-HHR.MS.MRG.3IMERG.
    20240815-S053000-E055959...
    """

    match = re.search(
        r"3IMERG\.(\d{8})-S(\d{6})",
        filename
    )

    if not match:
        return None

    date_part = match.group(1)
    time_part = match.group(2)

    return datetime.strptime(
        date_part + time_part,
        "%Y%m%d%H%M%S"
    )


def build_file_index():

    index = {}

    files = RAW_DIR.glob(
        "3B-HHR.MS.MRG.3IMERG.*.V07B.HDF5"
    )

    for file_path in files:

        dt = extract_datetime(
            file_path.name
        )

        if dt is not None:
            index[dt] = file_path

    return index


def get_required_times(peak_time):
    """
    Build exactly 24 frames.

    X:
        T-6h ... T-30m
        12 frames

    Y:
        T ... T+5h30m
        12 frames
    """

    start_time = (
        peak_time
        - FRAME_INTERVAL * INPUT_FRAMES
    )

    times = []

    for i in range(TOTAL_FRAMES):

        current = (
            start_time
            + FRAME_INTERVAL * i
        )

        times.append(current)

    return times


def main():

    print("=" * 70)
    print("BUILDING 24-FRAME RAINFALL EVENT WINDOWS")
    print("=" * 70)

    file_index = build_file_index()

    print(
        f"Available frames : {len(file_index)}"
    )

    if not file_index:

        print("No IMERG files found.")
        return

    # --------------------------------------------------------
    # Read screening results
    # --------------------------------------------------------

    screened = []

    with open(
        SCREENING_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            dt = extract_datetime(
                row["file"]
            )

            if dt is None:
                continue

            row["datetime"] = dt

            screened.append(row)

    # --------------------------------------------------------
    # Highest rainfall frames first
    # --------------------------------------------------------

    screened.sort(
        key=lambda x: float(
            x["max_rain"]
        ),
        reverse=True
    )

    windows = []
    used_peaks = []

    # --------------------------------------------------------
    # Find complete windows
    # --------------------------------------------------------

    for row in screened:

        peak_time = row["datetime"]

        # Avoid selecting overlapping events.
        # Two peaks must be at least 12 hours apart.

        too_close = any(
            abs(peak_time - previous_peak)
            < timedelta(hours=12)
            for previous_peak in used_peaks
        )

        if too_close:
            continue

        required_times = get_required_times(
            peak_time
        )

        # Check that every required frame exists.

        missing_times = [
            dt
            for dt in required_times
            if dt not in file_index
        ]

        if missing_times:
            continue

        # ----------------------------------------------------
        # Verify exact 30-minute spacing
        # ----------------------------------------------------

        spacing_ok = all(
            required_times[i + 1]
            - required_times[i]
            == FRAME_INTERVAL
            for i in range(
                len(required_times) - 1
            )
        )

        if not spacing_ok:
            continue

        # ----------------------------------------------------
        # Create event record
        # ----------------------------------------------------

        event_id = (
            f"{peak_time:%Y%m%d_%H%M}"
        )

        windows.append({

            "event_id":
                event_id,

            "peak_time":
                peak_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "input_start":
                required_times[0].strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "input_end":
                required_times[
                    INPUT_FRAMES - 1
                ].strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "target_start":
                required_times[
                    INPUT_FRAMES
                ].strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "target_end":
                required_times[-1].strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "peak_max_rain":
                float(row["max_rain"]),

            "peak_mean_rain":
                float(row["mean_rain"]),

            "peak_p95_rain":
                float(row["p95_rain"]),

            "category":
                row["category"],

            "input_frames":
                INPUT_FRAMES,

            "target_frames":
                TARGET_FRAMES,

            "total_frames":
                TOTAL_FRAMES,

            "frames_available":
                len(required_times),

        })

        used_peaks.append(
            peak_time
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "event_id",
        "peak_time",
        "input_start",
        "input_end",
        "target_start",
        "target_end",
        "peak_max_rain",
        "peak_mean_rain",
        "peak_p95_rain",
        "category",
        "input_frames",
        "target_frames",
        "total_frames",
        "frames_available",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(windows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVENT WINDOW SUMMARY")
    print("=" * 70)

    print(
        f"Complete event windows : "
        f"{len(windows)}"
    )

    print(
        f"Required per event     : "
        f"{TOTAL_FRAMES} frames"
    )

    print(
        f"Input                  : "
        f"{INPUT_FRAMES} frames"
    )

    print(
        f"Target                 : "
        f"{TARGET_FRAMES} frames"
    )

    print(
        f"Output                 : "
        f"{OUTPUT_FILE}"
    )

    if windows:

        print("\nFirst events:")

        for event in windows[:10]:

            print(
                f"{event['event_id']} | "
                f"{event['category']} | "
                f"peak={event['peak_max_rain']:.2f} mm/hr"
            )


if __name__ == "__main__":
    main()