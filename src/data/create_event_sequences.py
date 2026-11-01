from pathlib import Path
from datetime import datetime, timedelta
import csv
import re

import numpy as np

from load_data import load_imerg_file, crop_region


# ============================================================
# CONFIG
# ============================================================

RAW_DIR = Path("data/raw")

EVENT_FILE = Path(
    "data/processed/event_windows.csv"
)

OUTPUT_DIR = Path(
    "data/processed/sequences"
)

LAT_MIN = 28.0
LAT_MAX = 30.0

LON_MIN = 76.0
LON_MAX = 79.0

INPUT_FRAMES = 12
TARGET_FRAMES = 12

INTERVAL = timedelta(minutes=30)


# ============================================================
# FILENAME → DATETIME
# ============================================================

def extract_datetime(filename):

    match = re.search(
        r"3IMERG\.(\d{8})-S(\d{6})",
        filename
    )

    if not match:
        return None

    return datetime.strptime(
        match.group(1) + match.group(2),
        "%Y%m%d%H%M%S"
    )


# ============================================================
# BUILD FILE INDEX
# ============================================================

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


# ============================================================
# LOAD ONE FRAME
# ============================================================

def load_frame(file_path):

    data = load_imerg_file(
        file_path
    )

    regional = crop_region(
        data,
        LAT_MIN,
        LAT_MAX,
        LON_MIN,
        LON_MAX
    )

    rainfall = regional[
        "precipitation"
    ]

    rainfall = np.asarray(
        rainfall,
        dtype=np.float32
    )

    # Remove time dimension.
    # Expected:
    # (1, lat, lon)
    # ->
    # (lat, lon)

    if rainfall.ndim == 3:

        if rainfall.shape[0] != 1:
            raise ValueError(
                f"Unexpected time dimension: "
                f"{rainfall.shape}"
            )

        rainfall = rainfall[0]

    if rainfall.ndim != 2:

        raise ValueError(
            f"Expected 2D rainfall frame, "
            f"got {rainfall.shape}"
        )

    return rainfall


# ============================================================
# CREATE ONE EVENT
# ============================================================

def create_event_sequence(
    event,
    file_index
):

    event_id = event["event_id"]

    target_start = datetime.strptime(
        event["target_start"],
        "%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------
    # X starts 6 hours before target start.
    # --------------------------------------------------------

    input_start = (
        target_start
        - INTERVAL * INPUT_FRAMES
    )

    timestamps = []

    for i in range(
        INPUT_FRAMES + TARGET_FRAMES
    ):

        timestamps.append(
            input_start
            + INTERVAL * i
        )

    # --------------------------------------------------------
    # Check all files exist
    # --------------------------------------------------------

    missing = [
        dt
        for dt in timestamps
        if dt not in file_index
    ]

    if missing:

        raise FileNotFoundError(
            f"{event_id}: "
            f"{len(missing)} frames missing"
        )

    # --------------------------------------------------------
    # Load frames
    # --------------------------------------------------------

    frames = []

    for dt in timestamps:

        file_path = file_index[dt]

        print(
            f"  Loading "
            f"{dt:%Y-%m-%d %H:%M} "
            f"{file_path.name}"
        )

        frame = load_frame(
            file_path
        )

        frames.append(frame)

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    sequence = np.stack(
        frames,
        axis=0
    )

    # Expected:
    # (24, 20, 30)

    if sequence.shape[0] != 24:

        raise ValueError(
            f"Wrong temporal dimension: "
            f"{sequence.shape}"
        )

    # --------------------------------------------------------
    # Split X / Y
    # --------------------------------------------------------

    X = sequence[
        :INPUT_FRAMES
    ]

    Y = sequence[
        INPUT_FRAMES:
    ]

    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    if X.shape != (12, 20, 30):

        raise ValueError(
            f"Unexpected X shape: "
            f"{X.shape}"
        )

    if Y.shape != (12, 20, 30):

        raise ValueError(
            f"Unexpected Y shape: "
            f"{Y.shape}"
        )

    if np.isnan(X).any():

        raise ValueError(
            "NaN found in X"
        )

    if np.isnan(Y).any():

        raise ValueError(
            "NaN found in Y"
        )

    # Rainfall cannot be negative.

    if np.any(X < 0):

        raise ValueError(
            "Negative rainfall found in X"
        )

    if np.any(Y < 0):

        raise ValueError(
            "Negative rainfall found in Y"
        )

    return X, Y, timestamps


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREATING EVENT SEQUENCES")
    print("=" * 70)

    if not EVENT_FILE.exists():

        raise FileNotFoundError(
            f"Event file not found: "
            f"{EVENT_FILE}"
        )

    file_index = build_file_index()

    print(
        f"Available raw frames : "
        f"{len(file_index)}"
    )

    events = []

    with open(
        EVENT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            events.append(row)

    print(
        f"Events found         : "
        f"{len(events)}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    successful = 0

    for event in events:

        event_id = event[
            "event_id"
        ]

        print(
            f"\n[EVENT] {event_id}"
        )

        try:

            X, Y, timestamps = (
                create_event_sequence(
                    event,
                    file_index
                )
            )

            output_file = (
                OUTPUT_DIR
                / f"{event_id}.npz"
            )

            np.savez_compressed(
                output_file,
                X=X,
                Y=Y,
                timestamps=np.array(
                    [
                        int(
                            dt.timestamp()
                        )
                        for dt in timestamps
                    ],
                    dtype=np.int64
                )
            )

            print(
                f"  X shape : {X.shape}"
            )

            print(
                f"  Y shape : {Y.shape}"
            )

            print(
                f"  X range : "
                f"{X.min():.4f} → "
                f"{X.max():.4f}"
            )

            print(
                f"  Y range : "
                f"{Y.min():.4f} → "
                f"{Y.max():.4f}"
            )

            print(
                f"  Saved   : {output_file}"
            )

            successful += 1

        except Exception as e:

            print(
                f"  [ERROR] {e}"
            )

    print("\n" + "=" * 70)
    print("SEQUENCE CREATION COMPLETE")
    print("=" * 70)

    print(
        f"Successful events : "
        f"{successful}/{len(events)}"
    )

    print(
        f"Output directory  : "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()