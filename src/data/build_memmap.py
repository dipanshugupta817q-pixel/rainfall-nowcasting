from pathlib import Path
from datetime import datetime
import json
import re

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imerg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "memmap"
)

OUTPUT_ARRAY = (
    OUTPUT_DIR
    / "imerg_rainfall.npy"
)

OUTPUT_INDEX = (
    OUTPUT_DIR
    / "frame_index.json"
)

# ============================================================
# DATA CONFIG
# ============================================================

HEIGHT = 20
WIDTH = 30

DTYPE = np.float32


# ============================================================
# TIMESTAMP PARSER
# ============================================================

CANONICAL_PATTERN = re.compile(
    r"^(\d{8})_(\d{4})\.npz$"
)

IMERG_PATTERN = re.compile(
    r"\.3IMERG\.(\d{8})-S(\d{2})(\d{2})(\d{2})-"
)


def parse_timestamp(path):

    name = path.name

    # Canonical:
    # YYYYMMDD_HHMM.npz

    match = CANONICAL_PATTERN.match(name)

    if match:

        date_part = match.group(1)

        time_part = match.group(2)

        return datetime.strptime(
            date_part + time_part,
            "%Y%m%d%H%M"
        )

    # Original IMERG:
    # ...YYYYMMDD-SHHMMSS-...

    match = IMERG_PATTERN.search(name)

    if match:

        date_part = match.group(1)

        hour = match.group(2)

        minute = match.group(3)

        return datetime.strptime(
            date_part + hour + minute,
            "%Y%m%d%H%M"
        )

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("IMERG → MEMORY-MAPPED DATASET")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    print()
    print("Scanning NPZ files...")

    files = list(
        DATA_DIR.glob("*.npz")
    )

    print(
        "NPZ files found:",
        f"{len(files):,}"
    )

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    timestamp_files = {}

    unrecognized = 0

    for path in files:

        timestamp = parse_timestamp(
            path
        )

        if timestamp is None:

            unrecognized += 1

            continue

        # Known 2022 test-day files are
        # outside Model A's dataset period.
        if timestamp.date() == datetime(
            2022,
            6,
            1
        ).date():

            continue

        timestamp_files[timestamp] = path

    timestamps = sorted(
        timestamp_files.keys()
    )

    print(
        "Unique historical frames:",
        f"{len(timestamps):,}"
    )

    print(
        "Unrecognized files:",
        f"{unrecognized:,}"
    )

    if not timestamps:

        raise RuntimeError(
            "No valid IMERG frames found."
        )

    # --------------------------------------------------------
    # Expected size
    # --------------------------------------------------------

    expected_bytes = (
        len(timestamps)
        * HEIGHT
        * WIDTH
        * np.dtype(DTYPE).itemsize
    )

    expected_mb = (
        expected_bytes
        / 1024**2
    )

    print()
    print(
        "Memmap shape:",
        f"({len(timestamps)}, {HEIGHT}, {WIDTH})"
    )

    print(
        "Expected size:",
        f"{expected_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Remove old output
    # --------------------------------------------------------

    if OUTPUT_ARRAY.exists():

        print()
        print(
            "Removing existing memmap..."
        )

        OUTPUT_ARRAY.unlink()

    # --------------------------------------------------------
    # Create memmap
    # --------------------------------------------------------

    print()
    print("Creating memmap...")

    rainfall = np.lib.format.open_memmap(
        OUTPUT_ARRAY,
        mode="w+",
        dtype=DTYPE,
        shape=(
            len(timestamps),
            HEIGHT,
            WIDTH
        )
    )

    # --------------------------------------------------------
    # Frame index
    # --------------------------------------------------------

    frame_index = {}

    # --------------------------------------------------------
    # Convert frames
    # --------------------------------------------------------

    for i, timestamp in enumerate(
        timestamps
    ):

        path = timestamp_files[
            timestamp
        ]

        data = np.load(
            path,
            allow_pickle=False
        )

        precipitation = data[
            "precipitation"
        ]

        # Expected source shape:
        # (1, 20, 30)

        if precipitation.ndim == 3:

            precipitation = (
                precipitation[0]
            )

        if precipitation.shape != (
            HEIGHT,
            WIDTH
        ):

            raise ValueError(
                f"Unexpected shape "
                f"{precipitation.shape} "
                f"in {path.name}"
            )

        precipitation = np.asarray(
            precipitation,
            dtype=DTYPE
        )

        if not np.isfinite(
            precipitation
        ).all():

            raise ValueError(
                f"Invalid values in "
                f"{path.name}"
            )

        if (
            precipitation < 0
        ).any():

            raise ValueError(
                f"Negative rainfall in "
                f"{path.name}"
            )

        rainfall[i] = precipitation

        frame_index[
            timestamp.isoformat()
        ] = i

        if (
            (i + 1) % 5000 == 0
            or i == len(timestamps) - 1
        ):

            progress = (
                (i + 1)
                / len(timestamps)
                * 100
            )

            print(
                f"\rProgress: "
                f"{i + 1:,}/"
                f"{len(timestamps):,} "
                f"({progress:.1f}%)",
                end=""
            )

    print()

    # Flush data to disk

    rainfall.flush()

    del rainfall

    # --------------------------------------------------------
    # Save timestamp → frame index
    # --------------------------------------------------------

    with open(
        OUTPUT_INDEX,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            frame_index,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    print()
    print("Verifying memmap...")

    mmap = np.load(
        OUTPUT_ARRAY,
        mmap_mode="r"
    )

    print(
        "Shape:",
        mmap.shape
    )

    print(
        "Dtype:",
        mmap.dtype
    )

    print(
        "First frame min:",
        float(mmap[0].min())
    )

    print(
        "First frame max:",
        float(mmap[0].max())
    )

    print(
        "Last frame min:",
        float(mmap[-1].min())
    )

    print(
        "Last frame max:",
        float(mmap[-1].max())
    )

    assert mmap.shape == (
        len(timestamps),
        HEIGHT,
        WIDTH
    )

    assert mmap.dtype == np.float32

    assert np.isfinite(
        mmap[0]
    ).all()

    assert (
        mmap[0].min() >= 0
    )

    assert (
        mmap[-1].min() >= 0
    )

    print()
    print("=" * 70)
    print("STATUS: IMERG MEMMAP CREATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(
        "Array:",
        OUTPUT_ARRAY
    )

    print(
        "Index:",
        OUTPUT_INDEX
    )


if __name__ == "__main__":

    main()