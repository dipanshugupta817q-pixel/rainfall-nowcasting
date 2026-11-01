from pathlib import Path
from datetime import datetime, timedelta
import json


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "era5_aligned"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "era5_sequences"

INPUT_FRAMES = 12       # 6 hours
OUTPUT_FRAMES = 12      # next 6 hours
FRAME_MINUTES = 30


# ============================================================
# SPLITS
# ============================================================

SPLITS = {
    "train": (2021, 2023),
    "validation": (2024, 2024),
    "test": (2025, 2025),
}


# ============================================================
# GET AVAILABLE DAILY FILES
# ============================================================

files = sorted(DATA_DIR.glob("*.npz"))

print("=" * 60)
print("MODEL B — ERA5 SEQUENCE INDEX")
print("=" * 60)

print("ERA5 files found:", len(files))

if len(files) != 610:
    raise RuntimeError(
        f"Expected 610 ERA5 files, found {len(files)}"
    )


# ============================================================
# DATE PARSER
# ============================================================

def parse_date(path):

    return datetime.strptime(
        path.stem,
        "%Y%m%d"
    )


file_map = {
    parse_date(f).strftime("%Y%m%d"): str(f)
    for f in files
}


# ============================================================
# BUILD CONTINUOUS 30-MIN TIMELINE
# ============================================================

# Each day has exactly 48 frames.
#
# For a sequence:
#
# Input:
#   t0 ... t11
#
# Target:
#   t12 ... t23
#
# Therefore we need 24 consecutive frames.


all_frames = []

for file in files:

    date = parse_date(file)

    for frame in range(48):

        timestamp = date + timedelta(
            minutes=frame * FRAME_MINUTES
        )

        all_frames.append({
            "timestamp": timestamp,
            "file": str(file),
            "frame": frame,
        })


# ============================================================
# TIMESTAMP MAP
# ============================================================

frame_map = {
    item["timestamp"]: item
    for item in all_frames
}


# ============================================================
# BUILD SEQUENCES
# ============================================================

def build_sequences(start_year, end_year):

    sequences = []

    timestamps = sorted(frame_map.keys())

    for i, start_time in enumerate(timestamps):

        # ----------------------------------------------------
        # Sequence timestamps
        # ----------------------------------------------------

        required_times = [
            start_time
            + timedelta(minutes=j * FRAME_MINUTES)
            for j in range(
                INPUT_FRAMES + OUTPUT_FRAMES
            )
        ]

        # ----------------------------------------------------
        # Must all exist
        # ----------------------------------------------------

        if not all(
            t in frame_map
            for t in required_times
        ):
            continue

        # ----------------------------------------------------
        # Split by target start date
        # ----------------------------------------------------

        target_start = required_times[
            INPUT_FRAMES
        ]

        if not (
            start_year
            <= target_start.year
            <= end_year
        ):
            continue

        # ----------------------------------------------------
        # Ensure entire sequence belongs to same split
        # ----------------------------------------------------

        target_end = required_times[-1]

        if target_end.year != target_start.year:
            continue

        # ----------------------------------------------------
        # File/frame references
        # ----------------------------------------------------

        input_frames = [
            {
                "file": frame_map[t]["file"],
                "frame": frame_map[t]["frame"],
                "timestamp": t.isoformat(),
            }
            for t in required_times[
                :INPUT_FRAMES
            ]
        ]

        target_frames = [
            {
                "file": frame_map[t]["file"],
                "frame": frame_map[t]["frame"],
                "timestamp": t.isoformat(),
            }
            for t in required_times[
                INPUT_FRAMES:
            ]
        ]

        sequences.append({
            "input": input_frames,
            "target": target_frames,
        })

    return sequences


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BUILD EACH SPLIT
# ============================================================

results = {}

for split_name, (start_year, end_year) in SPLITS.items():

    sequences = build_sequences(
        start_year,
        end_year
    )

    output_file = (
        OUTPUT_DIR
        / f"{split_name}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            sequences,
            f,
            indent=2
        )

    results[split_name] = sequences

    print()
    print(
        f"{split_name.upper():12s}: "
        f"{len(sequences):,} sequences"
    )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL B SEQUENCE SUMMARY")
print("=" * 60)

for split_name, sequences in results.items():

    print(
        f"{split_name.upper():12s}: "
        f"{len(sequences):,}"
    )


# ============================================================
# SHOW FIRST SEQUENCE
# ============================================================

for split_name in [
    "train",
    "validation",
    "test"
]:

    sequences = results[split_name]

    if not sequences:
        continue

    first = sequences[0]

    print()
    print(f"FIRST {split_name.upper()} SEQUENCE")

    print(
        "Input:",
        first["input"][0]["timestamp"],
        "->",
        first["input"][-1]["timestamp"]
    )

    print(
        "Target:",
        first["target"][0]["timestamp"],
        "->",
        first["target"][-1]["timestamp"]
    )


print()
print("=" * 60)
print("STATUS: MODEL B SEQUENCE INDEX CREATED")
print("=" * 60)