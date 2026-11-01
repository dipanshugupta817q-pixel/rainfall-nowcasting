from pathlib import Path
from datetime import datetime, timedelta
import re
import json


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed/imerg")

OUTPUT_DIR = Path("data/processed/sequences")

INPUT_FRAMES = 12          # Previous 6 hours
OUTPUT_FRAMES = 12         # Next 6 hours
FRAME_MINUTES = 30

TOTAL_FRAMES = INPUT_FRAMES + OUTPUT_FRAMES


# Model A split
TRAIN_START = datetime(2008, 1, 1)
TRAIN_END = datetime(2020, 12, 31, 23, 59)

VAL_START = datetime(2021, 1, 1)
VAL_END = datetime(2024, 12, 31, 23, 59)

TEST_START = datetime(2025, 1, 1)
TEST_END = datetime(2025, 12, 31, 23, 59)


# Known 2022 test-day files that are outside the
# historical training dataset
EXCLUDED_DATES = {
    datetime(2022, 6, 1).date()
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("       MODEL A — SPATIO-TEMPORAL SEQUENCE BUILDER")
print("=" * 70)

print()
print("MODEL A SPLIT")
print("-" * 70)
print("TRAIN      : 2008–2020")
print("VALIDATION : 2021–2024")
print("TEST       : 2025")
print()


# ============================================================
# FILENAME TIMESTAMP PARSERS
# ============================================================

# Canonical:
# 20080601_0000.npz

canonical_pattern = re.compile(
    r"^(\d{8})_(\d{4})\.npz$"
)


# Original IMERG:
# 3B-HHR.MS.MRG.3IMERG.20080601-S000000-E002959.0000.V07B.HDF5.npz

imerg_pattern = re.compile(
    r"\.3IMERG\.(\d{8})-S(\d{6})-"
)


# ============================================================
# SCAN DATASET
# ============================================================

timestamp_to_file = {}
unrecognized = []
duplicate_timestamps = []


all_files = list(DATA_DIR.glob("*.npz"))

print(f"NPZ files found: {len(all_files):,}")
print()
print("Scanning filenames...")


for file in all_files:

    name = file.name
    timestamp = None

    # --------------------------------------------------------
    # Canonical format
    # --------------------------------------------------------

    match = canonical_pattern.match(name)

    if match:

        date_str, time_str = match.groups()

        try:
            timestamp = datetime.strptime(
                date_str + time_str,
                "%Y%m%d%H%M"
            )

        except ValueError:
            timestamp = None

    # --------------------------------------------------------
    # Original IMERG format
    # --------------------------------------------------------

    if timestamp is None:

        match = imerg_pattern.search(name)

        if match:

            date_str, time_str = match.groups()

            try:
                timestamp = datetime.strptime(
                    date_str + time_str[:4],
                    "%Y%m%d%H%M"
                )

            except ValueError:
                timestamp = None

    # --------------------------------------------------------
    # Unrecognized
    # --------------------------------------------------------

    if timestamp is None:

        unrecognized.append(name)
        continue

    # --------------------------------------------------------
    # Exclude known 2022 test files
    # --------------------------------------------------------

    if timestamp.date() in EXCLUDED_DATES:
        continue

    # --------------------------------------------------------
    # Duplicate timestamp
    # --------------------------------------------------------

    if timestamp in timestamp_to_file:

        duplicate_timestamps.append(
            (
                timestamp,
                timestamp_to_file[timestamp],
                str(file)
            )
        )

        continue

    timestamp_to_file[timestamp] = str(file)


# ============================================================
# TIMELINE
# ============================================================

timestamps = sorted(timestamp_to_file.keys())


print()
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    f"Files on disk             : "
    f"{len(all_files):,}"
)

print(
    f"Recognized unique frames  : "
    f"{len(timestamps):,}"
)

print(
    f"Unrecognized files        : "
    f"{len(unrecognized):,}"
)

print(
    f"Duplicate timestamps      : "
    f"{len(duplicate_timestamps):,}"
)

print()


# ============================================================
# BASIC VALIDATION
# ============================================================

if unrecognized:

    print("WARNING: Unrecognized filenames detected.")

    for name in unrecognized[:10]:
        print(" ", name)

    print()


if duplicate_timestamps:

    print("WARNING: Duplicate timestamps detected.")

    for item in duplicate_timestamps[:10]:
        print(
            " ",
            item[0],
            "->",
            item[1],
            "|",
            item[2]
        )

    print()


# ============================================================
# BUILD CONTIGUOUS SEQUENCES
# ============================================================

def build_sequences(
    timestamps,
    timestamp_to_file,
    start_date,
    end_date
):

    sequences = []

    for i in range(
        len(timestamps) - TOTAL_FRAMES + 1
    ):

        window = timestamps[
            i:i + TOTAL_FRAMES
        ]

        # ----------------------------------------------------
        # Sequence must lie entirely inside requested split
        # ----------------------------------------------------

        if window[0] < start_date:
            continue

        if window[-1] > end_date:
            continue

        # ----------------------------------------------------
        # Check 30-minute continuity
        # ----------------------------------------------------

        contiguous = True

        for j in range(1, len(window)):

            expected = (
                window[j - 1]
                + timedelta(minutes=FRAME_MINUTES)
            )

            if window[j] != expected:

                contiguous = False
                break

        if not contiguous:
            continue

        # ----------------------------------------------------
        # Input / target
        # ----------------------------------------------------

        input_timestamps = window[:INPUT_FRAMES]

        target_timestamps = window[
            INPUT_FRAMES:
        ]

        sequences.append(
            {
                "input": [
                    timestamp_to_file[t]
                    for t in input_timestamps
                ],

                "target": [
                    timestamp_to_file[t]
                    for t in target_timestamps
                ],

                "input_start":
                    input_timestamps[0].isoformat(),

                "input_end":
                    input_timestamps[-1].isoformat(),

                "target_start":
                    target_timestamps[0].isoformat(),

                "target_end":
                    target_timestamps[-1].isoformat(),
            }
        )

    return sequences


# ============================================================
# BUILD EACH SPLIT
# ============================================================

print("=" * 70)
print("BUILDING MODEL A SPLITS")
print("=" * 70)


train_sequences = build_sequences(
    timestamps,
    timestamp_to_file,
    TRAIN_START,
    TRAIN_END
)


validation_sequences = build_sequences(
    timestamps,
    timestamp_to_file,
    VAL_START,
    VAL_END
)


test_sequences = build_sequences(
    timestamps,
    timestamp_to_file,
    TEST_START,
    TEST_END
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_json(
    filename,
    sequences
):

    output_file = OUTPUT_DIR / filename

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

    return output_file


train_file = save_json(
    "train.json",
    train_sequences
)


validation_file = save_json(
    "validation.json",
    validation_sequences
)


test_file = save_json(
    "test.json",
    test_sequences
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("MODEL A SEQUENCE SUMMARY")
print("=" * 70)

print(
    f"Input frames              : "
    f"{INPUT_FRAMES}"
)

print(
    f"Target frames             : "
    f"{OUTPUT_FRAMES}"
)

print(
    f"Input duration            : "
    f"{INPUT_FRAMES * FRAME_MINUTES / 60:.1f} hours"
)

print(
    f"Forecast duration         : "
    f"{OUTPUT_FRAMES * FRAME_MINUTES / 60:.1f} hours"
)

print()

print(
    f"TRAIN sequences            : "
    f"{len(train_sequences):,}"
)

print(
    f"VALIDATION sequences       : "
    f"{len(validation_sequences):,}"
)

print(
    f"TEST sequences             : "
    f"{len(test_sequences):,}"
)

print()

print(
    f"Total sequences            : "
    f"{len(train_sequences) + len(validation_sequences) + len(test_sequences):,}"
)

print()

print("OUTPUT FILES")
print("-" * 70)

print(
    f"Train      : {train_file}"
)

print(
    f"Validation : {validation_file}"
)

print(
    f"Test       : {test_file}"
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

if train_sequences:

    print()
    print("=" * 70)
    print("FIRST TRAIN SEQUENCE")
    print("=" * 70)

    first = train_sequences[0]

    print(
        f"Input  : "
        f"{first['input_start']} "
        f"→ "
        f"{first['input_end']}"
    )

    print(
        f"Target : "
        f"{first['target_start']} "
        f"→ "
        f"{first['target_end']}"
    )


if validation_sequences:

    print()
    print("=" * 70)
    print("FIRST VALIDATION SEQUENCE")
    print("=" * 70)

    first = validation_sequences[0]

    print(
        f"Input  : "
        f"{first['input_start']} "
        f"→ "
        f"{first['input_end']}"
    )

    print(
        f"Target : "
        f"{first['target_start']} "
        f"→ "
        f"{first['target_end']}"
    )


if test_sequences:

    print()
    print("=" * 70)
    print("FIRST TEST SEQUENCE")
    print("=" * 70)

    first = test_sequences[0]

    print(
        f"Input  : "
        f"{first['input_start']} "
        f"→ "
        f"{first['input_end']}"
    )

    print(
        f"Target : "
        f"{first['target_start']} "
        f"→ "
        f"{first['target_end']}"
    )


# ============================================================
# STATUS
# ============================================================

print()
print("=" * 70)

if (
    len(train_sequences) > 0
    and len(validation_sequences) > 0
    and len(test_sequences) > 0
):

    print(
        "STATUS: MODEL A SEQUENCE SPLITS CREATED"
    )

else:

    print(
        "STATUS: CHECK DATASET / SPLIT"
    )

print("=" * 70)