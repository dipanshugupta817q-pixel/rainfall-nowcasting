from pathlib import Path
from datetime import datetime, timezone, timedelta
import re

DATA_DIR = Path("data/processed/imerg")

START = datetime(
    2008, 6, 1,
    tzinfo=timezone.utc
)

END = datetime(
    2020, 9, 30,
    tzinfo=timezone.utc
)

TARGET_MONTHS = {6, 7, 8, 9}


def parse_timestamp(filename):

    # Canonical format
    m = re.match(
        r"^(\d{8})_(\d{4})\.npz$",
        filename
    )

    if m:
        return datetime.strptime(
            m.group(1) + m.group(2),
            "%Y%m%d%H%M"
        ).replace(tzinfo=timezone.utc)

    # Old format
    m = re.search(
        r"\.3IMERG\.(\d{8})-S"
        r"(\d{2})(\d{2})(\d{2})-",
        filename
    )

    if m:
        return datetime.strptime(
            m.group(1)
            + m.group(2)
            + m.group(3),
            "%Y%m%d%H%M"
        ).replace(tzinfo=timezone.utc)

    return None


# ---------------------------------------------------------
# Generate expected timeline
# ---------------------------------------------------------

expected = []

day = START

while day <= END:

    if day.month in TARGET_MONTHS:

        day_start = day.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        for i in range(48):

            expected.append(
                day_start +
                timedelta(minutes=30 * i)
            )

    day += timedelta(days=1)


expected_set = set(expected)


# ---------------------------------------------------------
# Read actual timestamps
# ---------------------------------------------------------

actual = []

for file in DATA_DIR.glob("*.npz"):

    # Ignore 2022 test data
    if file.name.startswith("2022"):
        continue

    ts = parse_timestamp(file.name)

    if ts is not None:
        actual.append(ts)


actual_set = set(actual)


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

missing = sorted(
    expected_set - actual_set
)

extra = sorted(
    actual_set - expected_set
)

duplicates = (
    len(actual) - len(actual_set)
)


# ---------------------------------------------------------
# Check 30-minute alignment
# ---------------------------------------------------------

misaligned = [
    ts for ts in actual_set
    if ts.minute not in (0, 30)
    or ts.second != 0
]


# ---------------------------------------------------------
# Check each day has exactly 48 frames
# ---------------------------------------------------------

daily_counts = {}

for ts in actual_set:

    day = ts.date()

    daily_counts[day] = (
        daily_counts.get(day, 0) + 1
    )


bad_days = [
    (day, count)
    for day, count in daily_counts.items()
    if count != 48
]


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 70)
print("             IMERG TIMESTAMP VALIDATION")
print("=" * 70)

print(
    f"Expected timestamps : {len(expected_set):,}"
)

print(
    f"Actual timestamps   : {len(actual_set):,}"
)

print(
    f"Missing timestamps  : {len(missing):,}"
)

print(
    f"Extra timestamps    : {len(extra):,}"
)

print(
    f"Duplicate timestamps: {duplicates:,}"
)

print(
    f"Misaligned timestamps: {len(misaligned):,}"
)

print(
    f"Days with !=48 frames: {len(bad_days):,}"
)

print("=" * 70)


if not missing and not extra and not duplicates and not misaligned and not bad_days:

    print()
    print("STATUS: TIMESTAMP SYNCHRONIZATION VERIFIED")
    print()
    print("Canonical temporal resolution: 30 minutes")
    print("Frames per day: 48")
    print("All timestamps aligned to :00 / :30")
    print()

else:

    print()
    print("STATUS: TIMESTAMP ISSUES FOUND")

    if missing:
        print("\nMissing examples:")
        for x in missing[:10]:
            print(x.isoformat())

    if extra:
        print("\nExtra examples:")
        for x in extra[:10]:
            print(x.isoformat())

    if misaligned:
        print("\nMisaligned examples:")
        for x in misaligned[:10]:
            print(x.isoformat())

    if bad_days:
        print("\nBad-day examples:")
        for x in bad_days[:10]:
            print(x)

print("=" * 70)