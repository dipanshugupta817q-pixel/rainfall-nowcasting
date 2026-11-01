from pathlib import Path
from datetime import datetime, timezone, timedelta
import re

DATA_DIR = Path("data/processed/imerg")

found = {}
bad = []
duplicates = []

# ---------------------------------------------------------
# Read all NPZ filenames
# ---------------------------------------------------------

for f in DATA_DIR.glob("*.npz"):

    # New format:
    # YYYYMMDD_HHMM.npz
    m = re.match(r"^(\d{8})_(\d{4})\.npz$", f.name)

    if m:
        ts = datetime.strptime(
            m.group(1) + m.group(2),
            "%Y%m%d%H%M"
        ).replace(tzinfo=timezone.utc)

    else:
        # Old format
        m = re.search(
            r"\.3IMERG\.(\d{8})-S"
            r"(\d{2})(\d{2})(\d{2})-",
            f.name
        )

        if m:
            ts = datetime.strptime(
                m.group(1) + m.group(2) + m.group(3),
                "%Y%m%d%H%M"
            ).replace(tzinfo=timezone.utc)

        else:
            bad.append(f.name)
            continue

    if ts in found:
        duplicates.append(
            (ts, found[ts], f.name)
        )
    else:
        found[ts] = f.name


# ---------------------------------------------------------
# Generate expected timestamps
# ---------------------------------------------------------

START = datetime(
    2008, 6, 1,
    tzinfo=timezone.utc
)

END = datetime(
    2020, 9, 30,
    tzinfo=timezone.utc
)

expected = set()

day = START

while day <= END:

    if day.month in {6, 7, 8, 9}:

        day_start = day.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        for i in range(48):
            expected.add(
                day_start +
                timedelta(minutes=30 * i)
            )

    day += timedelta(days=1)


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

missing = sorted(expected - set(found))
extra = sorted(set(found) - expected)

print()
print("=" * 60)
print("             IMERG DATASET INTEGRITY")
print("=" * 60)

print(f"NPZ files on disk       : {len(list(DATA_DIR.glob('*.npz'))):,}")
print(f"Recognized timestamps   : {len(found):,}")
print(f"Expected timestamps     : {len(expected):,}")
print(f"Missing frames          : {len(missing):,}")
print(f"Extra frames            : {len(extra):,}")
print(f"Duplicate timestamps    : {len(duplicates):,}")
print(f"Unrecognized filenames  : {len(bad):,}")

print("=" * 60)

if missing:
    print("\nFIRST 20 MISSING:")
    for x in missing[:20]:
        print(x.isoformat())

if extra:
    print("\nFIRST 20 EXTRA:")
    for x in extra[:20]:
        print(x.isoformat())

if duplicates:
    print("\nFIRST 10 DUPLICATES:")
    for x in duplicates[:10]:
        print(x)

if bad:
    print("\nFIRST 20 UNRECOGNIZED FILES:")
    for x in bad[:20]:
        print(x)

print()

if (
    len(missing) == 0
    and len(duplicates) == 0
):
    print("STATUS: DATASET COVERAGE COMPLETE")
else:
    print("STATUS: REVIEW REQUIRED")

print("=" * 60)