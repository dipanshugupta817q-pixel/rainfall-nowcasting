from pathlib import Path
import numpy as np
import re
from datetime import datetime, timezone

DATA_DIR = Path("data/processed/imerg")


def get_timestamp(filename):

    # New format
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
            m.group(1) + m.group(2) + m.group(3),
            "%Y%m%d%H%M"
        ).replace(tzinfo=timezone.utc)

    return None


files_by_timestamp = {}

for f in DATA_DIR.glob("*.npz"):

    ts = get_timestamp(f.name)

    if ts is not None:
        files_by_timestamp.setdefault(
            ts, []
        ).append(f)


duplicate_groups = [
    (ts, files)
    for ts, files in files_by_timestamp.items()
    if len(files) > 1
]


print()
print("=" * 70)
print("             DUPLICATE DATA VERIFICATION")
print("=" * 70)

print(
    f"Duplicate timestamp groups: "
    f"{len(duplicate_groups)}"
)

identical = 0
different = 0

different_examples = []


for ts, files in duplicate_groups:

    # Usually one new-format and one old-format file
    new_files = [
        f for f in files
        if re.match(
            r"^\d{8}_\d{4}\.npz$",
            f.name
        )
    ]

    old_files = [
        f for f in files
        if "_regional.npz" in f.name
    ]

    if not new_files or not old_files:
        continue

    new_file = new_files[0]
    old_file = old_files[0]

    try:

        new_data = np.load(
            new_file,
            allow_pickle=False
        )

        old_data = np.load(
            old_file,
            allow_pickle=False
        )

        same_precip = np.array_equal(
            new_data["precipitation"],
            old_data["precipitation"],
            equal_nan=True
        )

        same_lat = np.array_equal(
            new_data["lat"],
            old_data["lat"],
            equal_nan=True
        )

        same_lon = np.array_equal(
            new_data["lon"],
            old_data["lon"],
            equal_nan=True
        )

        if same_precip and same_lat and same_lon:

            identical += 1

        else:

            different += 1

            if len(different_examples) < 10:
                different_examples.append(
                    (
                        ts,
                        new_file.name,
                        old_file.name
                    )
                )

    except Exception as e:

        different += 1

        if len(different_examples) < 10:
            different_examples.append(
                (
                    ts,
                    new_file.name,
                    f"ERROR: {e}"
                )
            )


print()
print(
    f"Identical duplicate groups : {identical}"
)

print(
    f"Different duplicate groups  : {different}"
)

if different_examples:

    print()
    print("Examples requiring review:")

    for item in different_examples:
        print()
        print(item[0].isoformat())
        print(" NEW:", item[1])
        print(" OLD:", item[2])

print()
print("=" * 70)