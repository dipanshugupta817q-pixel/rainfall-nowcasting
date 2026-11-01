from pathlib import Path
import re
import shutil

DATA_DIR = Path("data/processed/imerg")
ARCHIVE_DIR = Path("data/processed/imerg_archive")

ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def get_timestamp(filename):

    # Canonical new format
    m = re.match(
        r"^(\d{8})_(\d{4})\.npz$",
        filename
    )

    if m:
        return m.group(1) + "_" + m.group(2)

    # Old format
    m = re.search(
        r"\.3IMERG\.(\d{8})-S"
        r"(\d{2})(\d{2})(\d{2})-",
        filename
    )

    if m:
        return (
            m.group(1)
            + "_"
            + m.group(2)
            + m.group(3)
        )

    return None


# ---------------------------------------------------------
# Find canonical new-format files
# ---------------------------------------------------------

canonical = {}

for f in DATA_DIR.glob("*.npz"):

    if re.match(
        r"^\d{8}_\d{4}\.npz$",
        f.name
    ):
        ts = get_timestamp(f.name)

        if ts:
            canonical[ts] = f


# ---------------------------------------------------------
# Move ONLY old-format files whose timestamp already
# exists as a canonical new-format file.
# ---------------------------------------------------------

moved = 0
checked_old = 0

for f in DATA_DIR.glob("*_regional.npz"):

    checked_old += 1

    ts = get_timestamp(f.name)

    if ts is None:
        continue

    # This is the critical condition:
    # only move if the new-format copy exists.
    if ts in canonical:

        destination = ARCHIVE_DIR / f.name

        if destination.exists():
            print(
                f"Already archived, skipping: {f.name}"
            )
            continue

        shutil.move(
            str(f),
            str(destination)
        )

        moved += 1

print()
print("=" * 65)
print("             SAFE DUPLICATE CLEANUP")
print("=" * 65)

print(
    f"Canonical new-format files : {len(canonical):,}"
)

print(
    f"Old-format files checked    : {checked_old:,}"
)

print(
    f"Duplicate files moved       : {moved:,}"
)

print(
    f"Archive                     : {ARCHIVE_DIR}"
)

print("=" * 65)