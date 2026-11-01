from pathlib import Path
import re
import shutil

DATA_DIR = Path("data/processed/imerg")
ARCHIVE_DIR = Path("data/processed/imerg_archive")

ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def get_timestamp(filename):

    # New format
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


files = list(DATA_DIR.glob("*.npz"))

new_files = {}
old_files = []
extra_2022 = []

for f in files:

    ts = get_timestamp(f.name)

    if ts is None:
        continue

    # New canonical format
    if re.match(
        r"^\d{8}_\d{4}\.npz$",
        f.name
    ):
        new_files[ts] = f

    else:
        old_files.append(f)


# ---------------------------------------------------------
# Move old duplicate-format files
# ---------------------------------------------------------

moved_duplicates = 0

for f in old_files:

    destination = ARCHIVE_DIR / f.name

    shutil.move(
        str(f),
        str(destination)
    )

    moved_duplicates += 1


# ---------------------------------------------------------
# Move 2022 test files
# ---------------------------------------------------------

for f in list(DATA_DIR.glob("*.npz")):

    if f.name.startswith("2022"):

        destination = ARCHIVE_DIR / f.name

        shutil.move(
            str(f),
            str(destination)
        )

        extra_2022.append(f.name)


print()
print("=" * 65)
print("                 IMERG CLEANUP")
print("=" * 65)

print(
    f"Old duplicate files moved : {moved_duplicates}"
)

print(
    f"2022 test files moved      : "
    f"{len(extra_2022)}"
)

print(
    f"Archive                    : "
    f"{ARCHIVE_DIR}"
)

print()
print(
    "Historical canonical files remain in:"
)

print(
    f"  {DATA_DIR}"
)

print("=" * 65)