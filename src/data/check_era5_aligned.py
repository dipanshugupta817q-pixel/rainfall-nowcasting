from pathlib import Path
from datetime import datetime
import calendar

ROOT = Path("data/processed/era5_aligned")

START_YEAR = 2021
END_YEAR = 2025
MONTHS = [6, 7, 8, 9]

expected = []

for year in range(START_YEAR, END_YEAR + 1):

    for month in MONTHS:

        days_in_month = calendar.monthrange(year, month)[1]

        for day in range(1, days_in_month + 1):

            expected.append(
                f"{year:04d}{month:02d}{day:02d}"
            )


actual = {
    p.stem
    for p in ROOT.glob("*.npz")
}

expected_set = set(expected)

missing = sorted(expected_set - actual)
extra = sorted(actual - expected_set)


print("=" * 60)
print("ERA5 ALIGNED DATA AUDIT")
print("=" * 60)

print(f"Expected days : {len(expected)}")
print(f"Actual files  : {len(actual)}")
print(f"Missing days  : {len(missing)}")
print(f"Extra files   : {len(extra)}")

print("\nMISSING DAYS:")

if missing:
    for day in missing:
        print(day)
else:
    print("NONE")

print("\nEXTRA FILES:")

if extra:
    for day in extra:
        print(day)
else:
    print("NONE")

print("=" * 60)