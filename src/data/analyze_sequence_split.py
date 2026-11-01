from pathlib import Path
import json
from collections import Counter

INDEX_FILE = Path(
    "data/processed/sequence_index.json"
)

with open(
    INDEX_FILE,
    "r",
    encoding="utf-8"
) as f:
    sequences = json.load(f)

print("=" * 70)
print("             SEQUENCE TEMPORAL ANALYSIS")
print("=" * 70)

print(
    f"Total sequences : {len(sequences):,}"
)

# ---------------------------------------------------------
# Count by target year
# ---------------------------------------------------------

target_years = Counter()

for seq in sequences:

    year = int(
        seq["target_start"][:4]
    )

    target_years[year] += 1

print()
print("=" * 70)
print("SEQUENCES BY TARGET YEAR")
print("=" * 70)

for year in sorted(target_years):

    print(
        f"{year} : "
        f"{target_years[year]:,}"
    )

# ---------------------------------------------------------
# Check whether any sequence crosses years
# ---------------------------------------------------------

cross_year = []

for seq in sequences:

    input_year = seq["input_start"][:4]
    target_year = seq["target_end"][:4]

    if input_year != target_year:

        cross_year.append(seq)

print()
print("=" * 70)
print("CROSS-YEAR SEQUENCES")
print("=" * 70)

print(
    f"Cross-year sequences : "
    f"{len(cross_year):,}"
)

if cross_year:

    print()
    print("Examples:")

    for seq in cross_year[:5]:

        print(
            seq["input_start"],
            "→",
            seq["target_end"]
        )

print("=" * 70)

# ---------------------------------------------------------
# Proposed split
# ---------------------------------------------------------

train = sum(
    count
    for year, count in target_years.items()
    if 2008 <= year <= 2017
)

validation = sum(
    count
    for year, count in target_years.items()
    if 2018 <= year <= 2019
)

test = sum(
    count
    for year, count in target_years.items()
    if year == 2020
)

print()
print("=" * 70)
print("PROPOSED CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"TRAIN       2008–2017 : {train:,}"
)

print(
    f"VALIDATION  2018–2019 : {validation:,}"
)

print(
    f"TEST        2020      : {test:,}"
)

print("=" * 70)