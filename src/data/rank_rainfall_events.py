from pathlib import Path
import csv
import re
from collections import defaultdict


RAW_DIR = Path("data/raw")

SCREENING_FILE = Path(
    "data/processed/rainfall_event_screening.csv"
)

OUTPUT_FILE = Path(
    "data/processed/ranked_rainfall_events.csv"
)


def extract_date(filename):
    """
    Extract YYYYMMDD from IMERG filename.
    """

    match = re.search(
        r"3IMERG\.(\d{8})-",
        filename
    )

    if match:
        return match.group(1)

    return None


def main():

    if not SCREENING_FILE.exists():

        print(
            f"Screening file not found:\n"
            f"{SCREENING_FILE}"
        )

        return

    # --------------------------------------------------------
    # Group available frames by date
    # --------------------------------------------------------

    files_by_date = defaultdict(list)

    files = RAW_DIR.glob(
        "3B-HHR.MS.MRG.3IMERG.*.V07B.HDF5"
    )

    for file_path in files:

        date = extract_date(
            file_path.name
        )

        if date:
            files_by_date[date].append(
                file_path
            )

    # --------------------------------------------------------
    # Read screening results
    # --------------------------------------------------------

    results = []

    with open(
        SCREENING_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            filename = row["file"]

            date = extract_date(
                filename
            )

            row["date"] = date

            row["frames_available"] = len(
                files_by_date.get(
                    date,
                    []
                )
            )

            results.append(row)

    # --------------------------------------------------------
    # Rank by maximum rainfall
    # --------------------------------------------------------

    results.sort(
        key=lambda x: float(
            x["max_rain"]
        ),
        reverse=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "date",
        "file",
        "max_rain",
        "mean_rain",
        "p95_rain",
        "valid_pixels",
        "category",
        "frames_available"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    # --------------------------------------------------------
    # Display top events
    # --------------------------------------------------------

    print("=" * 70)
    print("RAINFALL EVENT RANKING")
    print("=" * 70)

    print(
        f"Total screened frames : {len(results)}"
    )

    print("\nTop rainfall frames:")

    print(
        f"{'Rank':<6}"
        f"{'Date':<12}"
        f"{'Max':<12}"
        f"{'Mean':<12}"
        f"{'P95':<12}"
        f"{'Category':<12}"
    )

    print("-" * 70)

    for i, row in enumerate(
        results[:20],
        start=1
    ):

        print(
            f"{i:<6}"
            f"{row['date']:<12}"
            f"{float(row['max_rain']):<12.2f}"
            f"{float(row['mean_rain']):<12.2f}"
            f"{float(row['p95_rain']):<12.2f}"
            f"{row['category']:<12}"
        )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()