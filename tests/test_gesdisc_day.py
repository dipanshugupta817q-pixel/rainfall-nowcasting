import earthaccess
from pathlib import Path
import time

print("=" * 70)
print("GES DISC / IMERG FINAL V07 - 1 DAY DOWNLOAD TEST")
print("=" * 70)

print("\n[1] Logging into NASA Earthdata...")

auth = earthaccess.login(
    persist=True,
    strategy="interactive"
)

if not auth.authenticated:
    raise RuntimeError("Earthdata authentication failed.")

print("[OK] Authentication successful.")

print("\n[2] Searching 2022-06-01...")

results = earthaccess.search_data(
    short_name="GPM_3IMERGHH",
    version="07",
    temporal=(
        "2022-06-01T00:00:00",
        "2022-06-01T23:59:59"
    ),
)

print(f"[OK] Granules found: {len(results)}")

if len(results) != 48:
    print("[WARNING] Expected 48 granules for one full day.")

output_dir = Path("data/test_gesdisc_day")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

print(f"\n[3] Download directory:")
print(output_dir.resolve())

print("\n[4] Starting download...")
print("Threads: 4")

start = time.perf_counter()

files = earthaccess.download(
    results,
    local_path=str(output_dir),
    threads=4
)

elapsed = time.perf_counter() - start

total_bytes = sum(
    Path(file).stat().st_size
    for file in files
)

total_mb = total_bytes / 1024 / 1024
speed = total_mb / elapsed if elapsed > 0 else 0

print("\n" + "=" * 70)
print("DOWNLOAD RESULT")
print("=" * 70)

print(f"Files downloaded : {len(files)}")
print(f"Total size       : {total_mb:.2f} MB")
print(f"Time             : {elapsed / 60:.2f} minutes")
print(f"Average speed    : {speed:.2f} MB/s")

print("=" * 70)

if len(files) == 48:
    print("[SUCCESS] Full 24-hour IMERG dataset downloaded.")
else:
    print("[WARNING] Not all 48 granules were downloaded.")