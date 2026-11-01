import earthaccess
from pathlib import Path

print("=" * 70)
print("GES DISC / IMERG FINAL V07 ACCESS TEST")
print("=" * 70)

# NASA Earthdata login
print("\n[1] Logging into NASA Earthdata...")

auth = earthaccess.login(
    persist=True,
    strategy="interactive"
)

if not auth.authenticated:
    raise RuntimeError("NASA Earthdata login failed.")

print("[OK] Earthdata authentication successful.")


# Search exactly one IMERG Final V07 granule
print("\n[2] Searching for IMERG Final V07...")

results = earthaccess.search_data(
    short_name="GPM_3IMERGHH",
    version="07",
    temporal=(
        "2022-06-01T00:00:00",
        "2022-06-01T00:29:59"
    ),
)

print(f"[OK] Granules found: {len(results)}")

if not results:
    raise RuntimeError(
        "No IMERG Final V07 granule found."
    )


# Show first result
print("\n[3] First granule:")
print(results[0])


# Download exactly one granule
output_dir = Path("data/test_gesdisc")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

print("\n[4] Downloading ONE granule...")
print(f"Output: {output_dir.resolve()}")

files = earthaccess.download(
    results[0],
    local_path=str(output_dir)
)

print("\n[5] RESULT")
print("-" * 70)

for file in files:
    path = Path(file)

    print(f"File : {path}")
    print(
        f"Size : "
        f"{path.stat().st_size / 1024 / 1024:.2f} MB"
    )

print("-" * 70)
print("[SUCCESS] One IMERG Final V07 granule downloaded.")