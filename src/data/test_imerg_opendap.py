import requests
import earthaccess
from pathlib import Path

# ============================================================
# TEST: TRUE SERVER-SIDE IMERG SUBSETTING
# ============================================================

OUT = Path("data/test_regional_imerg")
OUT.mkdir(parents=True, exist_ok=True)

print("Logging into Earthdata...")
earthaccess.login(persist=True, strategy="interactive")

# Exact granule we already tested
granule_url = (
    "https://data.gesdisc.earthdata.nasa.gov/data/"
    "GPM_L3/GPM_3IMERGHH.07/2022/152/"
    "3B-HHR.MS.MRG.3IMERG.20220601-S000000-E002959.0000.V07B.HDF5"
)

# NASA Earthdata cloud OPeNDAP form
opendap_url = granule_url.replace(
    "https://data.gesdisc.earthdata.nasa.gov/data/",
    "https://opendap.earthdata.nasa.gov/collections/"
)

print("\nOriginal:")
print(granule_url)

print("\nCloud OPeNDAP base:")
print(opendap_url)

print("\nNOTE:")
print("We are first testing the OPeNDAP endpoint itself.")
print("No global HDF5 will be downloaded by this test.")

r = requests.get(
    opendap_url,
    timeout=60
)

print("\nHTTP Status:", r.status_code)
print("Response size:", len(r.content), "bytes")

if r.status_code == 200:
    print("\n[SUCCESS] OPeNDAP endpoint responded.")
    print(r.text[:1000])
else:
    print("\n[FAILED]")
    print(r.text[:1000])