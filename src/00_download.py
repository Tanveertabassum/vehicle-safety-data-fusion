"""Step 0 - download the raw files and verify their SHA-256 hashes.

The project was built from public GitHub mirrors of the official files, because the build environment
could not reach nhtsa.gov or fueleconomy.gov. The hashes pin the exact files used, so the reported
numbers are reproducible.

Official sources, to refresh the data:
  NHTSA complaints & recalls: https://www.nhtsa.gov/nhtsa-datasets-and-apis  (U.S. public domain)
  EPA vehicles.csv:           https://www.fueleconomy.gov/feg/download.shtml  (EPA data: public domain)
Newer official files will have different hashes and give different numbers; 01_ingest.py warns about that.
"""
import hashlib
import urllib.request

from common import RAW, RAW_SHA256

NHTSA_MIRROR = "https://raw.githubusercontent.com/DavisM1212/NHTSA-ODI-Complaint-Analytics/HEAD"
FILES = {
    "COMPLAINTS_RECEIVED_2020-2024.zip": f"{NHTSA_MIRROR}/data/raw/COMPLAINTS_RECEIVED_2020-2024.zip",
    "COMPLAINTS_RECEIVED_2025-2026.zip": f"{NHTSA_MIRROR}/data/raw/COMPLAINTS_RECEIVED_2025-2026.zip",
    "RCL_FROM_2020_2024.zip": f"{NHTSA_MIRROR}/data/raw/RCL_FROM_2020_2024.zip",
    "RCL_FROM_2025_2026.zip": f"{NHTSA_MIRROR}/data/raw/RCL_FROM_2025_2026.zip",
    "CMPL.txt": f"{NHTSA_MIRROR}/docs/CMPL.txt",  # NHTSA data dictionary for the complaints file
    "epa_vehicles.csv": "https://raw.githubusercontent.com/jhernandez4/vehicles_dataset/HEAD/vehicles.csv",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists() and (name not in RAW_SHA256 or sha256(dest) == RAW_SHA256[name]):
            print(f"ok (cached)  {name}")
            continue
        print(f"downloading  {name}")
        urllib.request.urlretrieve(url, dest)
        if name in RAW_SHA256:
            got = sha256(dest)
            print(f"  sha256 {'OK' if got == RAW_SHA256[name] else 'MISMATCH: ' + got}")


if __name__ == "__main__":
    main()
