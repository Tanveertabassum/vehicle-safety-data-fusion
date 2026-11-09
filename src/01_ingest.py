"""Step 1 - ingest the three raw sources into typed-as-string parquet files, unchanged.

Sources and formats:
  * NHTSA ODI complaints  - headerless, TAB-delimited flat file inside ZIP; column names come from
                            a separate data dictionary (CMPL.txt). Latin-1 encoded.
  * NHTSA recall documents - quoted, comma-delimited CSV with header inside ZIP; one row per
                            recall *document*, so one campaign appears many times.
  * EPA fuel-economy table - wide CSV (84 columns), one row per tested vehicle configuration.

Nothing is cleaned here; this step only parses and records what came in.
"""
import hashlib
import io
import sys
import zipfile

import pandas as pd

from common import RAW, INTERIM, RAW_SHA256, REPORTS, save_json
from schema import CMPL_COLUMNS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify():
    out = {}
    for name, expected in RAW_SHA256.items():
        p = RAW / name
        if not p.exists():
            sys.exit(f"Missing {p}. Run `python src/00_download.py` first.")
        got = sha256(p)
        out[name] = {"sha256": got, "matches_expected": got == expected}
        if got != expected:
            print(f"WARNING: {name} differs from the build-time file; numbers in the report may change.")
    return out


def read_complaints():
    parts = []
    for name in ["COMPLAINTS_RECEIVED_2020-2024.zip", "COMPLAINTS_RECEIVED_2025-2026.zip"]:
        with zipfile.ZipFile(RAW / name) as zf, zf.open(zf.namelist()[0]) as f:
            reader = pd.read_csv(
                io.TextIOWrapper(f, encoding="latin-1"), sep="\t", header=None, names=CMPL_COLUMNS,
                dtype=str, quoting=3, na_filter=False, chunksize=100_000,
            )
            for chunk in reader:
                chunk = chunk.drop(columns=["DEALER_NAME", "DEALER_TEL", "DEALER_CITY", "DEALER_ZIP", "CITY"])
                chunk["SOURCE_FILE"] = name
                parts.append(chunk.astype("string[pyarrow]"))
    return pd.concat(parts, ignore_index=True)


def read_recalls():
    parts = []
    for name in ["RCL_FROM_2020_2024.zip", "RCL_FROM_2025_2026.zip"]:
        with zipfile.ZipFile(RAW / name) as zf, zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(f, dtype=str, encoding="latin-1", keep_default_na=False)
            df["SOURCE_FILE"] = name
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main():
    hashes = verify()
    cmpl = read_complaints()
    cmpl.to_parquet(INTERIM / "complaints_raw.parquet", index=False)
    rcl = read_recalls()
    rcl.to_parquet(INTERIM / "recalls_raw.parquet", index=False)
    epa = pd.read_csv(RAW / "epa_vehicles.csv", low_memory=False)
    epa.to_parquet(INTERIM / "epa_raw.parquet", index=False)
    summary = {
        "hashes": hashes,
        "complaints": {"rows": len(cmpl), "columns": len(CMPL_COLUMNS)},
        "recalls": {"rows": len(rcl), "columns": rcl.shape[1] - 1},
        "epa": {"rows": len(epa), "columns": epa.shape[1]},
    }
    save_json(summary, REPORTS / "ingest_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
