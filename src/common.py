"""Shared paths and helpers."""
from pathlib import Path
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
for _p in (INTERIM, PROCESSED, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# Expected SHA-256 of the raw files this project was built on (measured at build time).
RAW_SHA256 = {
    "COMPLAINTS_RECEIVED_2020-2024.zip": "a6db51f5fecd9ee5b8d73e8d5cfc5c6727ca1ba5ab9a4a103346b3090ff91918",
    "COMPLAINTS_RECEIVED_2025-2026.zip": "360165967651af0e8f00623a03acdfe557f073fafe3aaaf229463f82d7253aba",
    "RCL_FROM_2020_2024.zip": "e66dc10edf6de4016f116ef53857cf7a3791e96355fe113ca29ab135dbfa811f",
    "RCL_FROM_2025_2026.zip": "df6a0ad9b16af326eb495b6aaa7222bfc0d58f0af07ce0316b5cf2131b45a1ca",
    "epa_vehicles.csv": "8d57f973234ab5247a40826bb611131fa846e07e3266cdb587b611f8ec329988",
}

# US states + DC + territories + military/freely-associated postal codes (all legitimate).
VALID_STATES = set(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
    "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR VI GU AS MP "
    "AA AE AP FM PW MH".split()
)

# NHTSA changed its component taxonomy over time; the legacy and new labels coexist in the file.
# This maps the top-level label (text before the first ':') to one harmonized group.
COMPONENT_GROUPS = {
    "ENGINE": "ENGINE",
    "ENGINE AND ENGINE COOLING": "ENGINE",
    "ELECTRICAL SYSTEM": "ELECTRICAL",
    "POWER TRAIN": "POWERTRAIN",
    "STEERING": "STEERING",
    "SERVICE BRAKES": "BRAKES",
    "SERVICE BRAKES, HYDRAULIC": "BRAKES",
    "SERVICE BRAKES, AIR": "BRAKES",
    "SERVICE BRAKES, ELECTRIC": "BRAKES",
    "SERVICE BRAKES, HYDRAULIC; AUTOHOLD BRAKE SYSTEM/BRAKE HOLD": "BRAKES",
    "PARKING BRAKE": "BRAKES",
    "FUEL/PROPULSION SYSTEM": "FUEL_PROPULSION",
    "FUEL SYSTEM, GASOLINE": "FUEL_PROPULSION",
    "FUEL SYSTEM, DIESEL": "FUEL_PROPULSION",
    "FUEL SYSTEM, OTHER": "FUEL_PROPULSION",
    "HYBRID PROPULSION SYSTEM": "FUEL_PROPULSION",
    "FORWARD COLLISION AVOIDANCE": "DRIVER_ASSIST",
    "LANE DEPARTURE": "DRIVER_ASSIST",
    "BACK OVER PREVENTION": "DRIVER_ASSIST",
    "AIR BAGS": "AIR_BAGS",
    "SEAT BELTS": "SEATS_BELTS",
    "SEATS": "SEATS_BELTS",
    "STRUCTURE": "STRUCTURE",
    "VEHICLE SPEED CONTROL": "SPEED_CONTROL",
    "SUSPENSION": "SUSPENSION",
    "WHEELS": "WHEELS_TIRES",
    "TIRES": "WHEELS_TIRES",
    "EXTERIOR LIGHTING": "LIGHTING",
    "INTERIOR LIGHTING": "LIGHTING",
    "VISIBILITY/WIPER": "VISIBILITY",
    "VISIBILITY": "VISIBILITY",
    "UNKNOWN OR OTHER": "UNKNOWN",
    "Other/I am not sure": "UNKNOWN",
    "NONE": "UNKNOWN",
    "": "UNKNOWN",
}  # anything not listed -> "OTHER"


def component_group(compdesc: pd.Series) -> pd.Series:
    top = compdesc.fillna("").str.split(":").str[0].str.strip()
    return top.map(COMPONENT_GROUPS).fillna("OTHER")


def norm_key(s: pd.Series) -> pd.Series:
    """Upper-case and strip everything that is not A-Z/0-9 ("F-150" -> "F150", "Mercedes Benz" -> "MERCEDESBENZ")."""
    return s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def clean_model_text(s: pd.Series) -> pd.Series:
    """Remove NHTSA data-entry artefacts seen in MODELTXT (e.g. 'redundant SIENNA')."""
    return s.astype("string").str.replace(r"^\s*redundant\s+", "", regex=True, flags=re.I).str.strip()


# Remainders after a prefix match that indicate a heavier/different vehicle than the EPA base model.
# EPA fuel-economy data does not cover heavy-duty pickups/vans (>8,500 lb GVWR), so 'SILVERADO2500'
# must NOT be linked to EPA's light-duty 'SILVERADO'.
HEAVY_SUFFIX = re.compile(r"^(2500|3500|4500|5500|250|350|450|550|HD|SD|2500HD|3500HD|SUPERDUTY|CHASSIS)")

# Remainders that mean a different powertrain or nameplate, so specs of the shorter base model would be wrong.
DIFFERENT_VEHICLE = re.compile(r"^(EV|BEV|ELECTRIC|LIGHTNING|CONNECT|CITY|ETRON|EQ)")


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
