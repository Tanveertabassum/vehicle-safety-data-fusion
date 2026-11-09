"""Step 2 - measure data quality, clean, and integrate the three sources.

Every number written to reports/dq_metrics.json is computed here from the data; the data-quality
report is generated from that file, so it cannot contain numbers that were not measured.
"""
import numpy as np
import pandas as pd

from common import (INTERIM, PROCESSED, REPORTS, VALID_STATES, HEAVY_SUFFIX, DIFFERENT_VEHICLE, clean_model_text,
                    component_group, norm_key, save_json)

M = {}  # all measured metrics


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def blank(s):
    return s.isna() | (s.astype("string").str.strip() == "")


# --------------------------------------------------------------------------------------------
# 1. Complaints
# --------------------------------------------------------------------------------------------
def clean_complaints():
    c = pd.read_parquet(INTERIM / "complaints_raw.parquet")
    m = {"raw_rows": len(c), "raw_unique_complaints_odino": int(c.ODINO.nunique())}

    # --- completeness on raw rows (key fields) ---
    key_fields = ["MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC", "CDESCR", "VIN", "STATE",
                  "FAILDATE", "MILES", "DRIVE_TRAIN", "FUEL_TYPE", "NUM_CYLS", "TRANS_TYPE"]
    comp = {}
    for f in key_fields:
        n_blank = int(blank(c[f]).sum())
        comp[f] = {"blank": n_blank, "blank_pct": pct(n_blank, len(c))}
    comp["YEARTXT"]["unknown_9999"] = int((c.YEARTXT == "9999").sum())
    comp["MILES"]["zero"] = int((c.MILES.str.strip() == "0").sum())
    m["completeness_raw"] = comp

    # --- duplicates ---
    m["dup_exact_rows"] = int(c.drop(columns="SOURCE_FILE").duplicated().sum())
    m["dup_cmplid"] = int(c.CMPLID.duplicated().sum())
    m["dup_odino_component"] = int(c.duplicated(["ODINO", "COMPDESC"]).sum())
    m["rows_from_multi_component_complaints"] = int(c.ODINO.duplicated(keep=False).sum())

    # --- scope: motor vehicles only (drop tyres, equipment, child seats) ---
    m["prod_type_counts"] = c.PROD_TYPE.value_counts(dropna=False).to_dict()
    c = c[c.PROD_TYPE == "V"].copy()
    m["rows_after_vehicle_filter"] = len(c)

    # --- resubmitted complaints: same vehicle + same narrative under a new ODINO ---
    first = c.drop_duplicates("ODINO")
    vin_key = first.VIN.where(~blank(first.VIN), "NOVIN_" + first.LDATE)
    dup_mask = first.assign(_k=vin_key).duplicated(["_k", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR"])
    resubmitted = set(first.loc[dup_mask, "ODINO"])
    m["resubmitted_complaints_removed"] = len(resubmitted)
    c = c[~c.ODINO.isin(resubmitted)].copy()

    # --- inconsistent / invalid values ---
    inc = {}
    year = pd.to_numeric(c.YEARTXT, errors="coerce")
    fail = pd.to_datetime(c.FAILDATE, format="%Y%m%d", errors="coerce")
    recv = pd.to_datetime(c.LDATE, format="%Y%m%d", errors="coerce")
    miles = pd.to_numeric(c.MILES.replace("", np.nan), errors="coerce")
    inc["model_year_unknown_9999"] = int((year == 9999).sum())
    inc["failure_after_received_date"] = int((fail > recv).sum())
    inc["failure_before_1990"] = int((fail < "1990-01-01").sum())
    inc["failure_more_than_1y_before_model_year"] = int(((fail.dt.year < year - 1) & (year != 9999)).sum())
    inc["miles_over_500k"] = int((miles > 500_000).sum())
    inc["state_not_valid_postal_code"] = int((~c.STATE.isin(VALID_STATES)).sum())
    inc["make_spellings_raw"] = int(c.MAKETXT.nunique())
    c["MAKE_N"] = norm_key(c.MAKETXT)
    inc["make_spellings_after_normalisation"] = int(c.MAKE_N.nunique())
    inc["make_mercedes_variants"] = c.MAKETXT[c.MAKE_N == "MERCEDESBENZ"].value_counts().to_dict()
    inc["model_redundant_prefix"] = int(c.MODELTXT.str.contains(r"^\s*redundant\s", case=False, regex=True).sum())
    # Narratives with non-ASCII characters (curly quotes, ellipses, accents). Reading the file with the
    # wrong encoding (Latin-1 instead of UTF-8) garbles exactly these, e.g. an ellipsis becomes 'â\x80¦'.
    first_txt = c.drop_duplicates("ODINO").CDESCR
    inc["narratives_with_non_ascii_chars"] = int(first_txt.str.contains(r"[^\x00-\x7f]", regex=True).sum())
    inc["narratives_with_mojibake_after_utf8_read"] = int(first_txt.str.contains("â\x80|Ã©|Â", regex=True).sum())
    inc["component_top_level_labels_raw"] = int(c.COMPDESC.str.split(":").str[0].str.strip().nunique())
    m["inconsistent_values_raw"] = inc

    # --- fixes ---
    c["MODEL_CLEAN"] = clean_model_text(c.MODELTXT)
    c["MODEL_N"] = norm_key(c.MODEL_CLEAN)
    c["MODEL_YEAR"] = year.where(year != 9999).astype("Int64")
    c["FAIL_DATE"] = fail.where((fail <= recv) & (fail >= "1990-01-01"))
    c["RECEIVED_DATE"] = recv
    c["MILES_NUM"] = miles.where((miles > 0) & (miles <= 500_000))  # 0 and blank both mean "not given"
    c["STATE_CLEAN"] = c.STATE.where(c.STATE.isin(VALID_STATES))
    c["COMPONENT_GROUP"] = component_group(c.COMPDESC)
    c["CRASH_FLAG"] = (c.CRASH == "Y").astype(int)
    c["FIRE_FLAG"] = (c.FIRE == "Y").astype(int)
    c["INJURED_NUM"] = pd.to_numeric(c.INJURED, errors="coerce").fillna(0)
    c["DEATHS_NUM"] = pd.to_numeric(c.DEATHS, errors="coerce").fillna(0)

    after = {
        "rows": len(c),
        "unique_complaints": int(c.ODINO.nunique()),
        "model_year_missing": int(c.MODEL_YEAR.isna().sum()),
        "fail_date_missing_or_invalid": int(c.FAIL_DATE.isna().sum()),
        "miles_usable": int(c.MILES_NUM.notna().sum()),
        "state_missing_or_invalid": int(c.STATE_CLEAN.isna().sum()),
        "model_redundant_prefix": int(c.MODEL_CLEAN.str.contains(r"^redundant\s", case=False, regex=True).sum()),
        "make_spellings": int(c.MAKE_N.nunique()),
        "component_groups": int(c.COMPONENT_GROUP.nunique()),
        "component_group_counts": c.COMPONENT_GROUP.value_counts().to_dict(),
    }
    m["after_cleaning"] = after
    M["complaints"] = m

    # Text table for NLP: one row per complaint with its component(s)
    text = (c.groupby("ODINO", sort=False)
              .agg(CDESCR=("CDESCR", "first"), RECEIVED_DATE=("RECEIVED_DATE", "first"),
                   N_COMPONENTS=("COMPONENT_GROUP", "nunique"), COMPONENT_GROUP=("COMPONENT_GROUP", "first"),
                   MAKE_N=("MAKE_N", "first"), MODEL_N=("MODEL_N", "first"), MODEL_YEAR=("MODEL_YEAR", "first"))
              .reset_index())
    text.to_parquet(PROCESSED / "complaints_text.parquet", index=False)

    keep = ["CMPLID", "ODINO", "MAKETXT", "MODEL_CLEAN", "MAKE_N", "MODEL_N", "MODEL_YEAR", "COMPDESC",
            "COMPONENT_GROUP", "CRASH_FLAG", "FIRE_FLAG", "INJURED_NUM", "DEATHS_NUM", "STATE_CLEAN",
            "FAIL_DATE", "RECEIVED_DATE", "MILES_NUM", "CMPL_TYPE"]
    return c[keep].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# 2. Recalls
# --------------------------------------------------------------------------------------------
def clean_recalls():
    r = pd.read_parquet(INTERIM / "recalls_raw.parquet")
    m = {"raw_rows": len(r), "raw_campaigns": int(r["NHTSA ID"].nunique())}
    m["completeness_raw"] = {col: int(blank(r[col]).sum()) for col in ["NHTSA ID", "MAKE", "MODEL", "MODEL YEAR", "SUMMARY"]}
    m["dup_exact_rows"] = int(r.drop(columns="SOURCE_FILE").duplicated().sum())
    m["model_year_unknown_9999"] = int((r["MODEL YEAR"] == "9999").sum())
    m["recall_type_counts_rows"] = r["NHTSA ID"].str[2].value_counts().to_dict()
    pattern_ok = r["NHTSA ID"].str.match(r"^\d{2}[VETC]\d{3}$")
    m["campaign_id_nonstandard_rows"] = int((~pattern_ok).sum())
    m["campaign_id_nonstandard_campaigns"] = int(r.loc[~pattern_ok, "NHTSA ID"].nunique())
    m["docs_per_campaign_median"] = float(r.groupby("NHTSA ID").size().median())
    m["docs_per_campaign_max"] = int(r.groupby("NHTSA ID").size().max())
    m["make_spellings_raw"] = int(r.MAKE.nunique())

    # Fix: exact duplicates out; keep vehicle campaigns with a known model year; collapse documents
    # to one row per campaign x vehicle (make, model, model year).
    r = r.drop_duplicates(subset=["NHTSA ID", "DOCUMENT NAME", "MAKE", "MODEL", "MODEL YEAR", "SUMMARY"])
    r = r[r["NHTSA ID"].str[2] == "V"]
    r = r[r["MODEL YEAR"] != "9999"]
    r["MAKE_N"] = norm_key(r.MAKE)
    r["MODEL_N"] = norm_key(clean_model_text(r.MODEL))
    r["MODEL_YEAR"] = pd.to_numeric(r["MODEL YEAR"], errors="coerce").astype("Int64")
    r["FILED_YEAR"] = 2000 + pd.to_numeric(r["NHTSA ID"].str[:2], errors="coerce")
    cv = r.drop_duplicates(["NHTSA ID", "MAKE_N", "MODEL_N", "MODEL_YEAR"])[
        ["NHTSA ID", "MAKE", "MODEL", "MAKE_N", "MODEL_N", "MODEL_YEAR", "FILED_YEAR"]].reset_index(drop=True)
    m["after_cleaning"] = {
        "campaign_vehicle_rows": len(cv),
        "vehicle_campaigns": int(cv["NHTSA ID"].nunique()),
        "make_spellings": int(cv.MAKE_N.nunique()),
        "campaigns_by_filed_year": cv.drop_duplicates("NHTSA ID").FILED_YEAR.value_counts().sort_index().to_dict(),
    }
    M["recalls"] = m
    return cv


# --------------------------------------------------------------------------------------------
# 3. EPA specifications
# --------------------------------------------------------------------------------------------
def clean_epa():
    e = pd.read_parquet(INTERIM / "epa_raw.parquet")
    m = {"raw_rows": len(e), "model_year_range": [int(e.year.min()), int(e.year.max())],
         "dup_exact_rows": int(e.duplicated().sum()), "dup_id": int(e.id.duplicated().sum())}
    cols = ["make", "model", "baseModel", "year", "VClass", "drive", "cylinders", "displ", "comb08",
            "fuelType1", "atvType", "tCharger", "trany"]
    m["completeness_raw"] = {col: int(e[col].isna().sum()) for col in cols}
    is_ev = e.atvType.eq("EV")
    m["cylinders_missing_and_electric"] = int((e.cylinders.isna() & is_ev).sum())
    m["cylinders_missing_not_electric"] = int((e.cylinders.isna() & ~is_ev).sum())
    m["dup_make_basemodel_year_configs"] = int(e.duplicated(["make", "baseModel", "year"]).sum())

    e = e[e.year >= 2005].copy()
    e["MAKE_N"] = norm_key(e.make)
    e["BASE_N"] = norm_key(e.baseModel)
    e["TRIM_N"] = norm_key(e.model)
    e["is_ev"] = is_ev.loc[e.index].astype(int)
    e["is_hybrid"] = e.atvType.isin(["Hybrid", "Plug-in Hybrid"]).astype(int)
    e["is_turbo"] = e.tCharger.notna().astype(int)  # EPA leaves tCharger blank when not turbocharged
    e["is_awd_4wd"] = e.drive.fillna("").str.contains("4-Wheel|All-Wheel|4-Wheel or All", regex=True).astype(int)
    vclass = e.VClass.fillna("").str.lower()
    e["body"] = np.select(
        [vclass.str.contains("pickup"), vclass.str.contains("sport utility"), vclass.str.contains("van"),
         vclass.str.contains("wagon"), vclass.str.contains("two seat|minicompact|subcompact|compact|midsize|large", regex=True)],
        ["PICKUP", "SUV", "VAN", "WAGON", "CAR"], default="OTHER")

    base = (e.groupby(["MAKE_N", "BASE_N", "year"])
              .agg(make=("make", "first"), base_model=("baseModel", "first"), n_configs=("id", "size"),
                   displ_median=("displ", "median"), cyl_median=("cylinders", "median"),
                   mpg_comb_median=("comb08", "median"), share_ev=("is_ev", "mean"),
                   share_hybrid=("is_hybrid", "mean"), share_turbo=("is_turbo", "mean"),
                   share_awd=("is_awd_4wd", "mean"), body=("body", lambda s: s.mode().iat[0]))
              .reset_index().rename(columns={"year": "MODEL_YEAR"}))
    m["after_cleaning"] = {"base_model_year_rows": len(base), "model_years_kept": "2005+",
                           "rows_kept_2005plus": len(e)}
    M["epa"] = m
    trims = e[["MAKE_N", "TRIM_N", "BASE_N", "year"]].drop_duplicates()
    return base, trims


# --------------------------------------------------------------------------------------------
# 4. Linking NHTSA (make, model, year) -> EPA (make, base model, year)
# --------------------------------------------------------------------------------------------
def build_linker(base, trims):
    base_keys = set(zip(base.MAKE_N, base.BASE_N, base.MODEL_YEAR))
    body = {(a, b, y): bd for a, b, y, bd in base[["MAKE_N", "BASE_N", "MODEL_YEAR", "body"]].itertuples(index=False)}
    # EPA sometimes files two nameplates under one base model ("Legacy/Outback"); index each part.
    alias = {}
    for mk, bm_raw, bm, y in base[["MAKE_N", "base_model", "BASE_N", "MODEL_YEAR"]].itertuples(index=False):
        if "/" in str(bm_raw):
            for part in str(bm_raw).split("/"):
                alias.setdefault((mk, norm_key(pd.Series([part])).iat[0], y), bm)
    trim_map = {(a, b, y): bm for a, b, bm, y in trims.itertuples(index=False)}
    trims_by_make_year = trims.groupby(["MAKE_N", "year"])[["TRIM_N", "BASE_N"]].apply(
        lambda d: list(d.itertuples(index=False, name=None))).to_dict()
    by_make_year = base.groupby(["MAKE_N", "MODEL_YEAR"]).BASE_N.apply(lambda s: sorted(set(s), key=len, reverse=True)).to_dict()

    def link(mk, md, y):
        if pd.isna(y) or not isinstance(md, str) or not md:
            return None, "no_key"
        y = int(y)
        if (mk, md, y) in base_keys:
            return md, "1_exact_base_model"
        if (mk, md, y) in trim_map:
            return trim_map[(mk, md, y)], "2_exact_trim_or_alias"
        if (mk, md, y) in alias:
            return alias[(mk, md, y)], "2_exact_trim_or_alias"
        if md.startswith(mk) and len(md) > len(mk):  # 'MAZDA3' -> '3', 'POLESTAR2' -> '2'
            stripped = md[len(mk):]
            if (mk, stripped, y) in base_keys:
                return stripped, "3_make_prefix_removed"
        for bm in by_make_year.get((mk, y), []):  # longest EPA base model that prefixes the NHTSA model
            if len(bm) >= 3 and md.startswith(bm):
                rest = md[len(bm):]
                # EPA does not cover heavy-duty pickups/vans: 'SILVERADO2500' must not become 'SILVERADO'
                if body.get((mk, bm, y)) in ("PICKUP", "VAN") and HEAVY_SUFFIX.match(rest):
                    return None, "x_heavy_duty_not_in_epa"
                if DIFFERENT_VEHICLE.match(rest):  # 'EQUINOXEV' is not the petrol 'EQUINOX'
                    return None, "x_unmatched"
                return bm, "4_prefix_match"
        if len(md) >= 4:  # NHTSA model prefixes EPA trims of exactly one base model: 'GLE450' -> 'GLE4504MATIC' -> 'GLECLASS'
            hits = {b for t, b in trims_by_make_year.get((mk, y), []) if t.startswith(md)}
            # 'F150' must not become 'F150LIGHTNING', 'TRANSIT' not 'TRANSITCONNECT'
            hits = {b for b in hits if not (b.startswith(md) and b[len(md):] not in ("", "PICKUP", "CLASS", "WAGON"))}
            if len(hits) == 1:
                return hits.pop(), "5_trim_prefix_match"
        return None, "x_unmatched"

    return link


def apply_link(df, link, label):
    keys = df[["MAKE_N", "MODEL_N", "MODEL_YEAR"]].drop_duplicates()
    res = [link(a, b, c) for a, b, c in keys.itertuples(index=False)]
    keys["BASE_N"] = [r[0] for r in res]
    keys["LINK_RULE"] = [r[1] for r in res]
    out = df.merge(keys, on=["MAKE_N", "MODEL_N", "MODEL_YEAR"], how="left")
    rules = out.LINK_RULE.value_counts().to_dict()
    M.setdefault("linking", {})[label] = {
        "rows": len(out), "rule_counts": rules,
        "match_rate_pct_cumulative": {},
    }
    cum = 0
    for rule in ["1_exact_base_model", "2_exact_trim_or_alias", "3_make_prefix_removed", "4_prefix_match", "5_trim_prefix_match"]:
        cum += rules.get(rule, 0)
        M["linking"][label]["match_rate_pct_cumulative"][rule] = pct(cum, len(out))
    return out


def raw_exact_match_rate(cmpl_raw_like, base, case_sensitive):
    f = (lambda s: s.astype(str)) if case_sensitive else (lambda s: s.astype(str).str.upper())
    keys = set(zip(f(base.make), f(base.base_model), base.MODEL_YEAR))
    k = list(zip(f(cmpl_raw_like.MAKETXT), f(cmpl_raw_like.MODEL_CLEAN), cmpl_raw_like.MODEL_YEAR))
    hits = sum(1 for x in k if x in keys)
    return pct(hits, len(k))


def main():
    cmpl = clean_complaints()
    rcl = clean_recalls()
    base, trims = clean_epa()
    link = build_linker(base, trims)

    M.setdefault("linking", {})["complaints_naive_exact_string_match_pct"] = raw_exact_match_rate(cmpl, base, True)
    M["linking"]["complaints_case_insensitive_exact_string_match_pct"] = raw_exact_match_rate(cmpl, base, False)
    cmpl = apply_link(cmpl, link, "complaints")
    rcl = apply_link(rcl, link, "recall_campaign_vehicles")

    # Spot check of fuzzy links for the report (the most frequent prefix matches)
    pm = cmpl[cmpl.LINK_RULE == "4_prefix_match"].groupby(["MAKE_N", "MODEL_N", "BASE_N"]).size().sort_values(ascending=False).head(15)
    M["linking"]["top_prefix_matches"] = [f"{a} {b} -> {c} ({n})" for (a, b, c), n in pm.items()]
    hd = cmpl[cmpl.LINK_RULE == "x_heavy_duty_not_in_epa"].groupby(["MAKE_N", "MODEL_N"]).size().sort_values(ascending=False).head(10)
    M["linking"]["top_heavy_duty_blocked"] = [f"{a} {b} ({n})" for (a, b), n in hd.items()]
    um = cmpl[cmpl.LINK_RULE == "x_unmatched"].groupby(["MAKE_N", "MODEL_N"]).size().sort_values(ascending=False).head(10)
    M["linking"]["top_unmatched"] = [f"{a} {b} ({n})" for (a, b), n in um.items()]

    epa_max_year = int(base.MODEL_YEAR.max())
    unl = cmpl[cmpl.BASE_N.isna()]
    M["linking"]["complaints_unlinked_breakdown"] = {
        "total": len(unl),
        "model_year_after_epa_snapshot_or_partial": int((unl.MODEL_YEAR >= epa_max_year).sum()),
        "epa_snapshot_last_model_year": epa_max_year,
        "heavy_duty_blocked": int((unl.LINK_RULE == "x_heavy_duty_not_in_epa").sum()),
        "make_not_in_epa": int((~unl.MAKE_N.isin(set(base.MAKE_N))).sum()),
    }
    in_scope = rcl.MAKE_N.isin(set(base.MAKE_N)) & (rcl.MODEL_YEAR < epa_max_year)
    M["linking"]["recalls_in_scope"] = {
        "rows_with_epa_make_and_model_year_before_snapshot_end": int(in_scope.sum()),
        "linked_among_those_pct": pct(int(rcl[in_scope].BASE_N.notna().sum()), int(in_scope.sum())),
    }
    cmpl.to_parquet(PROCESSED / "complaints_clean.parquet", index=False)
    rcl.to_parquet(PROCESSED / "recall_campaign_vehicles.parquet", index=False)
    base.to_parquet(PROCESSED / "epa_base_models.parquet", index=False)
    save_json(M, REPORTS / "dq_metrics.json")
    print("linking:", {k: v for k, v in M["linking"].items() if k.startswith("compl") or k.startswith("recall")})


if __name__ == "__main__":
    main()
