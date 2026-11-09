"""Step 3 - build the vehicle-level modelling table.

Unit of analysis: one light vehicle = EPA (make, base model, model year), model years 2015-2022.
Features come only from information available by 31 Dec 2022:
  * complaints received by NHTSA 2020-01-01 .. 2022-12-31 (linked to the vehicle)
  * recall campaigns filed 2020-2022 for the vehicle
  * EPA specifications
Label: the vehicle is named in at least one NHTSA *vehicle* recall campaign filed in 2023 or later.
Using a hard date cut keeps label information out of the features.
"""
import numpy as np
import pandas as pd

from common import PROCESSED, REPORTS, save_json

FEATURE_END = pd.Timestamp("2022-12-31")
MY_MIN, MY_MAX = 2015, 2022
GROUPS = ["ENGINE", "ELECTRICAL", "POWERTRAIN", "BRAKES", "DRIVER_ASSIST", "STEERING", "FUEL_PROPULSION",
          "AIR_BAGS", "STRUCTURE", "SPEED_CONTROL", "SUSPENSION", "VISIBILITY", "LIGHTING", "SEATS_BELTS",
          "WHEELS_TIRES", "UNKNOWN"]


def main():
    base = pd.read_parquet(PROCESSED / "epa_base_models.parquet")
    cmpl = pd.read_parquet(PROCESSED / "complaints_clean.parquet")
    rcl = pd.read_parquet(PROCESSED / "recall_campaign_vehicles.parquet")
    key = ["MAKE_N", "BASE_N", "MODEL_YEAR"]

    pop = base[(base.MODEL_YEAR >= MY_MIN) & (base.MODEL_YEAR <= MY_MAX)].copy()
    pop["MODEL_YEAR"] = pop.MODEL_YEAR.astype("Int64")

    # ---- complaint features (window 2020-2022) ----
    c = cmpl[cmpl.BASE_N.notna() & (cmpl.RECEIVED_DATE <= FEATURE_END)].copy()
    per_complaint = (c.groupby(key + ["ODINO"])
                       .agg(crash=("CRASH_FLAG", "max"), fire=("FIRE_FLAG", "max"),
                            injured=("INJURED_NUM", "max"), deaths=("DEATHS_NUM", "max"),
                            year_recv=("RECEIVED_DATE", lambda s: s.dt.year.min()))
                       .reset_index())
    agg = (per_complaint.groupby(key)
             .agg(n_complaints=("ODINO", "size"), n_crash=("crash", "sum"), n_fire=("fire", "sum"),
                  n_injury_complaints=("injured", lambda s: (s > 0).sum()), n_deaths=("deaths", "sum"),
                  n_2022=("year_recv", lambda s: (s == 2022).sum()))
             .reset_index())
    comp = (c.drop_duplicates(key + ["ODINO", "COMPONENT_GROUP"])
              .pivot_table(index=key, columns="COMPONENT_GROUP", values="ODINO", aggfunc="count", fill_value=0))
    comp = comp.reindex(columns=GROUPS, fill_value=0).add_prefix("cmp_").reset_index()

    # ---- recall features and label ----
    r = rcl[rcl.BASE_N.notna()]
    prior = r[r.FILED_YEAR <= 2022].groupby(key)["NHTSA ID"].nunique().rename("prior_recalls_2020_2022").reset_index()
    future = r[r.FILED_YEAR >= 2023].groupby(key)["NHTSA ID"].nunique().rename("n_recalls_2023plus").reset_index()

    df = pop.merge(agg, on=key, how="left").merge(comp, on=key, how="left") \
            .merge(prior, on=key, how="left").merge(future, on=key, how="left")
    fill0 = [col for col in df.columns if col.startswith(("n_", "cmp_", "prior_"))]
    df[fill0] = df[fill0].fillna(0)
    df["label_recall_2023plus"] = (df.n_recalls_2023plus > 0).astype(int)

    # ---- derived features ----
    n = df.n_complaints
    df["log_complaints"] = np.log1p(n)
    df["share_crash"] = np.where(n > 0, df.n_crash / n.clip(lower=1), 0)
    df["share_fire"] = np.where(n > 0, df.n_fire / n.clip(lower=1), 0)
    df["share_injury"] = np.where(n > 0, df.n_injury_complaints / n.clip(lower=1), 0)
    df["share_2022"] = np.where(n > 0, df.n_2022 / n.clip(lower=1), 0)  # is complaint volume accelerating?
    for g in GROUPS:
        df[f"share_{g.lower()}"] = np.where(n > 0, df[f"cmp_{g}"] / n.clip(lower=1), 0)
    df["vehicle_age_2022"] = 2022 - df.MODEL_YEAR.astype(int)
    df["log_configs"] = np.log1p(df.n_configs)
    df["vehicle_id"] = df.MAKE_N + "|" + df.BASE_N  # grouping key for CV (all model years of a nameplate)

    df.to_parquet(PROCESSED / "vehicle_model_table.parquet", index=False)
    stats = {
        "rows": len(df), "nameplates": int(df.vehicle_id.nunique()), "makes": int(df.MAKE_N.nunique()),
        "positives": int(df.label_recall_2023plus.sum()),
        "positive_rate_pct": round(100 * df.label_recall_2023plus.mean(), 2),
        "rows_with_zero_complaints": int((df.n_complaints == 0).sum()),
        "complaints_used_in_features": int(agg.n_complaints.sum()),
        "missing_specs": {col: int(df[col].isna().sum()) for col in ["displ_median", "cyl_median", "mpg_comb_median"]},
    }
    save_json(stats, REPORTS / "vehicle_table_stats.json")
    print(stats)


if __name__ == "__main__":
    main()
