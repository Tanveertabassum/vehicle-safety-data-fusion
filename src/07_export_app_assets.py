"""Step 7 - export the small files the Streamlit app needs into app/assets/ (a few MB in total)."""
import shutil

import pandas as pd

from common import PROCESSED, REPORTS, ROOT, save_json

ASSETS = ROOT / "app" / "assets"


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in ["dq_metrics.json", "recall_model_metrics.json", "text_model_metrics.json",
                 "vehicle_table_stats.json", "ingest_summary.json"]:
        shutil.copy(REPORTS / name, ASSETS / name)
    shutil.copy(ROOT / "models" / "component_text_clf.joblib", ASSETS / "component_text_clf.joblib")
    shutil.copy(PROCESSED / "vehicle_oof_predictions.parquet", ASSETS / "vehicle_oof_predictions.parquet")
    shutil.copy(PROCESSED / "unknown_reclassified_sample.parquet", ASSETS / "unknown_reclassified_sample.parquet")

    c = pd.read_parquet(PROCESSED / "complaints_clean.parquet",
                        columns=["ODINO", "COMPONENT_GROUP", "RECEIVED_DATE", "BASE_N"])
    trend = (c.drop_duplicates(["ODINO", "COMPONENT_GROUP"])
               .assign(quarter=lambda d: d.RECEIVED_DATE.dt.to_period("Q").astype(str))
               .groupby(["quarter", "COMPONENT_GROUP"]).size().rename("complaints").reset_index())
    trend.to_parquet(ASSETS / "complaints_by_quarter.parquet", index=False)
    save_json({"complaints_received_min": str(c.RECEIVED_DATE.min().date()),
               "complaints_received_max": str(c.RECEIVED_DATE.max().date())}, ASSETS / "data_window.json")
    print("assets:", sorted(p.name for p in ASSETS.iterdir()))


if __name__ == "__main__":
    main()
