"""Step 4 - will a vehicle get a new safety recall (filed 2023+), given what was known by end-2022?

Metric: average precision (area under precision-recall curve) - suited to ranking vehicles for review
when positives are the minority. ROC-AUC and precision in the top 10% are reported alongside.
Validation: 5-fold StratifiedGroupKFold grouped by nameplate (make + base model), so model years of
the same vehicle never sit in both train and test. Repeated with 3 different shuffles.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from common import PROCESSED, REPORTS, save_json

GROUPS = ["engine", "electrical", "powertrain", "brakes", "driver_assist", "steering", "fuel_propulsion",
          "air_bags", "structure", "speed_control", "suspension", "visibility", "lighting", "seats_belts",
          "wheels_tires", "unknown"]
F_COMPLAINTS = ["log_complaints", "share_crash", "share_fire", "share_injury", "share_2022"] + [f"share_{g}" for g in GROUPS]
F_SPECS = ["log_configs", "displ_median", "cyl_median", "mpg_comb_median", "share_ev", "share_hybrid",
           "share_turbo", "share_awd", "vehicle_age_2022", "body_CAR", "body_SUV", "body_PICKUP", "body_VAN", "body_WAGON"]
F_PRIOR = ["prior_recalls_2020_2022"]

FEATURE_SETS = {
    "A_complaints_only": F_COMPLAINTS,
    "A2_complaints_plus_vehicle_age": F_COMPLAINTS + ["vehicle_age_2022"],
    "B_complaints_plus_specs": F_COMPLAINTS + F_SPECS,
    "C_complaints_specs_prior_recalls": F_COMPLAINTS + F_SPECS + F_PRIOR,
}
SEEDS = [0, 1, 2]


def precision_at_top(y, p, frac=0.10):
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-p)[:k]
    return float(np.mean(np.asarray(y)[idx]))


def models():
    return {
        "dummy_prevalence": lambda: DummyClassifier(strategy="prior"),
        "baseline_logreg_complaint_volume": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(C=0.5, max_iter=2000)),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=1.0, random_state=0),
    }


def evaluate(df, features, make_model, seeds=SEEDS):
    y, groups = df.label_recall_2023plus.values, df.vehicle_id.values
    X = df[features]
    rows, oof_all = [], []
    for seed in seeds:
        oof = np.zeros(len(df))
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups)):
            m = make_model()
            m.fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            oof[te] = p
            rows.append({"seed": seed, "fold": fold, "ap": average_precision_score(y[te], p),
                         "roc_auc": roc_auc_score(y[te], p), "p_at_top10": precision_at_top(y[te], p)})
        oof_all.append(oof)
    r = pd.DataFrame(rows)
    summary = {k: {"mean": round(r[k].mean(), 4), "std": round(r[k].std(), 4)} for k in ["ap", "roc_auc", "p_at_top10"]}
    summary["_fold_ap"] = r.ap.round(5).tolist()
    return summary, np.mean(oof_all, axis=0)


def main():
    df = pd.read_parquet(PROCESSED / "vehicle_model_table.parquet")
    for b in ["CAR", "SUV", "PICKUP", "VAN", "WAGON"]:
        df[f"body_{b}"] = (df.body == b).astype(int)
    results = {"n_rows": len(df), "n_groups": int(df.vehicle_id.nunique()),
               "positive_rate": round(float(df.label_recall_2023plus.mean()), 4),
               "cv": "StratifiedGroupKFold(5) by nameplate x 3 seeds = 15 folds", "results": {}}
    M = models()

    s, _ = evaluate(df, ["log_complaints"], M["dummy_prevalence"])
    results["results"]["dummy_prevalence"] = s
    s, _ = evaluate(df, ["log_complaints"], M["baseline_logreg_complaint_volume"])
    results["results"]["baseline_complaint_volume_only"] = s
    s, _ = evaluate(df, ["log_complaints", "prior_recalls_2020_2022"], M["baseline_logreg_complaint_volume"])
    results["results"]["baseline_volume_plus_prior_recalls"] = s

    best_oof = None
    for fs_name, feats in FEATURE_SETS.items():
        for mname in ["logreg", "gradient_boosting"]:
            s, oof = evaluate(df, feats, M[mname])
            results["results"][f"{mname}__{fs_name}"] = s
            print(f"{mname:18s} {fs_name:34s} AP={s['ap']['mean']:.3f}±{s['ap']['std']:.3f} AUC={s['roc_auc']['mean']:.3f}")
            if mname == "logreg" and fs_name == "C_complaints_specs_prior_recalls":
                best_oof = oof

    # Paired comparisons: folds are identical across runs (splits depend only on y and groups).
    R = results["results"]
    def paired(a, b):
        d = np.array(R[b]["_fold_ap"]) - np.array(R[a]["_fold_ap"])
        return {"from": a, "to": b, "mean_ap_gain": round(d.mean(), 4), "folds_improved": f"{int((d > 0).sum())}/{len(d)}"}
    results["paired_ap_gains"] = [
        paired("logreg__A_complaints_only", "logreg__A2_complaints_plus_vehicle_age"),
        paired("logreg__A2_complaints_plus_vehicle_age", "logreg__B_complaints_plus_specs"),
        paired("logreg__B_complaints_plus_specs", "logreg__C_complaints_specs_prior_recalls"),
        paired("baseline_volume_plus_prior_recalls", "logreg__C_complaints_specs_prior_recalls"),
    ]
    print(results["paired_ap_gains"])
    for v in R.values():
        v.pop("_fold_ap", None)

    # Final model = logistic regression on set C (as accurate as boosting, simpler to explain).
    # Permutation importance on held-out folds (one seed).
    feats = FEATURE_SETS["C_complaints_specs_prior_recalls"]
    y, groups, X = df.label_recall_2023plus.values, df.vehicle_id.values, df[feats]
    imps = []
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, groups):
        m = M["logreg"]()
        m.fit(X.iloc[tr], y[tr])
        pi = permutation_importance(m, X.iloc[te], y[te], scoring="average_precision", n_repeats=5, random_state=0)
        imps.append(pi.importances_mean)
    imp = pd.Series(np.mean(imps, axis=0), index=feats).sort_values(ascending=False)
    results["final_model"] = "logreg__C_complaints_specs_prior_recalls"
    results["permutation_importance_ap_drop"] = imp.round(4).head(15).to_dict()
    final = M["logreg"]().fit(X, y)
    coef = pd.Series(final[-1].coef_[0], index=feats).sort_values()
    results["final_logreg_standardised_coefficients"] = coef.round(3).to_dict()

    out = df[["MAKE_N", "BASE_N", "make", "base_model", "MODEL_YEAR", "body", "n_complaints",
              "prior_recalls_2020_2022", "n_recalls_2023plus", "label_recall_2023plus"]].copy()
    out["oof_risk_score"] = best_oof
    out.to_parquet(PROCESSED / "vehicle_oof_predictions.parquet", index=False)
    save_json(results, REPORTS / "recall_model_metrics.json")
    print({k: v["ap"] for k, v in results["results"].items()})
    print(results["permutation_importance_ap_drop"])


if __name__ == "__main__":
    main()
