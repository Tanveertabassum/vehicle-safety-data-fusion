"""Step 5 - small NLP task: predict the affected component group from the complaint narrative.

Why it is useful: 14% of complaints are filed under 'UNKNOWN OR OTHER'. A text classifier can suggest a
component for those, and can pre-fill the component for new complaints.

Data: complaints with exactly one component group, excluding UNKNOWN/OTHER (15 classes).
  * development set: received 2020-2024, stratified sample, 5-fold stratified CV
  * time hold-out:   received 2025-2026 (never seen during development)
Metric: macro-F1 (every component counts equally; accuracy would be dominated by ENGINE/ELECTRICAL).
Baselines: majority class, and class-prior random guessing.
"""
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline

from common import PROCESSED, REPORTS, ROOT, save_json

DEV_SIZE = 60_000
TEST_SIZE = 15_000
SEED = 42


def preprocess(text: pd.Series) -> pd.Series:
    t = text.fillna("").str.lower()
    t = t.str.replace(r"x{2,}", " redacted ", regex=True)  # NHTSA redacts personal data as XXX
    t = t.str.replace(r"\d+", " 0 ", regex=True)
    return t.str.replace(r"\s+", " ", regex=True).str.strip()


def build_pipeline():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, max_features=60_000, sublinear_tf=True),
        LogisticRegression(C=4.0, max_iter=300, class_weight="balanced"),
    )


def main():
    t = pd.read_parquet(PROCESSED / "complaints_text.parquet")
    stats = {"complaints_total": len(t),
             "multi_component": int((t.N_COMPONENTS > 1).sum()),
             "unknown_or_other_single": int(((t.N_COMPONENTS == 1) & t.COMPONENT_GROUP.isin(["UNKNOWN", "OTHER"])).sum())}
    t = t[(t.N_COMPONENTS == 1) & ~t.COMPONENT_GROUP.isin(["UNKNOWN", "OTHER"])].copy()
    t["text"] = preprocess(t.CDESCR)
    short = t.text.str.len() < 20
    stats["dropped_text_under_20_chars"] = int(short.sum())
    t = t[~short]
    stats["eligible"] = len(t)

    dev_pool = t[t.RECEIVED_DATE.dt.year <= 2024]
    test_pool = t[t.RECEIVED_DATE.dt.year >= 2025]
    dev, _ = train_test_split(dev_pool, train_size=min(DEV_SIZE, len(dev_pool) - 1),
                              stratify=dev_pool.COMPONENT_GROUP, random_state=SEED)
    test = test_pool.sample(n=min(TEST_SIZE, len(test_pool)), random_state=SEED)
    stats.update({"dev_rows": len(dev), "test_rows": len(test), "classes": int(dev.COMPONENT_GROUP.nunique()),
                  "dev_class_counts": dev.COMPONENT_GROUP.value_counts().to_dict()})
    X, y = dev.text, dev.COMPONENT_GROUP

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = {}
    for name, est in {"majority_class": DummyClassifier(strategy="most_frequent"),
                      "random_by_class_prior": DummyClassifier(strategy="stratified", random_state=SEED)}.items():
        r = cross_validate(est, X.to_frame(), y, cv=cv, scoring=["f1_macro", "accuracy"])
        res[name] = {"cv_macro_f1": round(r["test_f1_macro"].mean(), 4), "cv_accuracy": round(r["test_accuracy"].mean(), 4)}

    t0 = time.time()
    r = cross_validate(build_pipeline(), X, y, cv=cv, scoring=["f1_macro", "accuracy"], n_jobs=1)
    res["tfidf_logreg"] = {"cv_macro_f1": round(r["test_f1_macro"].mean(), 4), "cv_macro_f1_std": round(r["test_f1_macro"].std(), 4),
                           "cv_accuracy": round(r["test_accuracy"].mean(), 4), "cv_seconds": round(time.time() - t0, 1)}
    print(res)

    # Final fit on the whole development set, evaluate once on the 2025-2026 hold-out
    model = build_pipeline().fit(X, y)
    pred = model.predict(test.text)
    maj = y.value_counts().idxmax()
    res["holdout_2025_2026"] = {
        "tfidf_logreg_macro_f1": round(f1_score(test.COMPONENT_GROUP, pred, average="macro"), 4),
        "tfidf_logreg_accuracy": round(accuracy_score(test.COMPONENT_GROUP, pred), 4),
        "majority_class_macro_f1": round(f1_score(test.COMPONENT_GROUP, [maj] * len(test), average="macro"), 4),
    }
    conf = model.predict_proba(test.text).max(1)
    hi = conf >= 0.6
    res["holdout_2025_2026"]["share_with_confidence_ge_0.6"] = round(float(hi.mean()), 4)
    res["holdout_2025_2026"]["accuracy_when_confidence_ge_0.6"] = round(accuracy_score(test.COMPONENT_GROUP[hi], pred[hi]), 4)
    res["holdout_2025_2026"]["accuracy_when_confidence_lt_0.6"] = round(accuracy_score(test.COMPONENT_GROUP[~hi], pred[~hi]), 4)
    rep = classification_report(test.COMPONENT_GROUP, pred, output_dict=True, zero_division=0)
    res["holdout_per_class_f1"] = {k: round(v["f1-score"], 3) for k, v in rep.items() if k.isupper()}
    labels = sorted(y.unique())
    cm = pd.DataFrame(confusion_matrix(test.COMPONENT_GROUP, pred, labels=labels), index=labels, columns=labels)
    cm.to_csv(REPORTS / "text_confusion_matrix_holdout.csv")
    off = cm.where(~np.eye(len(labels), dtype=bool)).stack().sort_values(ascending=False).head(6)
    res["top_confusions_true_to_pred"] = [f"{a} -> {b}: {int(n)}" for (a, b), n in off.items()]

    # Apply to the complaints filed as UNKNOWN OR OTHER
    unk = pd.read_parquet(PROCESSED / "complaints_text.parquet")
    unk = unk[(unk.N_COMPONENTS == 1) & (unk.COMPONENT_GROUP == "UNKNOWN")].copy()
    unk["text"] = preprocess(unk.CDESCR)
    unk = unk[unk.text.str.len() >= 20]
    proba = model.predict_proba(unk.text)
    unk["pred_component"] = model.classes_[proba.argmax(1)]
    unk["pred_confidence"] = proba.max(1)
    confident = unk[unk.pred_confidence >= 0.6]
    res["unknown_complaints"] = {
        "scored": len(unk),
        "confidence_ge_0.6": len(confident),
        "confidence_ge_0.6_pct": round(100 * len(confident) / len(unk), 1),
        "predicted_distribution_confident": confident.pred_component.value_counts().to_dict(),
    }
    unk.sort_values("pred_confidence", ascending=False).groupby("pred_component").head(5)[
        ["ODINO", "MAKE_N", "MODEL_N", "MODEL_YEAR", "CDESCR", "pred_component", "pred_confidence"]
    ].to_parquet(PROCESSED / "unknown_reclassified_sample.parquet", index=False)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "component_text_clf.joblib", compress=3)
    save_json({"data": stats, "results": res}, REPORTS / "text_model_metrics.json")
    print(res["holdout_2025_2026"], res["unknown_complaints"]["confidence_ge_0.6_pct"])


if __name__ == "__main__":
    main()
