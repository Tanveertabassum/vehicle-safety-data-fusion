"""Streamlit front end. Run:  streamlit run app/streamlit_app.py

Reads only the small files in app/assets/ (produced by src/07_export_app_assets.py).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="Vehicle safety data fusion", page_icon="🚗", layout="wide")


@st.cache_data
def js(name):
    return json.loads((ASSETS / name).read_text())


@st.cache_data
def pq(name):
    return pd.read_parquet(ASSETS / name)


@st.cache_resource
def text_model():
    return joblib.load(ASSETS / "component_text_clf.joblib")


def preprocess(text):  # must match src/05_text_classifier.py
    s = pd.Series([text]).str.lower()
    s = s.str.replace(r"x{2,}", " redacted ", regex=True).str.replace(r"\d+", " 0 ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


dq, rm, tm = js("dq_metrics.json"), js("recall_model_metrics.json"), js("text_model_metrics.json")
vt, win = js("vehicle_table_stats.json"), js("data_window.json")
R = rm["results"]

st.title("Vehicle safety data fusion")
st.caption("NHTSA complaints + NHTSA recalls + EPA vehicle specifications → one dataset, one risk model, "
           "one text classifier. All numbers are read from files the pipeline wrote.")

tab_over, tab_dq, tab_model, tab_text = st.tabs(["Overview", "Data quality & linking", "Recall-risk model", "Complaint text classifier"])

# ------------------------------------------------------------------ Overview
with tab_over:
    lk = dq["linking"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distinct complaints (after cleaning)", f"{dq['complaints']['after_cleaning']['unique_complaints']:,}")
    c2.metric("Complaint rows linked to EPA specs", f"{lk['complaints']['match_rate_pct_cumulative']['5_trim_prefix_match']}%",
              f"naive join: {lk['complaints_naive_exact_string_match_pct']}%")
    best = R["logreg__C_complaints_specs_prior_recalls"]["ap"]["mean"]
    c3.metric("Recall model, average precision", f"{best:.3f}", f"base rate {R['dummy_prevalence']['ap']['mean']:.3f}")
    c4.metric("Text classifier, macro-F1 (2025-26 hold-out)", f"{tm['results']['holdout_2025_2026']['tfidf_logreg_macro_f1']:.3f}")
    st.markdown(f"""
**Question.** Using only what was public by the end of 2022 (owner complaints, earlier recalls, vehicle
specifications), can we rank light vehicles by their chance of a new safety recall in 2023 or later?
And can complaint text be routed to the right component automatically?

**Sources** (all U.S. public domain): NHTSA ODI complaints received {win['complaints_received_min']} to
{win['complaints_received_max']} (tab-delimited flat file), NHTSA recall documents 2020-2026 (CSV), EPA fuel-economy
table, model years up to 2025 (CSV).
""")
    trend = pq("complaints_by_quarter.parquet")
    last_q = pd.Period(win["complaints_received_max"], "Q").strftime("%YQ%q")
    groups = st.multiselect("Complaints per quarter by component group", sorted(trend.COMPONENT_GROUP.unique()),
                            default=["ELECTRICAL", "DRIVER_ASSIST", "POWERTRAIN", "UNKNOWN"])
    chart = trend[trend.COMPONENT_GROUP.isin(groups) & (trend.quarter != last_q)].pivot(
        index="quarter", columns="COMPONENT_GROUP", values="complaints").fillna(0)
    st.line_chart(chart)
    st.caption(f"{last_q} is left out because the file ends on {win['complaints_received_max']}. "
               "Counts are distinct complaints per component group.")

# ------------------------------------------------------------------ Data quality
with tab_dq:
    lk = dq["linking"]
    L = lk["complaints"]["match_rate_pct_cumulative"]
    st.subheader("Linking complaints to EPA specifications")
    funnel = pd.DataFrame({
        "step": ["Naive exact join", "Case-insensitive", "1 Normalised base model", "2 + trim / combined names",
                 "3 + strip repeated make", "4 + base-model prefix", "5 + unique trim prefix"],
        "% of complaint rows linked": [lk["complaints_naive_exact_string_match_pct"], lk["complaints_case_insensitive_exact_string_match_pct"],
                                       L["1_exact_base_model"], L["2_exact_trim_or_alias"], L["3_make_prefix_removed"],
                                       L["4_prefix_match"], L["5_trim_prefix_match"]]})
    st.bar_chart(funnel, x="step", y="% of complaint rows linked", horizontal=True, sort=False)
    st.markdown("Guards stop wrong links: heavy-duty trucks (EPA does not test them) and different vehicles "
                "sharing a prefix (F-150 vs F-150 Lightning, Transit vs Transit Connect).")

    st.subheader("Before / after cleaning (complaints)")
    inc, aft = dq["complaints"]["inconsistent_values_raw"], dq["complaints"]["after_cleaning"]
    st.dataframe(pd.DataFrame([
        ["Resubmitted duplicates (same vehicle, same text, new ID)", dq["complaints"]["resubmitted_complaints_removed"], 0],
        ["Make spellings", inc["make_spellings_raw"], inc["make_spellings_after_normalisation"]],
        ["Component top-level labels", inc["component_top_level_labels_raw"], aft["component_groups"]],
        ["Model text with 'redundant' prefix", inc["model_redundant_prefix"], aft["model_redundant_prefix"]],
        ["Impossible failure dates (after receipt / before 1990)", inc["failure_after_received_date"] + inc["failure_before_1990"], 0],
        ["Mileage > 500,000", inc["miles_over_500k"], 0],
    ], columns=["Issue", "Before", "After"]), hide_index=True, width="stretch")
    st.caption("Full report with completeness, duplicates and residual issues: reports/DATA_QUALITY.md")

# ------------------------------------------------------------------ Recall model
with tab_model:
    st.subheader("Will a vehicle get a new recall filed in 2023 or later?")
    st.markdown(f"{vt['rows']:,} vehicles (make × base model × model year 2015-2022), {vt['positive_rate_pct']}% positive. "
                "Features use only complaints received 2020-2022, recalls filed 2020-2022, and EPA specs. "
                f"Validation: {rm['cv']}, grouped so one nameplate's model years never appear in both train and test.")
    names = {
        "dummy_prevalence": "Baseline: predict base rate",
        "baseline_complaint_volume_only": "Baseline: complaint count only",
        "baseline_volume_plus_prior_recalls": "Baseline: complaint count + prior recalls",
        "logreg__A_complaints_only": "Complaint features",
        "logreg__A2_complaints_plus_vehicle_age": "+ vehicle age",
        "logreg__B_complaints_plus_specs": "+ EPA specifications",
        "logreg__C_complaints_specs_prior_recalls": "+ prior recalls (final)",
        "gradient_boosting__C_complaints_specs_prior_recalls": "Gradient boosting, same features",
    }
    tbl = pd.DataFrame([{"model": v, "avg precision": f"{R[k]['ap']['mean']:.3f} ± {R[k]['ap']['std']:.3f}",
                         "ROC-AUC": f"{R[k]['roc_auc']['mean']:.3f}", "precision in top 10%": f"{R[k]['p_at_top10']['mean']:.3f}"}
                        for k, v in names.items()])
    st.dataframe(tbl, hide_index=True, width="stretch")
    gains = rm["paired_ap_gains"]
    st.markdown("**Paired fold comparison** (same 15 folds): " + "; ".join(
        f"{names.get(g['to'], g['to'])}: {g['mean_ap_gain']:+.3f} AP, better in {g['folds_improved']} folds" for g in gains))

    imp = pd.Series(rm["permutation_importance_ap_drop"]).rename("drop in AP when shuffled")
    st.markdown("**Which features matter** (permutation importance on held-out folds)")
    st.bar_chart(imp, horizontal=True, sort="-drop in AP when shuffled")

    st.markdown("**Out-of-fold risk scores** (each vehicle scored by a model that never saw its nameplate)")
    oof = pq("vehicle_oof_predictions.parquet")
    makes = st.multiselect("Make", sorted(oof.make.unique()), default=[])
    view = oof[oof.make.isin(makes)] if makes else oof
    st.dataframe(view.sort_values("oof_risk_score", ascending=False)[
        ["make", "base_model", "MODEL_YEAR", "body", "n_complaints", "prior_recalls_2020_2022", "oof_risk_score", "label_recall_2023plus"]
    ].rename(columns={"n_complaints": "complaints 2020-22", "prior_recalls_2020_2022": "recalls 2020-22",
                      "oof_risk_score": "risk score", "label_recall_2023plus": "actually recalled 2023+"}).head(200),
        hide_index=True, width="stretch")
    st.caption("A risk score is a ranking aid for review, not a defect finding. Complaint volume partly reflects sales volume, which is not public.")

# ------------------------------------------------------------------ Text classifier
with tab_text:
    T = tm["results"]
    st.subheader("Route a complaint to a component group from its text")
    c1, c2, c3 = st.columns(3)
    c1.metric("CV macro-F1 (2020-24)", f"{T['tfidf_logreg']['cv_macro_f1']:.3f}",
              f"majority baseline {T['majority_class']['cv_macro_f1']:.3f}")
    h = T["holdout_2025_2026"]
    c2.metric("Hold-out macro-F1 (2025-26)", f"{h['tfidf_logreg_macro_f1']:.3f}")
    c3.metric("Accuracy when confidence ≥ 0.6", f"{h['accuracy_when_confidence_ge_0.6']:.3f}",
              f"{100 * h['share_with_confidence_ge_0.6']:.0f}% of hold-out complaints")
    text = st.text_area("Complaint narrative", height=120,
                        value="While driving on the highway the car suddenly braked hard by itself. "
                              "There was nothing in front of me. The collision warning came on for no reason.")
    if text.strip():
        model = text_model()
        proba = pd.Series(model.predict_proba(preprocess(text))[0], index=model.classes_).sort_values(ascending=False)
        st.write(f"**Prediction: {proba.index[0]}** (confidence {proba.iloc[0]:.2f})")
        st.bar_chart(proba.head(5).rename("probability"), horizontal=True, sort="-probability")
    st.markdown("**Per-class F1 on the 2025-26 hold-out**")
    st.bar_chart(pd.Series(T["holdout_per_class_f1"]).rename("F1"), horizontal=True, sort="-F1")
    st.caption("Most confusions: " + "; ".join(T["top_confusions_true_to_pred"][:4]))
    u = T["unknown_complaints"]
    st.markdown(f"**Applied to complaints filed as 'UNKNOWN OR OTHER':** {u['scored']:,} scored, "
                f"{u['confidence_ge_0.6_pct']}% get a suggestion with confidence ≥ 0.6. These have no true label, "
                "so accuracy on them is unmeasured and likely lower than on the labelled hold-out.")
    sample = pq("unknown_reclassified_sample.parquet")
    st.dataframe(sample[["pred_component", "pred_confidence", "MAKE_N", "MODEL_N", "MODEL_YEAR", "CDESCR"]]
                 .sort_values(["pred_component", "pred_confidence"], ascending=[True, False]),
                 hide_index=True, width="stretch")
